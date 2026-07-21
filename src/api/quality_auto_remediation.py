"""
品質ヘルス異常 → スクレイプキュー / DB同期 / megu 再計算の自動修復。

品質チェック完了後、またはバッチ/API から plan を生成し ScrapeJobQueue へ投入する。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.api.quality_health import CHECK_TYPES, get_health_view, load_health

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REMEDIATION_DIR = _PROJECT_ROOT / "data/local/meta/quality_remediation"

REMEDIATION_PRIORITY = 450_000
COOLDOWN_HOURS = 24

# presence: GCS カテゴリ → 再取得タスク
TRACK_CATEGORY_TO_RACE_TASK: dict[str, str] = {
    "race_shutuba": "race_shutuba",
    "race_shutuba_meta": "race_shutuba",
    "race_index": "race_index",
    "race_paddock": "race_paddock",
    "race_odds": "race_odds",
    "race_result_on_time": "race_result_on_time",
    "race_result_on_time_payoff": "race_result_on_time",
    "race_result_on_time_lap": "race_result_on_time",
    "race_result_on_time_corner": "race_result_on_time",
    "race_result": "race_result",
    "race_result_meta": "race_result",
    "race_result_payoff": "race_result",
    "race_result_track": "race_result",
    "race_result_corner": "race_result",
    "race_result_lap_times": "race_result_lap",
    "race_result_lap": "race_result_lap",
    "race_barometer": "race_barometer",
}

# raw_content: gap → 再取得タスク
GAP_TO_RACE_TASKS: dict[str, list[str]] = {
    "gcs_missing": ["race_result", "race_result_on_time"],
    "gcs_bad_meta": [],  # inline backfill
    "gcs_no_finishers": ["race_result", "race_result_on_time"],
    "gcs_no_lap_times": ["race_result_lap", "race_result_on_time"],
    "shutuba_no_entries": ["race_shutuba"],
}


def auto_remediate_enabled() -> bool:
    raw = os.environ.get("KEIBA_QUALITY_AUTO_REMEDIATE", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    # 未設定時: STG のみ有効
    return os.environ.get("KEIBA_ENV", "").strip().lower() == "stg"


def inline_sync_enabled() -> bool:
    raw = os.environ.get("KEIBA_QUALITY_REMEDIATE_SYNC", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def inline_compute_enabled() -> bool:
    raw = os.environ.get("KEIBA_QUALITY_REMEDIATE_COMPUTE", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _now_jst_iso() -> str:
    jst = timezone(timedelta(hours=9))
    return datetime.now(jst).isoformat(timespec="seconds")


def _remediation_path(date: str) -> Path:
    return REMEDIATION_DIR / date[:4] / f"{date}.json"


def load_remediation(date: str) -> dict | None:
    path = _remediation_path(date)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("remediation load failed [%s]: %s", date, e)
        return None


def save_remediation(date: str, payload: dict) -> dict:
    path = _remediation_path(date)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["date"] = date
    payload["updated_at"] = _now_jst_iso()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _should_skip_remediation(date: str, *, force: bool) -> bool:
    if force:
        return False
    prev = load_remediation(date)
    if not prev:
        return False
    at = prev.get("enqueued_at") or prev.get("updated_at")
    if not at:
        return False
    try:
        dt = datetime.fromisoformat(at.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone(timedelta(hours=9)))
        age = datetime.now(timezone(timedelta(hours=9))) - dt.astimezone(timezone(timedelta(hours=9)))
        return age < timedelta(hours=COOLDOWN_HOURS)
    except Exception:
        return False


def _merge_race_tasks(acc: dict[str, set[str]], race_id: str, tasks: list[str]) -> None:
    if not race_id or not tasks:
        return
    acc.setdefault(race_id, set()).update(tasks)


def _tasks_for_categories(categories: list[str]) -> list[str]:
    out: list[str] = []
    for cat in categories:
        task = TRACK_CATEGORY_TO_RACE_TASK.get(cat)
        if task and task not in out:
            out.append(task)
    return out


def _is_actionable_check(rec: dict) -> bool:
    reason = (rec.get("summary") or {}).get("reason")
    if reason in ("no_meeting", "pending", "missing_race_list"):
        return False
    st = rec.get("status") or "unknown"
    return st in ("warn", "fail")


def build_remediation_plan(date: str, health: dict | None = None) -> dict:
    """品質ヘルス JSON から修復アクション一覧を生成。"""
    from src.scraper.date_coverage import load_not_available
    from src.scraper.storage import HybridStorage
    from src.utils.race_quality_rules import (
        OBSTACLE_PRESENCE_NA_CATEGORIES,
        is_obstacle_race,
    )

    storage = HybridStorage()
    obstacle_cache: dict[str, bool] = {}

    def _is_obstacle(rid: str) -> bool:
        if rid not in obstacle_cache:
            rr = storage.load("race_result", rid)
            shutuba = storage.load("race_shutuba", rid) if not rr else None
            obstacle_cache[rid] = is_obstacle_race(rr, shutuba)
        return obstacle_cache[rid]

    view = get_health_view(date) if health is None else {
        "date": date,
        "checks": {
            k: dict(v)
            for k, v in (health.get("checks") or {}).items()
        },
    }
    checks = view.get("checks") or {}

    scrape_tasks: dict[str, set[str]] = {}
    backfill_meta: set[str] = set()
    sync_race_ids: set[str] = set()
    reasons: dict[str, list[str]] = {}
    notes: list[str] = []

    def note(race_id: str, reason: str) -> None:
        reasons.setdefault(race_id, [])
        if reason not in reasons[race_id]:
            reasons[race_id].append(reason)

    year = date[:4]

    presence = checks.get("presence") or {}
    if _is_actionable_check(presence):
        for issue in presence.get("issues") or []:
            rid = str(issue.get("race_id") or "")
            if _is_obstacle(rid):
                continue
            kind = issue.get("kind")
            if kind == "gcs_missing":
                cats = [
                    c for c in (issue.get("categories") or [])
                    if c not in OBSTACLE_PRESENCE_NA_CATEGORIES
                    and rid not in load_not_available(c, year)
                ]
                tasks = _tasks_for_categories(cats)
                if tasks:
                    _merge_race_tasks(scrape_tasks, rid, tasks)
                    note(rid, f"presence:gcs_missing:{','.join(cats[:3])}")
            elif kind == "pg_missing":
                sync_race_ids.add(rid)
                note(rid, "presence:pg_missing")

    raw = checks.get("raw_content") or {}
    if _is_actionable_check(raw):
        for issue in raw.get("issues") or []:
            rid = str(issue.get("race_id") or "")
            if _is_obstacle(rid):
                continue
            for gap in issue.get("gaps") or []:
                if gap == "gcs_bad_meta":
                    backfill_meta.add(rid)
                    note(rid, "raw:gcs_bad_meta")
                    continue
                if gap == "gcs_no_lap_times":
                    rr = storage.load("race_result", rid)
                    if rr and rr.get("lap_times"):
                        continue
                tasks = GAP_TO_RACE_TASKS.get(gap, [])
                if tasks:
                    _merge_race_tasks(scrape_tasks, rid, tasks)
                    note(rid, f"raw:{gap}")

    calculated = checks.get("calculated") or {}
    calc_reason = (calculated.get("summary") or {}).get("reason")
    needs_compute = False
    needs_date_scrape = False

    if _is_actionable_check(calculated):
        if calc_reason == "no_pg_finishers":
            needs_date_scrape = True
            notes.append("calculated:no_pg_finishers → date_results")
        else:
            needs_compute = True
            for issue in calculated.get("issues") or []:
                rid = str(issue.get("race_id") or "")
                if rid:
                    note(rid, "calculated:megu_incomplete")

    actions: list[dict[str, Any]] = []
    for rid, tasks in sorted(scrape_tasks.items()):
        actions.append({
            "kind": "scrape_race",
            "race_id": rid,
            "tasks": sorted(tasks),
            "reasons": reasons.get(rid, []),
        })
    for rid in sorted(backfill_meta):
        actions.append({
            "kind": "backfill_meta",
            "race_id": rid,
            "reasons": reasons.get(rid, ["raw:gcs_bad_meta"]),
        })
    for rid in sorted(sync_race_ids):
        actions.append({
            "kind": "sync_db",
            "race_id": rid,
            "reasons": reasons.get(rid, ["presence:pg_missing"]),
        })
    if needs_date_scrape:
        actions.append({
            "kind": "scrape_date",
            "date": date,
            "tasks": ["date_results"],
            "reasons": ["calculated:no_pg_finishers"],
        })
    if needs_compute:
        actions.append({
            "kind": "compute_date",
            "date": date,
            "reasons": ["calculated:megu_incomplete"],
        })

    actionable = any(_is_actionable_check(checks.get(ct) or {}) for ct in CHECK_TYPES)
    return {
        "date": date,
        "generated_at": _now_jst_iso(),
        "actionable": actionable and bool(actions),
        "action_count": len(actions),
        "actions": actions,
        "notes": notes,
        "check_summary": {
            ct: {
                "status": (checks.get(ct) or {}).get("status"),
                "issues_count": (checks.get(ct) or {}).get("issues_count", 0),
            }
            for ct in CHECK_TYPES
        },
    }


def _run_backfill_meta(race_ids: list[str], storage: Any) -> dict:
    from src.scripts.data.megu_index_scrape_readiness_audit import audit_gcs_race_result
    from src.utils.race_card_merge import patch_result_metadata_from_shutuba

    stats = {"processed": len(race_ids), "saved": 0, "skipped": 0, "error": 0}
    for rid in race_ids:
        try:
            rr = storage.load("race_result", rid, bypass_cache=True)
            rs = storage.load("race_shutuba", rid, bypass_cache=True)
            if not rr:
                stats["skipped"] += 1
                continue
            patched = patch_result_metadata_from_shutuba(rr, rs)
            if not patched:
                stats["skipped"] += 1
                continue
            before = audit_gcs_race_result(rr)
            after = audit_gcs_race_result(patched)
            if after and len(after) >= len(before):
                stats["skipped"] += 1
                continue
            if storage.save("race_result", rid, patched):
                storage.invalidate_load_cache("race_result", rid)
                stats["saved"] += 1
            else:
                stats["error"] += 1
        except Exception as e:
            stats["error"] += 1
            logger.warning("backfill_meta failed %s: %s", rid, e)
    return stats


def _run_sync_db(race_ids: list[str], *, limit: int = 50) -> dict:
    from src.scripts.data.etl_stg_db import process_race, _get_engine
    from sqlalchemy.orm import Session

    ids = race_ids[:limit]
    stats = {"processed": len(ids), "ok": 0, "error": 0}
    if not ids:
        return stats
    engine = _get_engine()
    with Session(engine) as session:
        for rid in ids:
            try:
                r = process_race(session, rid)
                session.commit()
                if r.get("status") == "ok":
                    stats["ok"] += 1
                else:
                    stats["error"] += 1
            except Exception as e:
                session.rollback()
                stats["error"] += 1
                logger.warning("sync_db failed %s: %s", rid, e)
    return stats


def _run_compute_date(date: str) -> dict:
    from src.pipeline.megu_index.compute import compute_for_date

    d = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    try:
        r = compute_for_date(d)
        return {"status": r.get("status"), "detail": r}
    except Exception as e:
        logger.warning("compute_date failed %s: %s", date, e)
        return {"status": "error", "error": str(e)}


def apply_remediation_plan(
    plan: dict,
    *,
    dry_run: bool = False,
    kick: bool = True,
    storage: Any | None = None,
) -> dict:
    """修復プランを実行（メタ補完 → キュー投入 → DB同期 → megu 計算）。"""
    date = plan.get("date") or ""
    actions = plan.get("actions") or []
    if not actions:
        return {"status": "skipped", "reason": "no_actions", "date": date}

    if storage is None:
        from src.scraper.storage import HybridStorage
        storage = HybridStorage()

    result: dict[str, Any] = {
        "date": date,
        "dry_run": dry_run,
        "backfill_meta": None,
        "scrape_queue": None,
        "sync_db": None,
        "compute_date": None,
    }

    backfill_ids = [a["race_id"] for a in actions if a.get("kind") == "backfill_meta"]
    scrape_specs: list[dict] = []
    for a in actions:
        if a.get("kind") != "scrape_race":
            continue
        scrape_specs.append({
            "job_kind": "race",
            "target_id": a["race_id"],
            "tasks": a.get("tasks") or ["race_result"],
            "priority": REMEDIATION_PRIORITY,
            "overwrite": True,
            "smart_skip": False,
            "date": date,
            "source": "quality_auto_remediation",
            "remediation_reasons": a.get("reasons") or [],
        })

    date_scrape = [a for a in actions if a.get("kind") == "scrape_date"]
    for a in date_scrape:
        scrape_specs.append({
            "job_kind": "date",
            "target_id": a.get("date") or date,
            "tasks": a.get("tasks") or ["date_results"],
            "priority": REMEDIATION_PRIORITY,
            "overwrite": True,
            "smart_skip": False,
            "source": "quality_auto_remediation",
            "remediation_reasons": a.get("reasons") or [],
        })

    sync_ids = [a["race_id"] for a in actions if a.get("kind") == "sync_db"]
    compute_actions = [a for a in actions if a.get("kind") == "compute_date"]

    if dry_run:
        result["status"] = "dry_run"
        result["backfill_meta"] = {"race_ids": backfill_ids}
        result["scrape_queue"] = {"jobs": len(scrape_specs), "specs_sample": scrape_specs[:5]}
        result["sync_db"] = {"race_ids": sync_ids}
        result["compute_date"] = [a.get("date") for a in compute_actions]
        return result

    if backfill_ids:
        result["backfill_meta"] = _run_backfill_meta(backfill_ids, storage)

    if scrape_specs:
        from src.scraper.job_queue import ScrapeJobQueue
        queue = ScrapeJobQueue()
        result["scrape_queue"] = queue.bulk_add_jobs(scrape_specs)

    if sync_ids and inline_sync_enabled():
        result["sync_db"] = _run_sync_db(sync_ids)

    if compute_actions and inline_compute_enabled():
        cd = compute_actions[0].get("date") or date
        result["compute_date"] = _run_compute_date(cd)

    if kick and scrape_specs:
        from src.scraper.job_queue import kick_process_queue_background
        kick_process_queue_background()

    result["status"] = "applied"
    result["enqueued_at"] = _now_jst_iso()
    return result


def remediate_date(
    date: str,
    *,
    dry_run: bool = False,
    force: bool = False,
    kick: bool = True,
    storage: Any | None = None,
) -> dict:
    """1開催日分の品質修復（plan 生成 → 実行 → 記録）。"""
    health = load_health(date)
    if not health:
        return {"status": "skipped", "reason": "no_health", "date": date}

    if _should_skip_remediation(date, force=force):
        return {"status": "skipped", "reason": "cooldown", "date": date}

    plan = build_remediation_plan(date, health)
    if not plan.get("actionable"):
        return {"status": "skipped", "reason": "nothing_to_do", "date": date, "plan": plan}

    apply_result = apply_remediation_plan(plan, dry_run=dry_run, kick=kick, storage=storage)
    record = {
        **plan,
        "apply_result": apply_result,
        "enqueued_at": apply_result.get("enqueued_at"),
    }
    if not dry_run:
        save_remediation(date, record)
    return {"status": apply_result.get("status"), "date": date, "plan": plan, "apply": apply_result}


def maybe_remediate_after_checks(date: str, *, kick: bool = True) -> dict | None:
    """品質チェック完了後に呼ぶ。全チェックが揃い warn/fail なら修復を試行。"""
    if not auto_remediate_enabled():
        return None

    health = load_health(date)
    if not health:
        return None

    checks = health.get("checks") or {}
    if len(checks) < len(CHECK_TYPES):
        return None

    view = get_health_view(date)
    overall = view.get("overall_display_status") or view.get("overall_status")
    if overall in ("ok", "na"):
        return None

    return remediate_date(date, kick=kick)
