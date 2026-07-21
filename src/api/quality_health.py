"""
品質ヘルス — 開催日単位のチェック結果 JSON と 3 段チェック実行。

保存: data/local/meta/quality_health/{YYYY}/{YYYYMMDD}.json
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import text

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
QUALITY_HEALTH_DIR = _PROJECT_ROOT / "data/local/meta/quality_health"

CHECK_TYPES = ("presence", "raw_content", "calculated")
HealthStatus = Literal["ok", "warn", "fail", "unknown"]
STALE_DAYS = 7

_STATUS_RANK = {"ok": 0, "warn": 1, "fail": 2, "unknown": 3}


def _now_jst_iso() -> str:
    jst = timezone(timedelta(hours=9))
    return datetime.now(jst).isoformat(timespec="seconds")


def _health_path(date: str) -> Path:
    return QUALITY_HEALTH_DIR / date[:4] / f"{date}.json"


def load_health(date: str) -> dict | None:
    path = _health_path(date)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("quality_health load failed [%s]: %s", date, e)
        return None


def save_health(date: str, payload: dict) -> dict:
    import fcntl

    path = _health_path(date)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".lock")
    payload["date"] = date
    payload["overall_status"] = compute_overall_status(payload.get("checks") or {})
    payload["overall_checked_at"] = _now_jst_iso()

    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        existing: dict = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
        merged_checks = dict(existing.get("checks") or {})
        merged_checks.update(payload.get("checks") or {})
        payload["checks"] = merged_checks
        payload["overall_status"] = compute_overall_status(merged_checks)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return payload


def _parse_checked_at(iso: str | None) -> datetime | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except Exception:
        return None


def is_stale(checked_at: str | None, *, stale_days: int = STALE_DAYS) -> bool:
    dt = _parse_checked_at(checked_at)
    if dt is None:
        return False
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone(timedelta(hours=9)))
    age = datetime.now(timezone(timedelta(hours=9))) - dt.astimezone(timezone(timedelta(hours=9)))
    return age > timedelta(days=stale_days)


def compute_overall_status(checks: dict[str, dict]) -> HealthStatus:
    if not checks:
        return "unknown"
    worst: HealthStatus = "ok"
    for rec in checks.values():
        st = rec.get("status") or "unknown"
        if st not in _STATUS_RANK:
            st = "unknown"
        if _STATUS_RANK[st] > _STATUS_RANK[worst]:
            worst = st  # type: ignore[assignment]
        if rec.get("stale") and _STATUS_RANK[worst] < _STATUS_RANK["warn"]:
            worst = "warn"
    return worst


def enrich_check_record(rec: dict) -> dict:
    rec = dict(rec)
    checked = rec.get("checked_at")
    rec["stale"] = is_stale(checked)
    reason = (rec.get("summary") or {}).get("reason")
    base = rec.get("status") or "unknown"
    if reason == "no_meeting":
        rec["display_status"] = "na"
    elif reason == "pending":
        rec["display_status"] = "ok"
    elif rec.get("stale") and base == "ok":
        rec["display_status"] = "warn"
    else:
        rec["display_status"] = base
    return rec


def get_health_view(date: str) -> dict:
    raw = load_health(date) or {"date": date, "checks": {}}
    checks = {k: enrich_check_record(v) for k, v in (raw.get("checks") or {}).items()}
    overall = compute_overall_status(checks)
    display = overall
    if overall == "unknown" and all(
        (c.get("summary") or {}).get("reason") == "no_meeting"
        for c in checks.values()
    ):
        display = "na"
    elif overall == "ok" and any(c.get("stale") for c in checks.values()):
        display = "warn"
    return {
        "date": date,
        "overall_status": overall,
        "overall_display_status": display,
        "overall_checked_at": raw.get("overall_checked_at"),
        "checks": checks,
    }


def load_year_health(year: int) -> dict[str, dict]:
    year_dir = QUALITY_HEALTH_DIR / str(year)
    if not year_dir.exists():
        return {}
    out: dict[str, dict] = {}
    for path in sorted(year_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            view = get_health_view(path.stem)
            out[path.stem] = {
                "date": path.stem,
                "overall_status": view["overall_display_status"],
                "overall_checked_at": view.get("overall_checked_at"),
                "checks": {
                    k: {
                        "status": v.get("display_status"),
                        "checked_at": v.get("checked_at"),
                        "stale": v.get("stale"),
                    }
                    for k, v in view.get("checks", {}).items()
                },
            }
        except Exception:
            pass
    return out


def _status_from_pct(pct: float, *, has_data: bool) -> HealthStatus:
    if not has_data:
        return "unknown"
    if pct >= 100:
        return "ok"
    if pct >= 80:
        return "warn"
    return "fail"


def _check_result_for_opening_date(date: str, *, check_name: str) -> dict | None:
    """非開催・未来開催向けの早期 return（race_lists 正本）。"""
    from datetime import date as date_cls

    from src.scraper.monitor_future_eligible import jst_today
    from src.utils.race_list_for_date import opening_date_kind

    kind = opening_date_kind(date)
    if kind == "no_meeting":
        return {
            "status": "unknown",
            "checked_at": _now_jst_iso(),
            "summary": {"races": 0, "reason": "no_meeting"},
            "issues_count": 0,
            "issues": [],
            "note": "非開催日（race_lists 空 / no_race_scheduled）",
        }
    if kind == "missing":
        return {
            "status": "unknown",
            "checked_at": _now_jst_iso(),
            "summary": {"races": 0, "reason": "missing_race_list"},
            "issues_count": 0,
            "issues": [],
        }

    try:
        rd = date_cls(int(date[:4]), int(date[4:6]), int(date[6:8]))
    except ValueError:
        rd = None
    if rd and rd > jst_today() and check_name == "calculated":
        from src.utils.race_list_for_date import load_jra_race_ids_for_opening_date

        n = len(load_jra_race_ids_for_opening_date(date))
        return {
            "status": "ok",
            "checked_at": _now_jst_iso(),
            "summary": {"races": n, "reason": "pending", "megu_coverage_pct": 0},
            "issues_count": 0,
            "issues": [],
            "note": "未来開催 — 結果・めぐ指数は未確定",
        }
    return None


def run_presence_check(date: str, storage: Any) -> dict:
    early = _check_result_for_opening_date(date, check_name="presence")
    if early:
        return early

    from src.api.monitor_coverage import build_raw_matrix

    data = build_raw_matrix(date, view="stg", storage=storage)
    race_ids = data.get("race_ids") or []
    if not race_ids:
        return {
            "status": "unknown",
            "checked_at": _now_jst_iso(),
            "summary": {"races": 0, "gcs_pct": 0, "pg_pct": 0},
            "issues_count": 0,
            "issues": [],
        }

    gcs = data.get("gcs") or {}
    cats = gcs.get("categories") or []
    matrix = gcs.get("matrix") or {}
    total_cell = 0
    ok_cell = 0
    for rid in race_ids:
        row = matrix.get(rid) or {}
        for c in cats:
            v = row.get(c)
            if v is None:
                continue
            total_cell += 1
            if v is True:
                ok_cell += 1
    gcs_pct = round(ok_cell / total_cell * 100, 1) if total_cell else 0.0

    pg = (data.get("postgresql") or {}).get("matrix") or {}
    pg_ok = sum(1 for rid in race_ids if (pg.get(rid) or {}).get("pg_races") and (pg.get(rid) or {}).get("pg_race_results", 0) > 0)
    pg_pct = round(pg_ok / len(race_ids) * 100, 1) if race_ids else 0.0

    issues: list[dict] = []
    for rid in race_ids:
        row = matrix.get(rid) or {}
        missing = [c for c in cats if row.get(c) is False]
        if missing:
            issues.append({"race_id": rid, "kind": "gcs_missing", "categories": missing[:5]})
        pg_row = pg.get(rid) or {}
        if not pg_row.get("pg_races") or (pg_row.get("pg_race_results") or 0) == 0:
            issues.append({"race_id": rid, "kind": "pg_missing"})

    combined_pct = min(gcs_pct, pg_pct)
    status = _status_from_pct(combined_pct, has_data=True)
    if gcs_pct < 100 or pg_pct < 100:
        status = "warn" if combined_pct >= 80 else "fail"

    return {
        "status": status,
        "checked_at": _now_jst_iso(),
        "summary": {
            "races": len(race_ids),
            "gcs_pct": gcs_pct,
            "pg_pct": pg_pct,
        },
        "issues_count": len(issues),
        "issues": issues[:50],
    }


def run_raw_content_check(date: str, storage: Any) -> dict:
    early = _check_result_for_opening_date(date, check_name="raw_content")
    if early:
        return early

    from src.api.monitor_coverage import _load_race_ids_for_date
    from src.utils.race_quality_rules import audit_gcs_race_result_for_health, is_obstacle_race

    race_ids = _load_race_ids_for_date(date)
    if not race_ids:
        return {
            "status": "unknown",
            "checked_at": _now_jst_iso(),
            "summary": {"races": 0, "anomalies": 0},
            "issues_count": 0,
            "issues": [],
        }

    issues: list[dict] = []
    skipped_obstacle = 0
    for rid in race_ids:
        rr = storage.load("race_result", rid)
        shutuba = storage.load("race_shutuba", rid)
        obstacle = is_obstacle_race(rr, shutuba)
        if obstacle:
            skipped_obstacle += 1
            continue
        gaps = audit_gcs_race_result_for_health(rr, obstacle=False)
        if gaps:
            issues.append({"race_id": rid, "gaps": gaps})
            continue
        entries = (shutuba or {}).get("entries") or []
        if not entries:
            issues.append({"race_id": rid, "gaps": ["shutuba_no_entries"]})

    eligible = len(race_ids) - skipped_obstacle
    anomaly_n = len(issues)
    pct_ok = round((eligible - anomaly_n) / eligible * 100, 1) if eligible else 100.0
    if eligible == 0 or anomaly_n == 0:
        status: HealthStatus = "ok"
    elif pct_ok >= 95:
        status = "warn"
    else:
        status = "fail"

    return {
        "status": status,
        "checked_at": _now_jst_iso(),
        "summary": {
            "races": len(race_ids),
            "eligible_races": eligible,
            "skipped_obstacle": skipped_obstacle,
            "anomalies": anomaly_n,
            "pct_ok": pct_ok,
        },
        "issues_count": anomaly_n,
        "issues": issues[:50],
    }


def run_calculated_check(date: str) -> dict:
    early = _check_result_for_opening_date(date, check_name="calculated")
    if early:
        return early

    from src.api.monitor_coverage import (
        _load_race_ids_for_date,
        _query_pg_calculated_for_date,
    )

    race_ids = _load_race_ids_for_date(date)
    if not race_ids:
        return {
            "status": "unknown",
            "checked_at": _now_jst_iso(),
            "summary": {"races": 0, "megu_coverage_pct": 0},
            "issues_count": 0,
            "issues": [],
        }

    try:
        from src.db.session import get_session, init_engine

        init_engine()
        with get_session() as session:
            matrix = _query_pg_calculated_for_date(session, date, race_ids)
    except Exception as e:
        logger.warning("calculated check PG error [%s]: %s", date, e)
        return {
            "status": "unknown",
            "checked_at": _now_jst_iso(),
            "summary": {"races": len(race_ids), "error": str(e)},
            "issues_count": 0,
            "issues": [],
            "error": str(e),
        }

    from src.pipeline.megu_index.flat_metadata import is_obstacle_race_name

    issues: list[dict] = []
    ok_n = 0
    eligible = 0
    skipped_obstacle = 0
    reason: str | None = None
    for rid in race_ids:
        row = matrix.get(rid) or {}
        fin = int(row.get("finisher_count") or 0)
        if fin <= 0:
            continue
        surf = (row.get("surface") or "").strip()
        if is_obstacle_race_name(row.get("race_name")) or surf not in ("芝", "ダート"):
            skipped_obstacle += 1
            continue
        eligible += 1
        if row.get("megu_coverage_ok"):
            ok_n += 1
        else:
            issues.append({
                "race_id": rid,
                "finishers": fin,
                "megu_v2": int(row.get("megu_v2") or 0),
                "flat_local": bool(row.get("flat_local")),
            })

    if eligible == 0:
        status: HealthStatus = "fail"
        pct = 0.0
        reason = "no_pg_finishers" if skipped_obstacle == 0 else None
    else:
        pct = round(ok_n / eligible * 100, 1)
        status = "ok" if pct >= 100 else ("warn" if pct >= 90 else "fail")

    return {
        "status": status,
        "checked_at": _now_jst_iso(),
        "summary": {
            "races": len(race_ids),
            "eligible_races": eligible,
            "skipped_obstacle": skipped_obstacle,
            "megu_coverage_pct": pct,
            "reason": reason,
        },
        "issues_count": len(issues),
        "issues": issues[:50],
    }


def run_check(date: str, check_type: str, storage: Any | None = None) -> dict:
    if check_type not in CHECK_TYPES:
        raise ValueError(f"invalid check_type: {check_type}")

    if storage is None:
        from src.scraper.storage import HybridStorage
        storage = HybridStorage()

    if check_type == "presence":
        result = run_presence_check(date, storage)
    elif check_type == "raw_content":
        result = run_raw_content_check(date, storage)
    else:
        result = run_calculated_check(date)

    existing = load_health(date) or {"date": date, "checks": {}}
    checks = dict(existing.get("checks") or {})
    checks[check_type] = result
    saved = save_health(date, {"checks": checks})

    try:
        from src.api.quality_auto_remediation import maybe_remediate_after_checks
        remediation = maybe_remediate_after_checks(date)
        if remediation and remediation.get("status") == "applied":
            saved["remediation"] = {
                "status": remediation.get("status"),
                "action_count": (remediation.get("plan") or {}).get("action_count"),
            }
    except Exception as e:
        logger.warning("auto remediation skipped [%s]: %s", date, e)

    return saved


def apply_check_result(date: str, check_type: str, job_id: str, result: dict) -> dict:
    existing = load_health(date) or {"date": date, "checks": {}}
    checks = dict(existing.get("checks") or {})
    rec = dict(result)
    rec["job_id"] = job_id
    checks[check_type] = rec
    return save_health(date, {"checks": checks})
