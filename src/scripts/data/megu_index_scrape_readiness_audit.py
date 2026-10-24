"""
めぐ指数計算に必要な「スクレイピング済み」データの GCS / PostgreSQL 充足監査。

対象: 芝/ダート入着レース（障害除外）。既定日付範囲 2020-01-01 〜 2026-06-01。

チェック項目（GCS race_result 正本）:
  - 存在
  - surface / distance（芝・ダート / >0）
  - 入着 entries（finish_position + time_sec）
  - lap_times（ペース補正用）

PostgreSQL:
  - races / race_results 同期
  - megu_index v2 (valid | out_of_range)

Usage:
    KEIBA_ENV=stg python -m src.scripts.data.megu_index_scrape_readiness_audit
    KEIBA_ENV=stg python -m src.scripts.data.megu_index_scrape_readiness_audit --refresh
    KEIBA_ENV=stg python -m src.scripts.data.megu_index_scrape_readiness_audit --run scrape_gcs
    KEIBA_ENV=stg python -m src.scripts.data.megu_index_scrape_readiness_audit --run sync_db
    KEIBA_ENV=stg python -m src.scripts.data.megu_index_scrape_readiness_audit --run compute_dates
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text

from src.pipeline.megu_index.flat_metadata import is_obstacle_race_name
from src.utils.keiba_logging import script_basic_config

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT_ROOT / "data/local/megu_index"
REPORT_PATH = OUT_DIR / "scrape_readiness_report.json"
TASKS_PATH = OUT_DIR / "scrape_readiness_tasks.json"
GAPS_PATH = OUT_DIR / "scrape_readiness_gaps.json"

DATE_FROM_DEFAULT = "2020-01-01"
DATE_TO_DEFAULT = "2026-06-01"


def _finisher_count(entries: list[dict]) -> int:
    n = 0
    for e in entries:
        pos = e.get("finish_position") or e.get("finish_pos") or e.get("rank")
        t = e.get("time_sec") or e.get("finish_time_sec")
        if pos and int(pos) > 0 and t and float(t) > 0:
            n += 1
    return n


def audit_gcs_race_result(rr: dict | None) -> list[str]:
    """GCS race_result のギャップ種別（複数可）。"""
    gaps: list[str] = []
    if not rr:
        return ["gcs_missing"]
    surf = (rr.get("surface") or "").strip()
    dist = int(rr.get("distance") or 0)
    if surf not in ("芝", "ダート") or dist <= 0:
        gaps.append("gcs_bad_meta")
    entries = rr.get("entries") or rr.get("results") or []
    if _finisher_count(entries) == 0:
        gaps.append("gcs_no_finishers")
    if not rr.get("lap_times"):
        gaps.append("gcs_no_lap_times")
    return gaps


def _load_target_races(session, date_from: str, date_to: str) -> list[dict]:
    rows = session.execute(
        text("""
            SELECT r.race_id,
                   TO_CHAR(r.race_date, 'YYYY-MM-DD') AS race_date,
                   r.race_name,
                   r.surface,
                   r.distance,
                   (SELECT COUNT(*) FROM race_results rr
                    WHERE rr.race_id = r.race_id AND rr.finish_pos > 0 AND rr.finish_time_sec > 0) AS db_finishers,
                   (SELECT COUNT(*) FROM megu_index mi
                    WHERE mi.race_id = r.race_id AND mi.model_version = 'v2'
                      AND mi.computation_status IN ('valid', 'out_of_range')) AS db_megu_rows
            FROM races r
            WHERE r.race_date >= :dfrom AND r.race_date <= :dto
              AND r.surface IN ('芝', 'ダート')
              AND r.distance > 0
              AND EXISTS (
                SELECT 1 FROM race_results rr
                WHERE rr.race_id = r.race_id AND rr.finish_pos > 0 AND rr.finish_time_sec > 0
              )
            ORDER BY r.race_date, r.race_id
        """),
        {"dfrom": date_from, "dto": date_to},
    ).fetchall()
    out = [dict(r._mapping) for r in rows]
    for row in out:
        row["is_obstacle_misclassified"] = is_obstacle_race_name(row.get("race_name"))
    return out


def _audit_one_race(race_row: dict, storage) -> dict:
    rid = race_row["race_id"]
    if race_row.get("is_obstacle_misclassified"):
        return {
            "race_id": rid,
            "race_date": race_row.get("race_date"),
            "gcs_gaps": [],
            "db_gaps": [],
            "legitimate_skip": True,
            "skip_reason": "obstacle_misclassified_in_db",
            "needs_scrape": False,
            "needs_db_sync": False,
            "needs_compute": False,
        }
    rr = storage.load("race_result", rid, bypass_cache=True)
    gcs_gaps = audit_gcs_race_result(rr)
    db_gaps: list[str] = []
    if int(race_row.get("db_finishers") or 0) == 0:
        db_gaps.append("db_no_finishers")
    if int(race_row.get("db_megu_rows") or 0) == 0:
        db_gaps.append("db_no_megu")
    return {
        "race_id": rid,
        "race_date": race_row.get("race_date"),
        "gcs_gaps": gcs_gaps,
        "db_gaps": db_gaps,
        "needs_scrape": any(g.startswith("gcs_") for g in gcs_gaps),
        "needs_db_sync": bool(db_gaps) and not gcs_gaps.count("gcs_missing"),
        "needs_compute": "db_no_megu" in db_gaps,
    }


def refresh_audit(
    date_from: str = DATE_FROM_DEFAULT,
    date_to: str = DATE_TO_DEFAULT,
    *,
    workers: int = 8,
) -> dict:
    from src.db.session import get_session, init_engine
    from src.scraper.storage import HybridStorage

    init_engine()
    with get_session() as session:
        targets = _load_target_races(session, date_from, date_to)

    logger.info("audit targets: %d races (%s .. %s)", len(targets), date_from, date_to)
    storage = HybridStorage()
    results: list[dict] = []
    summary: dict[str, int] = {}

    def bump(key: str, n: int = 1) -> None:
        summary[key] = summary.get(key, 0) + n

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_audit_one_race, row, storage): row for row in targets}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            done += 1
            for g in r["gcs_gaps"]:
                bump(f"gcs:{g}")
            for g in r["db_gaps"]:
                bump(f"db:{g}")
            if r["needs_scrape"]:
                bump("needs_scrape")
            if r["needs_db_sync"]:
                bump("needs_db_sync")
            if r["needs_compute"]:
                bump("needs_compute")
            if not r["gcs_gaps"] and not r["db_gaps"]:
                bump("fully_ready")
            if r.get("legitimate_skip"):
                bump("legitimate_skip_obstacle")
            if done % 2000 == 0:
                logger.info("  audited %d / %d", done, len(targets))

    scrape_ids = sorted({r["race_id"] for r in results if r["needs_scrape"]})
    sync_ids = sorted({r["race_id"] for r in results if r["needs_db_sync"]})
    compute_dates = sorted({r["race_date"] for r in results if r["needs_compute"] and r.get("race_date")})

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "date_from": date_from,
        "date_to": date_to,
        "total_races": len(targets),
        "summary": summary,
        "scrape_race_count": len(scrape_ids),
        "sync_race_count": len(sync_ids),
        "compute_date_count": len(compute_dates),
    }
    gaps = {
        "scrape_race_ids": scrape_ids,
        "sync_race_ids": sync_ids,
        "compute_dates": compute_dates,
        "by_race": {r["race_id"]: r for r in results if r["gcs_gaps"] or r["db_gaps"]},
    }
    tasks = _build_tasks(report, gaps)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    GAPS_PATH.write_text(json.dumps(gaps, indent=2, ensure_ascii=False), encoding="utf-8")
    TASKS_PATH.write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "audit done: ready=%d scrape=%d sync=%d compute_dates=%d",
        summary.get("fully_ready", 0),
        len(scrape_ids),
        len(sync_ids),
        len(compute_dates),
    )
    return report


def _build_tasks(report: dict, gaps: dict) -> dict:
    tasks: list[dict] = []
    if gaps["scrape_race_ids"]:
        tasks.append({
            "id": "backfill_gcs_meta:range",
            "status": "pending",
            "priority": 1,
            "title": "GCS race_result メタを race_shutuba から補完",
            "affected_count": len(gaps["scrape_race_ids"]),
            "command": "KEIBA_ENV=stg python -m src.scripts.data.megu_index_scrape_readiness_audit --run backfill_gcs_meta",
        })
        tasks.append({
            "id": "scrape_gcs:range",
            "status": "pending",
            "priority": 2,
            "title": "GCS race_result 再取得（backfill 後も lap 欠損等）",
            "affected_count": len(gaps["scrape_race_ids"]),
            "command": "KEIBA_ENV=stg python -m src.scripts.data.megu_index_scrape_readiness_audit --run scrape_gcs",
        })
    if gaps["sync_race_ids"]:
        tasks.append({
            "id": "sync_db:range",
            "status": "pending",
            "priority": 2,
            "title": "PostgreSQL races/race_results 同期",
            "affected_count": len(gaps["sync_race_ids"]),
            "command": "KEIBA_ENV=stg python -m src.scripts.data.megu_index_scrape_readiness_audit --run sync_db",
        })
    if gaps["compute_dates"]:
        tasks.append({
            "id": "compute_dates:range",
            "status": "pending",
            "priority": 3,
            "title": "megu_index v2 再計算（日別）",
            "affected_count": len(gaps["compute_dates"]),
            "command": "KEIBA_ENV=stg python -m src.scripts.data.megu_index_scrape_readiness_audit --run compute_dates",
        })
    if not tasks:
        tasks.append({
            "id": "scrape_readiness:ok",
            "status": "done",
            "priority": 0,
            "title": "GCS/DB データ充足（対象期間）",
            "affected_count": report.get("total_races", 0),
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        })
    return {"version": 1, "updated_at": datetime.now().isoformat(timespec="seconds"), "tasks": tasks}


def _load_gaps() -> dict:
    if GAPS_PATH.exists():
        return json.loads(GAPS_PATH.read_text(encoding="utf-8"))
    return {"scrape_race_ids": [], "sync_race_ids": [], "compute_dates": []}


def run_scrape_gcs(*, limit: int | None = None, sleep_sec: float = 2.0, year: str | None = None) -> dict:
    from src.scraper.run import ScraperRunner
    from src.scraper.storage import HybridStorage

    gaps = _load_gaps()
    ids = gaps.get("scrape_race_ids") or []
    if year:
        ids = [rid for rid in ids if rid.startswith(year)]
    if limit:
        ids = ids[:limit]
    runner = ScraperRunner()
    storage = HybridStorage()
    stats = {"ok": 0, "error": 0, "skipped": 0}
    errors: list[str] = []
    for i, rid in enumerate(ids):
        try:
            rr = runner.scrape_race_result(rid, skip_existing=False)
            storage.invalidate_load_cache("race_result", rid)
            rr = storage.load("race_result", rid, bypass_cache=True)
            if rr and not audit_gcs_race_result(rr):
                stats["ok"] += 1
            elif rr:
                stats["skipped"] += 1
                logger.warning("scrape incomplete: %s gaps=%s", rid, audit_gcs_race_result(rr))
            else:
                stats["error"] += 1
                errors.append(rid)
        except Exception as e:
            stats["error"] += 1
            errors.append(f"{rid}:{e}")
            logger.error("scrape error %s: %s", rid, e)
        if i < len(ids) - 1:
            time.sleep(random.uniform(sleep_sec * 0.8, sleep_sec * 1.2))
        if (i + 1) % 50 == 0:
            logger.info("scrape progress %d/%d ok=%d err=%d", i + 1, len(ids), stats["ok"], stats["error"])
    return {"processed": len(ids), **stats, "errors_sample": errors[:20]}


def run_backfill_gcs_meta(*, limit: int | None = None, year: str | None = None) -> dict:
    """race_shutuba から race_result のメタデータを GCS に書き戻す（再スクレイプ不要）。"""
    from src.scraper.storage import HybridStorage
    from src.utils.race_card_merge import patch_result_metadata_from_shutuba

    gaps = _load_gaps()
    ids = gaps.get("scrape_race_ids") or []
    if year:
        ids = [rid for rid in ids if rid.startswith(year)]
    if limit:
        ids = ids[:limit]

    storage = HybridStorage()
    stats = {"ok": 0, "skipped": 0, "error": 0, "saved": 0}
    for i, rid in enumerate(ids):
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
                stats["ok"] += 1
            else:
                stats["error"] += 1
        except Exception as e:
            stats["error"] += 1
            logger.error("backfill meta error %s: %s", rid, e)
        if (i + 1) % 500 == 0:
            logger.info("backfill progress %d/%d saved=%d", i + 1, len(ids), stats["saved"])
    return {"processed": len(ids), **stats}


def run_sync_db(*, limit: int | None = None) -> dict:
    from src.scripts.data.etl_stg_db import process_race, _get_engine
    from sqlalchemy.orm import Session

    gaps = _load_gaps()
    ids = gaps.get("sync_race_ids") or []
    if limit:
        ids = ids[:limit]
    engine = _get_engine()
    stats = {"ok": 0, "error": 0}
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
                logger.error("sync error %s: %s", rid, e)
    return {"processed": len(ids), **stats}


def run_compute_dates(*, limit: int | None = None) -> dict:
    from src.pipeline.megu_index.compute import compute_for_date

    gaps = _load_gaps()
    dates = gaps.get("compute_dates") or []
    if limit:
        dates = dates[:limit]
    stats = {"ok": 0, "skipped": 0, "error": 0}
    for d in dates:
        try:
            r = compute_for_date(d)
            st = r.get("status")
            if st == "ok":
                stats["ok"] += 1
            elif st == "skipped":
                stats["skipped"] += 1
                logger.warning("compute skipped %s: %s", d, r)
            else:
                stats["error"] += 1
        except Exception as e:
            stats["error"] += 1
            logger.error("compute error %s: %s", d, e)
    return {"processed": len(dates), **stats}


def print_summary() -> None:
    if not REPORT_PATH.exists():
        print("report not found – run --refresh first")
        return
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    print(f"\n=== scrape readiness ({report['date_from']} .. {report['date_to']}) ===")
    print(f"total races: {report['total_races']}")
    print(f"summary: {json.dumps(report.get('summary', {}), ensure_ascii=False)}")
    print(f"scrape needed: {report.get('scrape_race_count')}")
    print(f"db sync needed: {report.get('sync_race_count')}")
    print(f"compute dates: {report.get('compute_date_count')}")
    if TASKS_PATH.exists():
        tasks = json.loads(TASKS_PATH.read_text(encoding="utf-8")).get("tasks", [])
        pending = [t for t in tasks if t.get("status") != "done"]
        print(f"pending tasks: {len(pending)}")
        for t in pending:
            print(f"  [{t.get('priority')}] {t['id']}: {t['title']} (n={t.get('affected_count')})")


def main() -> None:
    parser = argparse.ArgumentParser(description="めぐ指数用 GCS/DB スクレイプ充足監査")
    parser.add_argument("--refresh", action="store_true", help="監査実行")
    parser.add_argument("--summary", action="store_true", help="結果サマリ表示")
    parser.add_argument("--date-from", default=DATE_FROM_DEFAULT)
    parser.add_argument("--date-to", default=DATE_TO_DEFAULT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--run", choices=["backfill_gcs_meta", "scrape_gcs", "sync_db", "compute_dates"])
    parser.add_argument("--limit", type=int, help="--run 時の処理上限")
    parser.add_argument("--year", type=str, help="--run scrape_gcs 対象年 (例: 2020)")
    parser.add_argument("--sleep", type=float, default=2.0, help="scrape 間隔秒")
    args = parser.parse_args()
    script_basic_config()

    if args.run == "backfill_gcs_meta":
        r = run_backfill_gcs_meta(limit=args.limit, year=args.year)
        print(json.dumps(r, indent=2, ensure_ascii=False))
        refresh_audit(args.date_from, args.date_to, workers=args.workers)
        print_summary()
        return

    if args.run == "scrape_gcs":
        r = run_scrape_gcs(limit=args.limit, sleep_sec=args.sleep, year=args.year)
        print(json.dumps(r, indent=2, ensure_ascii=False))
        refresh_audit(args.date_from, args.date_to, workers=args.workers)
        print_summary()
        return

    if args.run == "sync_db":
        r = run_sync_db(limit=args.limit)
        print(json.dumps(r, indent=2, ensure_ascii=False))
        refresh_audit(args.date_from, args.date_to, workers=args.workers)
        print_summary()
        return

    if args.run == "compute_dates":
        r = run_compute_dates(limit=args.limit)
        print(json.dumps(r, indent=2, ensure_ascii=False))
        refresh_audit(args.date_from, args.date_to, workers=args.workers)
        print_summary()
        return

    if args.refresh or (not args.summary):
        refresh_audit(args.date_from, args.date_to, workers=args.workers)
    print_summary()


if __name__ == "__main__":
    main()
