"""
date_coverage と race_lists の整合修復。

- race_list が空 / 非開催なのに date_coverage に stale race_ids が残る問題を解消
- 空 race_list に no_race_scheduled メタを付与（未設定時）
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.scraper.date_coverage import COVERAGE_DIR, _coverage_path
from src.utils.keiba_logging import script_basic_config
from src.utils.race_list_for_date import RACE_LIST_DIR, load_race_list_data, opening_date_kind

logger = logging.getLogger(__name__)


def repair_date(date: str, *, dry_run: bool = False) -> dict:
    kind = opening_date_kind(date)
    cov_path = _coverage_path(date)
    rl = load_race_list_data(date)
    actions: list[str] = []

    if kind in ("no_meeting", "missing"):
        if cov_path.exists():
            actions.append("remove_stale_coverage")
            if not dry_run:
                cov_path.unlink(missing_ok=True)

        if rl is not None and kind == "no_meeting":
            races = rl.get("races") or []
            meta = rl.get("_meta") if isinstance(rl.get("_meta"), dict) else {}
            if not races and meta.get("note") != "no_race_scheduled":
                actions.append("tag_no_race_scheduled")
                if not dry_run:
                    meta = dict(meta)
                    meta["note"] = "no_race_scheduled"
                    rl["races"] = []
                    rl["_meta"] = meta
                    path = RACE_LIST_DIR / f"{date}.json"
                    path.write_text(
                        json.dumps(rl, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8",
                    )
    elif kind == "meeting":
        if cov_path.exists() and not dry_run:
            actions.append("rebuild_coverage_recommended")
        else:
            actions.append("ok")

    return {"date": date, "kind": kind, "actions": actions}


def main() -> None:
    script_basic_config()
    parser = argparse.ArgumentParser(description="date_coverage / race_lists 整合修復")
    parser.add_argument("--year", type=int, action="append")
    parser.add_argument("--date", type=str)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dates: list[str] = []
    if args.date:
        dates = [args.date]
    else:
        years = args.year or list(range(2020, 2027))
        for y in years:
            year_dir = COVERAGE_DIR / str(y)
            if year_dir.exists():
                dates.extend(sorted(p.stem for p in year_dir.glob("*.json")))

    stats = {"removed": 0, "tagged": 0, "meeting": 0}
    for dt in dates:
        r = repair_date(dt, dry_run=args.dry_run)
        for a in r["actions"]:
            if a == "remove_stale_coverage":
                stats["removed"] += 1
            elif a == "tag_no_race_scheduled":
                stats["tagged"] += 1
        if r["kind"] == "meeting":
            stats["meeting"] += 1
        if r["actions"] and r["actions"] != ["ok"]:
            logger.info("%s kind=%s actions=%s", dt, r["kind"], r["actions"])

    logger.info(
        "repair done dry_run=%s removed=%d tagged=%d meeting=%d total=%d",
        args.dry_run,
        stats["removed"],
        stats["tagged"],
        stats["meeting"],
        len(dates),
    )


if __name__ == "__main__":
    main()
