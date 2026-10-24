"""
全開催日の品質チェックを一括実行（date_coverage 索引がある日）。

Usage:
  python -m src.scripts.data.run_quality_health_batch
  python -m src.scripts.data.run_quality_health_batch --year 2024
  python -m src.scripts.data.run_quality_health_batch --check presence
  python -m src.scripts.data.run_quality_health_batch --limit 10
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.api.quality_health import CHECK_TYPES, load_year_health, run_check
from src.scraper.date_coverage import load_year_coverage
from src.utils.keiba_logging import script_basic_config

logger = logging.getLogger(__name__)


def _dates_for_years(years: list[int]) -> list[str]:
    dates: list[str] = []
    for y in years:
        cov = load_year_coverage(y)
        dates.extend(sorted(cov.keys()))
    return dates


def main() -> None:
    script_basic_config()
    parser = argparse.ArgumentParser(description="品質ヘルス一括チェック")
    parser.add_argument("--year", type=int, action="append", help="対象年（複数可）")
    parser.add_argument(
        "--check",
        choices=CHECK_TYPES,
        action="append",
        help="チェック種別（省略時は全部）",
    )
    parser.add_argument("--limit", type=int, default=0, help="日数上限（0=無制限）")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    years = args.year or list(range(2020, 2027))
    checks = args.check or list(CHECK_TYPES)
    dates = _dates_for_years(years)
    if args.limit > 0:
        dates = dates[: args.limit]

    logger.info("batch: %d dates × checks %s", len(dates), checks)

    def _one_date(date: str) -> tuple[str, str]:
        results = []
        for check_type in checks:
            try:
                run_check(date, check_type, storage=None)
                results.append("ok")
            except Exception as e:
                logger.error("failed %s %s: %s", date, check_type, e)
                results.append("error")
        status = "ok" if all(r == "ok" for r in results) else "partial"
        return date, status

    ok = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = [pool.submit(_one_date, d) for d in dates]
        for fut in as_completed(futs):
            date, status = fut.result()
            if status == "ok":
                ok += 1
            else:
                logger.warning("%s -> %s", date, status)

    logger.info("done: %d / %d dates OK", ok, len(dates))

    # サマリ
    for y in years:
        health = load_year_health(y)
        if not health:
            continue
        counts = {"ok": 0, "warn": 0, "fail": 0, "unknown": 0}
        for row in health.values():
            st = row.get("overall_status") or "unknown"
            counts[st] = counts.get(st, 0) + 1
        logger.info("year %s health summary: %s (days=%d)", y, counts, len(health))


if __name__ == "__main__":
    main()
