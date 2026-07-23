"""
prod 向け PostgreSQL カバレッジキャッシュを日次構築する。

cron 例（毎日 0:00 JST）:
  KEIBA_ENV=prod python -m src.scripts.data.build_db_coverage_cache
  KEIBA_ENV=prod python -m src.scripts.data.build_db_coverage_cache --year 2025
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta, timezone

from src.api.monitor_coverage import build_db_coverage_cache_for_date
from src.scraper.date_coverage import load_year_coverage
from src.scripts.data.etl_stg_db import get_target_dates
from src.utils.keiba_logging import script_basic_config

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))


def main() -> None:
    script_basic_config()
    parser = argparse.ArgumentParser(description="db_coverage キャッシュ構築（prod 向け）")
    parser.add_argument("--year", type=int, help="対象年（省略時は当年）")
    parser.add_argument("--date", type=str, help="単日 YYYYMMDD")
    parser.add_argument("--recent-days", type=int, help="直近 N 日（race_lists ベース）")
    args = parser.parse_args()

    if args.date:
        dates = [args.date]
    elif args.recent_days is not None:
        dates = get_target_dates(None, args.recent_days)
    else:
        year = args.year or datetime.now(JST).year
        cov = load_year_coverage(year)
        dates = sorted(cov.keys()) if cov else []

    if not dates:
        logger.warning("対象日がありません")
        return

    ok = 0
    for d in dates:
        try:
            build_db_coverage_cache_for_date(d)
            ok += 1
        except Exception as e:
            logger.error("failed %s: %s", d, e)

    logger.info("db_coverage cache: %d / %d days", ok, len(dates))


if __name__ == "__main__":
    main()
