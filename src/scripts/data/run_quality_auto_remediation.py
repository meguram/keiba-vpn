"""
品質ヘルス異常日の自動修復バッチ。

Usage:
  python -m src.scripts.data.run_quality_auto_remediation --dry-run
  python -m src.scripts.data.run_quality_auto_remediation --date 20260711
  python -m src.scripts.data.run_quality_auto_remediation --year 2026 --force
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.api.quality_auto_remediation import remediate_date
from src.api.quality_health import get_health_view, load_year_health
from src.scraper.date_coverage import load_year_coverage
from src.utils.keiba_logging import script_basic_config

logger = logging.getLogger(__name__)


def _dates_for_years(years: list[int]) -> list[str]:
    dates: list[str] = []
    for y in years:
        cov = load_year_coverage(y)
        dates.extend(sorted(cov.keys()))
    return dates


def _needs_remediation(date: str) -> bool:
    view = get_health_view(date)
    overall = view.get("overall_display_status") or view.get("overall_status")
    return overall not in ("ok", "na", None)


def main() -> None:
    script_basic_config()
    parser = argparse.ArgumentParser(description="品質ヘルス自動修復")
    parser.add_argument("--date", type=str, help="単日 YYYYMMDD")
    parser.add_argument("--year", type=int, action="append", help="対象年")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="クールダウン無視")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-kick", action="store_true")
    args = parser.parse_args()

    if args.date:
        dates = [args.date.replace("-", "")]
    else:
        years = args.year or list(range(2020, 2027))
        dates = _dates_for_years(years)

    targets = [d for d in dates if _needs_remediation(d)]
    if args.limit > 0:
        targets = targets[: args.limit]

    logger.info("remediation targets: %d / %d dates", len(targets), len(dates))

    stats = {"applied": 0, "skipped": 0, "error": 0}

    def _one(d: str) -> tuple[str, str]:
        try:
            r = remediate_date(
                d,
                dry_run=args.dry_run,
                force=args.force,
                kick=not args.no_kick,
            )
            return d, r.get("status") or "unknown"
        except Exception as e:
            logger.error("remediation failed %s: %s", d, e)
            return d, "error"

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = [pool.submit(_one, d) for d in targets]
        for fut in as_completed(futs):
            date, status = fut.result()
            if status == "applied":
                stats["applied"] += 1
            elif status == "error":
                stats["error"] += 1
            else:
                stats["skipped"] += 1
            logger.info("%s -> %s", date, status)

    logger.info("done: %s", stats)

    if args.year and not args.date:
        for y in (args.year or []):
            health = load_year_health(y)
            counts: dict[str, int] = {}
            for rec in health.values():
                st = rec.get("overall_status") or "unknown"
                counts[st] = counts.get(st, 0) + 1
            logger.info("year %d health: %s", y, counts)


if __name__ == "__main__":
    main()
