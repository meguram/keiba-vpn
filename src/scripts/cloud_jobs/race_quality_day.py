#!/usr/bin/env python3
"""
指定日（既定: 本日 JST）の全 JRA レースのレース質を一括推定する CLI。

元は ``GET /api/race-quality/day``（src/api/app.py）が呼ぶ
``src.research.race.race_quality_model.analyze_date`` をそのまま呼び出す薄いラッパー。
処理ロジック自体はこの CLI では実装しない。

Cloud Scheduler + Cloud Run Jobs から日次実行する想定
（docs/operations/gcp-cloud-run-jobs.md 参照）。

Usage:
  python -m src.scripts.cloud_jobs.race_quality_day
  python -m src.scripts.cloud_jobs.race_quality_day --date 20260601
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone

_JST = timezone(timedelta(hours=9))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        default="",
        help="対象日 YYYYMMDD（省略時は本日 JST）",
    )
    args = parser.parse_args(argv)

    date_compact = (args.date or datetime.now(_JST).strftime("%Y%m%d")).replace("-", "")
    if len(date_compact) != 8 or not date_compact.isdigit():
        print(f"--date は YYYYMMDD 形式で指定してください: {args.date!r}", file=sys.stderr)
        return 1

    from src.research.race.race_quality_model import analyze_date
    from src.scraper.storage import HybridStorage

    result = analyze_date(HybridStorage(), date_compact)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
