#!/usr/bin/env python3
"""
出馬表 (race_shutuba) の自動キュー投入を1回実行する CLI。

今日から ``--days-ahead`` 日先までの race_lists を走査し、未取得の race_shutuba を
スクレイピングキューへ投入する（``smart_skip=True`` なので取得済みはスキップ）。

元は src/api/app.py の ``_daily_shutuba_enqueue_loop``（既定 毎日 07:00 JST、
環境変数 ``DAILY_SHUTUBA_HOUR_JST`` / ``DAILY_SHUTUBA_MINUTE_JST`` /
``DAILY_SHUTUBA_DAYS_AHEAD``）が毎日1回呼んでいた実処理
（``src.scraper.period_runners.enqueue_race_tasks_for_race_period``）を、
daemon thread のループ部分を除いてそのまま呼び出す薄いラッパー。
処理ロジック自体はこの CLI では実装しない。

Cloud Scheduler + Cloud Run Jobs から毎日 07:00 JST に起動する想定
（docs/operations/gcp-cloud-run-jobs.md 参照）。

Usage:
  python -m src.scripts.cloud_jobs.daily_shutuba_enqueue
  python -m src.scripts.cloud_jobs.daily_shutuba_enqueue --days-ahead 14 --limit 500
  python -m src.scripts.cloud_jobs.daily_shutuba_enqueue --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--days-ahead",
        type=int,
        default=14,
        help="今日から何日先までを対象にするか（既定: DAILY_SHUTUBA_DAYS_AHEAD と同じ 14、1-60）",
    )
    parser.add_argument("--limit", type=int, default=500, help="キュー投入するレース数の上限")
    parser.add_argument("--priority", type=int, default=10, help="キュー投入時の優先度")
    parser.add_argument(
        "--no-jra-only",
        dest="jra_only",
        action="store_false",
        default=True,
        help="JRA 以外（地方競馬等）も対象にする（既定は JRA のみ）",
    )
    parser.add_argument("--dry-run", action="store_true", help="キュー投入せず対象のみ列挙する")
    args = parser.parse_args(argv)

    days_ahead = max(1, min(60, args.days_ahead))

    from src.scraper.job_queue import ScrapeJobQueue, kick_process_queue_background
    from src.scraper.period_runners import enqueue_race_tasks_for_race_period
    from src.scraper.storage import HybridStorage

    storage = HybridStorage()
    queue = ScrapeJobQueue()

    today = date.today()
    end = today + timedelta(days=days_ahead)

    result = enqueue_race_tasks_for_race_period(
        storage,
        queue,
        start_date=today.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
        tasks=["race_shutuba"],
        limit=args.limit,
        dry_run=args.dry_run,
        jra_only=args.jra_only,
        smart_skip=True,
        priority=args.priority,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    if not args.dry_run and int(result.get("created") or 0) > 0:
        kick_process_queue_background()

    return 1 if result.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
