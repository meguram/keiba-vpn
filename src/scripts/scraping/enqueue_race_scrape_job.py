#!/usr/bin/env python3
"""
単一レースのスクレイプジョブをキューへ投入する CLI。

UI サイドバーからの「ターゲットレース再取得」と同等に、
``POST /api/scrape-queue/add-job`` と同じ payload を ``ScrapeJobQueue.add_job`` に渡す。

例（race_result_on_time を先頭付近で再投入）::

  python -m src.scripts.scraping.enqueue_race_scrape_job \\
    --race-id 202605020302 --tasks race_result_on_time --priority 500000

``--kick`` でワーカ起動キック（job_queue.kick_process_queue_background）を付ける。
"""

from __future__ import annotations

import argparse
import sys

from src.scraper.job_queue import ScrapeJobQueue, kick_process_queue_background
from src.scraper.queue_tasks import normalize_tasks, validate_tasks_for_kind


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="単一レースのスクレイプジョブをキューへ投入")
    p.add_argument("--race-id", required=True, help="12 桁 race_id（例 202605020302）")
    p.add_argument(
        "--tasks",
        required=True,
        help="カンマ区切りタスク ID（例 race_result_on_time または race_result,race_result_on_time）",
    )
    p.add_argument(
        "--priority",
        type=int,
        default=500_000,
        help="数値が大きいほど先に処理（既定 500000。最優先系は数百万以上）",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="既存 JSON を上書き取得（smart_skip を無効化）",
    )
    p.add_argument(
        "--kick",
        action="store_true",
        help="投入後に kick_process_queue_background を呼ぶ",
    )
    p.add_argument("--dry-run", action="store_true", help="payload のみ表示")
    args = p.parse_args(argv)

    rid = str(args.race_id).strip()
    raw_tasks = [x.strip() for x in str(args.tasks).split(",") if x.strip()]
    tasks = normalize_tasks(raw_tasks)
    if not tasks:
        print("tasks が空です", file=sys.stderr)
        return 2
    err = validate_tasks_for_kind("race", tasks)
    if err:
        print(err, file=sys.stderr)
        return 2

    job: dict = {
        "job_kind": "race",
        "target_id": rid,
        "tasks": tasks,
        "priority": int(args.priority),
    }
    if args.overwrite:
        job["overwrite"] = True
        job["smart_skip"] = False

    if args.dry_run:
        print(job)
        return 0

    q = ScrapeJobQueue()
    out = q.add_job(job)
    print(out)
    if args.kick:
        kick_process_queue_background()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
