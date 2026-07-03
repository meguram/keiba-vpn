#!/usr/bin/env python3
"""
期間内に開催された JRA レースの出走馬について、過去成績（horse_result）を再取得するキュー投入。

race_lists → 各レースの race_shutuba / race_result から horse_id を収集し、
horse_profile タスク（scrape_horse → horse_result を GCS 上書き保存）をキューへ載せる。

使い方:
  python -m src.scripts.scraping.enqueue_horse_result_period --dry-run
  python -m src.scripts.scraping.enqueue_horse_result_period \\
    --start 20240101 --end 20260521
"""

from __future__ import annotations

import argparse
import sys

from src.scraper.job_queue import ScrapeJobQueue
from src.scraper.period_runners import enqueue_horse_tasks_for_race_period
from src.scraper.storage import HybridStorage
from src.utils.logger import get_logger

logger = get_logger("EnqueueHorseResultPeriod")

DEFAULT_START = "20240101"
DEFAULT_END = "20260521"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="期間内レース出走馬の horse_result 再取得をキューへ投入（GCS 上書き）",
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        help=f"開始日 YYYYMMDD（既定 {DEFAULT_START}）",
    )
    parser.add_argument(
        "--end",
        default=DEFAULT_END,
        help=f"終了日 YYYYMMDD（既定 {DEFAULT_END}）",
    )
    parser.add_argument(
        "--task",
        default="horse_profile",
        choices=["horse_profile", "horse_pedigree"],
        help="馬タスク ID（horse_profile → horse_result）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="頭数・メタのみ表示しキューへ入れない",
    )
    parser.add_argument(
        "--include-nar",
        action="store_true",
        help="JRA 以外（NAR 等）の race_id も含める",
    )
    args = parser.parse_args(argv)

    start = str(args.start).replace("-", "")[:8]
    end = str(args.end).replace("-", "")[:8]
    if len(start) != 8 or len(end) != 8 or not start.isdigit() or not end.isdigit():
        parser.error("--start / --end は YYYYMMDD で指定してください")
    if start > end:
        parser.error("start は end 以前にしてください")

    storage = HybridStorage()
    queue = ScrapeJobQueue()
    result = enqueue_horse_tasks_for_race_period(
        storage,
        queue,
        start_date=start,
        end_date=end,
        tasks=[args.task],
        limit=0,
        dry_run=args.dry_run,
        jra_only=not args.include_nar,
        overwrite=True,
        smart_skip=False,
    )

    print(f"期間: {start} .. {end}")
    print(f"タスク: {[args.task]}（overwrite=True, smart_skip=False）")
    meta = result.get("meta") or {}
    print(f"収集メタ: {meta}")
    print(f"対象馬: {result.get('candidate_horses', 0):,} 頭")

    if args.dry_run:
        print(f"投入予定: {result.get('would_enqueue', 0):,} 頭")
        sample = result.get("sample_horse_ids") or []
        if sample:
            print("例:", ", ".join(sample[:10]))
        return 0

    if not result.get("candidate_horses"):
        print("対象馬がありません（race_lists / 出馬表・結果の不足を確認）")
        return 1

    print("キュー投入結果:", {
        k: result.get(k)
        for k in (
            "created",
            "requeued",
            "duplicate",
            "batches",
            "processed_horses",
        )
    })
    logger.info("完了: %s", result)
    print("ワーカー起動: python -m src.scraper.queue_worker または API サーバー経由")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
