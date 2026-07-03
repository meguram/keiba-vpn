#!/usr/bin/env python3
"""
出馬表 (race_shutuba) の馬番・枠番欠落を検出し、再スクレイプをキューへ投入する。

対象: entries の horse_number が全頭 0、または有効馬番が重複・欠落しているレース。

注意:
  netkeiba 出馬表で枠順・馬番が未発表のとき（td.Umaban が空、確定後は td.Umaban1 等）は
  再スクレイプしても horse_number は 0 のままです。位置取り推論は normalize_race_entries で補完。
  確定後に再取得する場合は --verify-live で「サイト上に馬番があるレース」だけ投入できます。

使い方:
  python -m src.scripts.scraping.enqueue_shutuba_horse_number_backfill --date-prefix 20260502 --dry-run
  python -m src.scripts.scraping.enqueue_shutuba_horse_number_backfill --date-prefix 202605
  python -m src.scripts.scraping.enqueue_shutuba_horse_number_backfill --race-id 202605021012
  python -m src.scripts.scraping.enqueue_shutuba_horse_number_backfill --date-prefix 202605 --verify-live
"""

from __future__ import annotations

import argparse
import sys

from src.scraper.job_queue import ScrapeJobQueue
from src.scraper.storage import HybridStorage
from src.utils.logger import get_logger

logger = get_logger("EnqueueShutubaHorseNumber")


def shutuba_needs_rescrape(entries: list[dict]) -> bool:
    if not entries:
        return False
    nums = [int(e.get("horse_number") or 0) for e in entries]
    valid = [n for n in nums if n > 0]
    if not valid:
        return True
    return len(valid) != len(nums) or len(set(valid)) != len(valid)


def live_shutuba_has_drawn_numbers(race_id: str) -> bool:
    """netkeiba 出馬表 HTML に馬番テキストがあるか（枠順確定後）。"""
    from src.scraper.parsers import RaceCardParser
    from src.scraper.run import ScraperRunner

    runner = ScraperRunner(interval=0.8, cache=False, auto_login=True)
    url = runner.RACE_CARD_URL.format(race_id=race_id)
    html = runner.client.fetch(url)
    data = RaceCardParser().parse(html, race_id=race_id)
    nums = [int(e.get("horse_number") or 0) for e in data.get("entries") or []]
    return bool(nums) and max(nums) > 0


def collect_race_ids(
    storage: HybridStorage,
    *,
    date_prefix: str = "",
    race_ids: list[str] | None = None,
    verify_live: bool = False,
) -> list[str]:
    if race_ids:
        todo: list[str] = []
        for rid in race_ids:
            rid = str(rid).strip()
            if not rid:
                continue
            data = storage.load("race_shutuba", rid) or {}
            if shutuba_needs_rescrape(data.get("entries") or []):
                todo.append(rid)
        if verify_live:
            todo = [rid for rid in todo if live_shutuba_has_drawn_numbers(rid)]
        return sorted(set(todo))

    prefix = date_prefix.strip().replace("-", "")
    keys = storage.list_keys("race_shutuba")
    todo = []
    for key in keys:
        rid = key.replace(".json", "")
        if prefix and not rid.startswith(prefix):
            continue
        data = storage.load("race_shutuba", rid) or {}
        if shutuba_needs_rescrape(data.get("entries") or []):
            todo.append(rid)
    if verify_live:
        verified: list[str] = []
        for rid in todo:
            if live_shutuba_has_drawn_numbers(rid):
                verified.append(rid)
            else:
                logger.info("スキップ（サイト上も馬番未確定）: %s", rid)
        todo = verified
    return sorted(todo)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="馬番欠落の出馬表を race_shutuba タスクでキュー再取得",
    )
    parser.add_argument(
        "--date-prefix",
        metavar="YYYYMMDD",
        help="race_id の先頭一致（例: 20260502 または 202605）",
    )
    parser.add_argument(
        "--race-id",
        action="append",
        default=[],
        help="個別 race_id（複数可）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="件数のみ表示しキューへ入れない",
    )
    parser.add_argument(
        "--priority",
        type=int,
        default=5,
        help="キュー優先度（小さいほど先）",
    )
    parser.add_argument(
        "--verify-live",
        action="store_true",
        help="netkeiba 上で馬番が見えるレースだけ投入（枠順未確定は除外）",
    )
    args = parser.parse_args(argv)

    if not args.date_prefix and not args.race_id:
        parser.error("--date-prefix または --race-id を指定してください")

    storage = HybridStorage()
    todo = collect_race_ids(
        storage,
        date_prefix=args.date_prefix or "",
        race_ids=args.race_id or None,
        verify_live=args.verify_live,
    )
    print(f"再取得対象: {len(todo)} レース")
    if todo[:20]:
        print("例:", ", ".join(todo[:20]))
    if len(todo) > 20:
        print(f"... 他 {len(todo) - 20} 件")

    if args.dry_run:
        print("--dry-run のため投入しません")
        return 0

    if not todo:
        print("対象なし")
        return 0

    specs = [
        {
            "job_kind": "race",
            "target_id": rid,
            "tasks": ["race_shutuba"],
            "overwrite": True,
            "smart_skip": False,
            "priority": args.priority,
            "note": "shutuba_horse_number_backfill",
        }
        for rid in todo
    ]
    q = ScrapeJobQueue()
    stats = q.bulk_add_jobs(specs)
    print("bulk_add_jobs:", stats)
    print(
        "キューワーカーが動作中なら順次 race_shutuba を上書き保存します。"
        " 単体: python -m src.scraper.queue_worker"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
