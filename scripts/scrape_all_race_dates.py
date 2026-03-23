"""全開催日のレース一覧を2020年1月1日から取得するスクリプト。"""

from __future__ import annotations

import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scraper.run import ScraperRunner
from scraper.storage import HybridStorage

logger = logging.getLogger("scrape_all_race_dates")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    storage = HybridStorage()
    runner = ScraperRunner()

    existing = set(storage.list_keys("race_lists"))
    logger.info("既存の race_lists: %d 件", len(existing))

    start = date(2020, 1, 1)
    end = date(2026, 3, 29)

    # Collect all Sat/Sun + known holiday dates
    candidates = []
    current = start
    while current <= end:
        if current.weekday() in (5, 6):  # Sat/Sun
            candidates.append(current)
        # Also check some weekdays - JRA sometimes has Mon/Fri racing
        elif current.weekday() in (0, 4):  # Mon/Fri  
            candidates.append(current)
        current += timedelta(days=1)

    # Filter out already scraped
    todo = []
    for d in candidates:
        ds = d.strftime("%Y%m%d")
        if ds not in existing:
            todo.append(ds)

    logger.info("候補日: %d, 未取得: %d, スキップ: %d",
                len(candidates), len(todo), len(candidates) - len(todo))

    found = 0
    empty = 0
    for i, ds in enumerate(todo):
        try:
            races = runner.scrape_race_list(ds)
            if races:
                found += 1
            else:
                empty += 1

            if (i + 1) % 50 == 0:
                logger.info("進捗: %d/%d (レースあり=%d, 空=%d)",
                            i + 1, len(todo), found, empty)
        except Exception as e:
            logger.error("失敗 [%s]: %s", ds, e)
            empty += 1

        time.sleep(0.3)

    logger.info("完了: レースあり=%d, 空=%d (計 %d)", found, empty, found + empty)

    # Final count
    all_dates = storage.list_keys("race_lists")
    logger.info("最終 race_lists 件数: %d", len(all_dates))
    years = {}
    for d in all_dates:
        y = d[:4]
        years[y] = years.get(y, 0) + 1
    for y in sorted(years):
        logger.info("  %s: %d 日", y, years[y])


if __name__ == "__main__":
    main()
