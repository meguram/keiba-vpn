#!/usr/bin/env python3
"""
未来のレースカレンダー情報を取得して race_lists に保存。

netkeiba レーストップ（https://race.netkeiba.com/top/?kaisai_date=YYYYMMDD）が
JavaScript で読み込む race_list と同じ HTTP 経路で race_id を取得する。

使い方:
  python scripts/fetch_race_calendar.py 2026-03-21
  python scripts/fetch_race_calendar.py 2026-03-21 2026-03-22
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from scraper.client import NetkeibaClient
from scraper.netkeiba_top_race_list import fetch_races_for_kaisai_date
from utils.logger import get_logger

logger = get_logger(__name__)


def fetch_race_calendar(start_date_str: str, end_date_str: str | None = None):
    """指定期間のレース一覧を netkeiba top の AJAX 相当で取得し保存する。"""

    if end_date_str is None:
        end_date_str = start_date_str

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    BASE_DIR = Path(__file__).parent.parent
    RACE_LIST_DIR = BASE_DIR / "data" / "local" / "race_lists"
    RACE_LIST_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== レースカレンダー取得: %s ~ %s (netkeiba top race_list) ===", start_date_str, end_date_str)

    client = NetkeibaClient(auto_login=True)
    total_races_found = 0
    current = start_date

    while current <= end_date:
        date_str = current.strftime("%Y%m%d")
        date_key = current.strftime("%Y-%m-%d")
        day_offset = (current - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).days

        races = fetch_races_for_kaisai_date(client, date_str)
        venues = sorted({r["venue"] for r in races})

        if races:
            data = {
                "date": date_str,
                "races": races,
                "_meta": {
                    "scraped_at": time.time(),
                    "scraped_at_jst": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "source": "fetch_race_calendar_script",
                    "kaisai_date": date_str,
                    "venues": venues,
                    "day_offset": day_offset,
                },
            }
            output_path = RACE_LIST_DIR / f"{date_str}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info("%s → 保存 %s (%d レース)", date_key, output_path, len(races))
            total_races_found += len(races)
        else:
            logger.info("%s: 開催なし（または一覧タブなし）", date_key)

        current += timedelta(days=1)
        time.sleep(0.5)

    logger.info("=== 完了: 合計 %d レース ===", total_races_found)
    return total_races_found


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python scripts/fetch_race_calendar.py 2026-03-21 [2026-03-22]")
        sys.exit(1)

    start = sys.argv[1]
    end = sys.argv[2] if len(sys.argv) >= 3 else start

    fetch_race_calendar(start, end)
