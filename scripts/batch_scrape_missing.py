"""
未取得データの一括スクレイピング (新しい日付から順に)

12:30 (または指定時刻) まで自動実行。各日付について:
  1. レース一覧を取得
  2. 各レースのフルデータ (出馬表, タイム指数, 馬柱, オッズ等) をスクレイピング
  3. smart_skip=True で既存データはスキップ
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import date, datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("batch_scrape")

sys.path.insert(0, ".")


def generate_race_dates(year: int, start_month: int = 1, end_month: int = 12) -> list[str]:
    """指定年の中央競馬開催候補日を新しい順で返す。"""
    dates = []
    d = date(year, end_month, 31) if end_month == 12 else date(year, end_month + 1, 1) - timedelta(days=1)
    start = date(year, start_month, 1)
    while d >= start:
        if d.weekday() in (5, 6):
            dates.append(d.strftime("%Y%m%d"))
        d -= timedelta(days=1)

    holidays = {
        2025: [
            "20250113", "20250211", "20250224", "20250320",
            "20250429", "20250503", "20250504", "20250505", "20250506",
            "20250714", "20250811", "20250915", "20250923",
            "20251103", "20251124",
        ],
        2024: [
            "20240108", "20240212", "20240223", "20240320",
            "20240429", "20240503", "20240504", "20240505", "20240506",
            "20240715", "20240812", "20240916", "20240923",
            "20241104", "20241123",
        ],
        2023: [
            "20230109", "20230211", "20230223", "20230321",
            "20230429", "20230503", "20230504", "20230505",
            "20230717", "20230811", "20230918", "20230923",
            "20231103", "20231123",
        ],
    }

    for h in holidays.get(year, []):
        if h not in dates:
            dates.append(h)

    dates.sort(reverse=True)
    return dates


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--deadline", default="12:30", help="終了時刻 (HH:MM)")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--start-month", type=int, default=1)
    parser.add_argument("--end-month", type=int, default=12)
    args = parser.parse_args()

    hh, mm = map(int, args.deadline.split(":"))
    deadline = datetime.now().replace(hour=hh, minute=mm, second=0, microsecond=0)

    logger.info("=== 一括スクレイピング開始 (期限: %s) ===", deadline.strftime("%H:%M"))

    from scraper.run import ScraperRunner

    runner = ScraperRunner(interval=1.0)
    candidate_dates = generate_race_dates(args.year, args.start_month, args.end_month)

    already_done = set(runner.storage.list_keys("race_lists"))
    shutuba_keys = set(runner.storage.list_keys("race_shutuba", year=str(args.year)))
    result_keys = set(runner.storage.list_keys("race_result", year=str(args.year)))

    total_races = 0
    total_dates = 0
    skipped_dates = 0

    for race_date in candidate_dates:
        if datetime.now() >= deadline:
            logger.info("期限到達 (%s)。終了します。", args.deadline)
            break

        if race_date in already_done:
            race_list_data = runner.storage.load("race_lists", race_date)
            if race_list_data:
                race_ids = [r["race_id"] for r in race_list_data.get("races", [])]
                has_shutuba = all(rid in shutuba_keys for rid in race_ids)
                has_result = all(rid in result_keys for rid in race_ids)
                if has_shutuba and has_result:
                    logger.info("[SKIP] %s: フルデータ済 (%d R)", race_date, len(race_ids))
                    skipped_dates += 1
                    continue

        remaining = (deadline - datetime.now()).total_seconds() / 60
        logger.info("[START] %s (残り %.0f 分)", race_date, remaining)

        try:
            result = runner.scrape_date_all(race_date, smart_skip=True)
            n_races = result.get("total", 0)
            total_races += n_races
            total_dates += 1
            logger.info("[DONE] %s: %d レース完了", race_date, n_races)
        except Exception as e:
            logger.error("[ERROR] %s: %s", race_date, e)
            continue

    runner.close()

    logger.info("=== 完了: %d 日, %d レース (スキップ %d 日) ===",
                total_dates, total_races, skipped_dates)


if __name__ == "__main__":
    main()
