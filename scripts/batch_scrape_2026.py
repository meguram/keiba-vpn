"""
2026年中央競馬 全データ再スクレイピング

Usage:
    # 2フェーズバッチ (推奨: 馬情報の重複取得を自動排除)
    PYTHONPATH=. python scripts/batch_scrape_2026.py
    PYTHONPATH=. python scripts/batch_scrape_2026.py --start-date 20260201
    PYTHONPATH=. python scripts/batch_scrape_2026.py --month 1          # 1月だけ
    PYTHONPATH=. python scripts/batch_scrape_2026.py --smart-skip       # 鮮度チェック有効

    # 旧モード (レース単位逐次)
    PYTHONPATH=. python scripts/batch_scrape_2026.py --legacy
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/tmp/batch_scrape_2026.log", mode="w"),
    ],
)
logger = logging.getLogger("batch_scrape")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scraper.run import ScraperRunner
from scraper.storage import HybridStorage

RACE_LIST_PATH = Path("/tmp/2026_races.json")


def _load_race_map(start_date: str, end_date: str) -> dict[str, list[str]]:
    """保存済みレースリストから対象日付のrace_mapを構築する。"""
    if RACE_LIST_PATH.exists():
        with open(RACE_LIST_PATH) as f:
            all_dates = json.load(f)
        return {
            d: ids for d, ids in sorted(all_dates.items())
            if start_date <= d <= end_date
        }
    return {}


def run_batch(args):
    """2フェーズバッチ: batch_scrape_dates() を使用。"""
    race_map = _load_race_map(args.start_date, args.end_date)

    if args.month:
        month_prefix = f"2026{args.month:02d}"
        race_map = {
            d: ids for d, ids in race_map.items()
            if d.startswith(month_prefix)
        }

    if not race_map:
        if args.month:
            logger.info("レースリストなし。month=%d でレース一覧を取得します", args.month)
            runner = ScraperRunner(interval=1.5, cache=True, auto_login=True)
            try:
                result = runner.batch_scrape_dates(
                    year="2026", month=args.month,
                    smart_skip=args.smart_skip,
                )
                _print_summary(result)
            finally:
                runner.close()
            return
        else:
            logger.error("レースリストが見つかりません: %s", RACE_LIST_PATH)
            return

    total_races = sum(len(v) for v in race_map.values())
    dates = sorted(race_map.keys())

    logger.info("=" * 60)
    logger.info("2026年中央競馬 効率バッチスクレイピング (2フェーズ)")
    logger.info("  対象: %d 開催日 / %d レース", len(dates), total_races)
    logger.info("  期間: %s 〜 %s", dates[0], dates[-1])
    logger.info("  smart_skip=%s", args.smart_skip)
    logger.info("=" * 60)

    if args.dry_run:
        for d in dates:
            logger.info("  %s: %d races", d, len(race_map[d]))
        logger.info("(dry-run: 実行しません)")
        return

    runner = ScraperRunner(interval=1.5, cache=True, auto_login=True)
    try:
        result = runner.batch_scrape_dates(
            race_map=race_map,
            smart_skip=args.smart_skip,
        )
        _print_summary(result)

        logger.info("\n=== カテゴリ別データ件数 ===")
        storage = HybridStorage(".")
        for cat in [
            "race_result", "race_shutuba", "race_index", "race_shutuba_past",
            "race_odds", "race_paddock", "race_barometer", "race_oikiri",
            "smartrc_race", "horse_result",
        ]:
            count = storage.count(cat, "2026")
            logger.info("  %s: %d", cat, count)
    finally:
        runner.close()


def run_legacy(args):
    """旧モード: レース単位逐次取得。"""
    import random

    race_map = _load_race_map(args.start_date, args.end_date)
    if not race_map:
        logger.error("レースリストが見つかりません: %s", RACE_LIST_PATH)
        return

    dates = sorted(race_map.keys())
    total_races = sum(len(race_map[d]) for d in dates)

    logger.info("=" * 60)
    logger.info("2026年中央競馬 レガシーバッチ (レース単位逐次)")
    logger.info("  対象: %d 開催日 / %d レース", len(dates), total_races)
    logger.info("=" * 60)

    if args.dry_run:
        for d in dates:
            logger.info("  %s: %d races", d, len(race_map[d]))
        return

    runner = ScraperRunner(interval=1.5, cache=True, auto_login=True)
    storage = HybridStorage(".")
    grand_total = 0
    grand_errors = 0
    start_time = time.time()

    try:
        for di, date in enumerate(dates):
            race_ids = race_map[date]
            logger.info("\n%s", "=" * 50)
            logger.info("[日付 %d/%d] %s: %d レース", di + 1, len(dates), date, len(race_ids))

            for ri, race_id in enumerate(race_ids):
                logger.info("  [%d/%d] %s", ri + 1, len(race_ids), race_id)
                try:
                    result = runner.scrape_race_all(race_id, smart_skip=args.smart_skip)
                    summary = result.get("summary", {})
                    n_horse_skipped = len([
                        s for s in result.get("skipped", [])
                        if s.startswith("horse_result/")
                    ])
                    logger.info(
                        "    -> horses=%d (skip=%d/%d), smartrc=%s",
                        summary.get("horses_scraped", 0),
                        n_horse_skipped,
                        summary.get("total_horses", 0),
                        summary.get("has_smartrc", False),
                    )
                    grand_total += 1
                except Exception as e:
                    logger.error("    ERROR: %s", e, exc_info=True)
                    grand_errors += 1

                if ri < len(race_ids) - 1:
                    time.sleep(random.uniform(3.0, 6.0))

            if di < len(dates) - 1:
                elapsed = time.time() - start_time
                races_done = sum(len(race_map[dates[j]]) for j in range(di + 1))
                rate = races_done / elapsed if elapsed > 0 else 0
                eta = (total_races - races_done) / rate if rate > 0 else 0
                logger.info("  進捗: %d/%d (%.1f%%) ETA: %.1f分",
                            races_done, total_races,
                            races_done / total_races * 100, eta / 60)
                time.sleep(random.uniform(30.0, 60.0))

        elapsed = time.time() - start_time
        logger.info("\n%s", "=" * 60)
        logger.info("完了: %d/%d レース, エラー=%d, 所要=%.1f分",
                     grand_total, total_races, grand_errors, elapsed / 60)

        logger.info("\n=== カテゴリ別データ件数 ===")
        for cat in [
            "race_result", "race_shutuba", "race_index", "race_shutuba_past",
            "race_odds", "race_paddock", "race_barometer", "race_oikiri",
            "smartrc_race", "horse_result",
        ]:
            count = storage.count(cat, "2026")
            logger.info("  %s: %d", cat, count)
    finally:
        runner.close()


def _print_summary(result: dict):
    logger.info("\n%s", "=" * 60)
    logger.info("バッチ完了サマリー")
    logger.info("  日付数: %d", result.get("dates", 0))
    logger.info("  レース数: %d", result.get("races", 0))
    logger.info("  ユニーク馬数: %d", result.get("horses_total", 0))
    logger.info("    取得: %d", result.get("horses_fetched", 0))
    logger.info("    スキップ: %d", result.get("horses_skipped", 0))
    logger.info("  エラー: %d", result.get("errors", 0))
    logger.info("  Phase 1 (レース): %.1f分", result.get("phase1_minutes", 0))
    logger.info("  Phase 2 (馬情報): %.1f分", result.get("phase2_minutes", 0))
    logger.info("  合計: %.1f分", result.get("total_minutes", 0))
    logger.info("=" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="2026年中央競馬 バッチスクレイピング"
    )
    parser.add_argument("--start-date", default="20260104")
    parser.add_argument("--end-date", default="20260315")
    parser.add_argument("--month", type=int, default=0,
                        help="月指定 (1-12)。指定時はその月のみ対象")
    parser.add_argument("--smart-skip", action="store_true",
                        help="レースデータの鮮度チェックを有効化")
    parser.add_argument("--legacy", action="store_true",
                        help="旧モード (レース単位逐次取得)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.legacy:
        run_legacy(args)
    else:
        run_batch(args)


if __name__ == "__main__":
    main()
