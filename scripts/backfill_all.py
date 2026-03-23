#!/usr/bin/env python3
"""
欠損データ一括バックフィルスクリプト

Phase 1: レースカレンダー取得（直近開催の未取得分を特定）
Phase 2: レースデータ欠損補完（厩舎コメント / バロメーター / 追い切り / パドック / 指数）
Phase 3: 馬個別データ（horse_result / pedigree_5gen）の欠損補完

Usage:
    python scripts/backfill_all.py [--phase 1|2|3|all] [--limit N] [--dry-run]
"""

import argparse
import datetime
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scraper.client import NetkeibaClient
from scraper.run import ScraperRunner
from scraper.storage import HybridStorage

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"backfill_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class AccessError(Exception):
    pass


class StopFlag:
    def __init__(self):
        self._stopped = False

    def stop(self):
        self._stopped = True

    @property
    def is_stopped(self):
        return self._stopped


def collect_all_horse_ids(storage: HybridStorage) -> set[str]:
    logger.info("全レースから馬IDを収集中...")
    horse_ids = set()
    for cat in ("race_shutuba", "race_result"):
        keys = storage.list_keys(cat)
        for rid in keys:
            data = storage.load(cat, rid)
            if not data or "entries" not in data:
                continue
            for e in data["entries"]:
                hid = e.get("horse_id", "")
                if hid:
                    horse_ids.add(hid)
    logger.info("馬ID収集完了: %d 頭", len(horse_ids))
    return horse_ids


def phase1_calendar(runner: ScraperRunner, storage: HybridStorage, dry_run: bool = False):
    """直近開催日のレース一覧を取得し、未取得レースを特定"""
    logger.info("=== Phase 1: レースカレンダー取得 ===")

    today = datetime.date.today()
    dates_to_check = []
    for i in range(60):
        d = today - datetime.timedelta(days=i)
        if d.weekday() in (5, 6):
            dates_to_check.append(d)

    existing_shutuba = set(storage.list_keys("race_shutuba"))
    existing_result = set(storage.list_keys("race_result"))

    new_race_ids = []
    for d in dates_to_check:
        ds = d.strftime("%Y%m%d")
        try:
            races = runner.scrape_race_list(ds)
            if not races:
                continue
            logger.info("  %s: %d レース発見", ds, len(races))
            for race in races:
                rid = race.get("race_id", "") if isinstance(race, dict) else str(race)
                if rid and rid not in existing_shutuba and rid not in existing_result:
                    new_race_ids.append(rid)
        except Exception as e:
            logger.warning("  %s: カレンダー取得失敗 (%s)", ds, e)
            time.sleep(2)

    logger.info("Phase 1 完了: 新規レース %d 件", len(new_race_ids))
    return new_race_ids


def phase2_race_data(runner: ScraperRunner, storage: HybridStorage,
                     new_race_ids: list[str], limit: int = 0,
                     dry_run: bool = False):
    """レースデータの欠損補完"""
    logger.info("=== Phase 2: レースデータ欠損補完 ===")

    shutuba_keys = set(storage.list_keys("race_shutuba"))
    result_keys = set(storage.list_keys("race_result"))
    all_race_ids = sorted(shutuba_keys | result_keys | set(new_race_ids))

    RACE_CATS = [
        ("race_trainer_comment", "scrape_trainer_comment"),
        ("race_barometer", "_scrape_barometer_with_session"),
        ("race_oikiri", "scrape_oikiri"),
        ("race_paddock", "scrape_paddock"),
        ("race_index", "scrape_speed_index"),
        ("race_shutuba_past", "scrape_shutuba_past"),
    ]

    existing = {}
    for cat, _ in RACE_CATS:
        existing[cat] = set(storage.list_keys(cat))

    tasks = []
    for rid in all_race_ids:
        for cat, method_name in RACE_CATS:
            if rid not in existing[cat]:
                tasks.append((rid, cat, method_name))

    # Also scrape full data for new races
    for rid in new_race_ids:
        if rid not in shutuba_keys:
            tasks.insert(0, (rid, "race_shutuba", "scrape_race_card"))
        if rid not in result_keys:
            tasks.insert(0, (rid, "race_result", "scrape_race_result"))

    if limit > 0:
        tasks = tasks[:limit]

    logger.info("Phase 2 タスク数: %d", len(tasks))
    if dry_run:
        for rid, cat, _ in tasks[:20]:
            logger.info("  [DRY-RUN] %s / %s", cat, rid)
        return

    stop = StopFlag()
    success = 0
    errors = 0
    skipped = 0

    for i, (rid, cat, method_name) in enumerate(tasks):
        if stop.is_stopped:
            logger.warning("StopFlag 検知 — 中断")
            break

        try:
            method = getattr(runner, method_name, None)
            if method is None:
                logger.warning("メソッド %s が見つかりません", method_name)
                skipped += 1
                continue

            logger.info("[%d/%d] %s / %s", i + 1, len(tasks), cat, rid)
            result = method(rid, skip_existing=True)
            if result:
                success += 1
            else:
                skipped += 1
            time.sleep(1)

        except Exception as e:
            errors += 1
            err_msg = str(e)
            if "403" in err_msg or "429" in err_msg or "503" in err_msg:
                logger.error("アクセスエラー: %s — 10秒待機", err_msg[:100])
                time.sleep(10)
                if errors >= 5:
                    logger.error("エラー連続 %d 回 — Phase 2 中断", errors)
                    stop.stop()
            else:
                logger.warning("エラー: %s / %s — %s", cat, rid, err_msg[:100])
                time.sleep(2)

    logger.info("Phase 2 完了: 成功=%d, スキップ=%d, エラー=%d", success, skipped, errors)


def _scrape_one_pedigree(horse_id: str, client: NetkeibaClient,
                         storage: HybridStorage) -> bool:
    """1馬分の5代血統を取得・保存"""
    from research.pedigree_similarity import parse_blood_table_5gen
    from scripts.scrape_pedigree_5gen import build_pedigree_record, PED_URL

    url = PED_URL.format(horse_id=horse_id)
    html = client.fetch(url)
    ancestors = parse_blood_table_5gen(html)
    if len(ancestors) < 5:
        return False
    record = build_pedigree_record(horse_id, ancestors, source="backfill")
    storage.save("horse_pedigree_5gen", horse_id, record)
    return True


def phase3_horse_data(runner: ScraperRunner, storage: HybridStorage,
                      limit: int = 0, dry_run: bool = False):
    """馬個別データの欠損補完"""
    logger.info("=== Phase 3: 馬個別データ補完 ===")

    all_horse_ids = collect_all_horse_ids(storage)
    hr_keys = set(storage.list_keys("horse_result"))
    ped5_keys = set(storage.list_keys("horse_pedigree_5gen"))

    missing_hr = sorted(all_horse_ids - hr_keys)
    missing_ped = sorted(all_horse_ids - ped5_keys)

    logger.info("horse_result 未取得: %d / pedigree_5gen 未取得: %d",
                len(missing_hr), len(missing_ped))

    tasks: list[tuple[str, str]] = []
    for hid in missing_hr:
        tasks.append((hid, "horse_result"))
    for hid in missing_ped:
        tasks.append((hid, "pedigree_5gen"))

    if limit > 0:
        tasks = tasks[:limit]

    logger.info("Phase 3 タスク数: %d", len(tasks))
    if dry_run:
        for hid, cat in tasks[:20]:
            logger.info("  [DRY-RUN] %s / %s", cat, hid)
        return

    client = NetkeibaClient(interval=3.0, auto_login=True)
    stop = StopFlag()
    success = 0
    errors = 0
    consecutive_errors = 0

    for i, (hid, cat) in enumerate(tasks):
        if stop.is_stopped:
            logger.warning("StopFlag 検知 — 中断")
            break

        try:
            logger.info("[%d/%d] %s / %s", i + 1, len(tasks), cat, hid)
            if cat == "horse_result":
                result = runner.scrape_horse(hid, skip_existing=True)
                ok = result is not None
            else:
                ok = _scrape_one_pedigree(hid, client, storage)

            if ok:
                success += 1
            consecutive_errors = 0
            time.sleep(1.5)

        except Exception as e:
            errors += 1
            consecutive_errors += 1
            err_msg = str(e)
            if "403" in err_msg or "429" in err_msg or "503" in err_msg:
                logger.error("アクセスエラー: %s — 15秒待機", err_msg[:100])
                time.sleep(15)
                if consecutive_errors >= 3:
                    logger.error("連続エラー %d 回 — Phase 3 中断", consecutive_errors)
                    stop.stop()
            else:
                logger.warning("エラー: %s / %s — %s", cat, hid, err_msg[:100])
                time.sleep(3)

        if (i + 1) % 100 == 0:
            logger.info("--- 進捗: %d/%d (成功=%d, エラー=%d) ---",
                        i + 1, len(tasks), success, errors)

    logger.info("Phase 3 完了: 成功=%d, エラー=%d", success, errors)


def main():
    parser = argparse.ArgumentParser(description="欠損データ一括バックフィル")
    parser.add_argument("--phase", choices=["1", "2", "3", "all"], default="all")
    parser.add_argument("--limit", type=int, default=0, help="各Phaseの最大タスク数 (0=無制限)")
    parser.add_argument("--dry-run", action="store_true", help="実行せずタスク一覧のみ表示")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("バックフィル開始 phase=%s limit=%d dry_run=%s", args.phase, args.limit, args.dry_run)
    logger.info("ログファイル: %s", log_file)
    logger.info("=" * 60)

    storage = HybridStorage()
    runner = ScraperRunner()

    new_race_ids = []

    if args.phase in ("1", "all"):
        new_race_ids = phase1_calendar(runner, storage, dry_run=args.dry_run)

    if args.phase in ("2", "all"):
        phase2_race_data(runner, storage, new_race_ids, limit=args.limit, dry_run=args.dry_run)

    if args.phase in ("3", "all"):
        phase3_horse_data(runner, storage, limit=args.limit, dry_run=args.dry_run)

    logger.info("=" * 60)
    logger.info("全Phase完了")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
