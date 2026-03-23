"""
過去データ一括取得 (Backfill)

JRA中央競馬の過去レースデータを効率的かつ安全に取得する。
cronジョブとして定期実行し、中断・再開を完全サポートする。

設計方針:
  - 進捗は JSON ファイルで永続化 (data/meta/backfill_progress.json)
  - 取得済み日付を記録し、再実行時はスキップ (レジューム)
  - bot検出回避のため長めのクールダウンを挿入
  - 複数プロセスで年度を分けて並列実行可能 (--year オプション)
  - 予測モデル学習に必要な最小セットを優先取得する「fast モード」

データ取得フェーズ:
  Phase 1 (fast):  race_list → race_result → race_shutuba
  Phase 2 (horse): 上記で得た horse_id → horse_result (profile + result + ped)
  Phase 3 (full):  race_index, race_odds, race_oikiri 等の補助データ

使い方:
  # 全年度を順次取得 (2020〜現在)
  python -m scraper.backfill

  # 特定年度のみ (並列用)
  python -m scraper.backfill --year 2024

  # Phase 指定
  python -m scraper.backfill --phase fast
  python -m scraper.backfill --phase horse
  python -m scraper.backfill --phase full

  # ドライラン (取得せず計画のみ表示)
  python -m scraper.backfill --dry-run

  # 進捗確認
  python -m scraper.backfill --status

cron設定例:
  # 4並列で年度分割 (毎日 AM 2:00〜)
  0 2 * * * cd /home/user/project/myproject/keiba && python -m scraper.backfill --year 2024 >> logs/backfill_2024.log 2>&1
  0 2 * * * cd /home/user/project/myproject/keiba && python -m scraper.backfill --year 2025 >> logs/backfill_2025.log 2>&1
  15 2 * * * cd /home/user/project/myproject/keiba && python -m scraper.backfill --year 2022 >> logs/backfill_2022.log 2>&1
  15 2 * * * cd /home/user/project/myproject/keiba && python -m scraper.backfill --year 2023 >> logs/backfill_2023.log 2>&1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("scraper.backfill")

PROGRESS_PATH = Path("data/meta/backfill_progress.json")
LOCK_DIR = Path("data/meta/backfill_locks")

DEFAULT_START_YEAR = 2020
MAX_DATES_PER_RUN = 10
MAX_RUNTIME_HOURS = 6

_PHASE_CATEGORIES = {
    "fast": ["race_result"],
    "horse": [],
    "full": [
        "race_index", "race_shutuba_past",
        "race_paddock", "race_oikiri",
    ],
}

# 過去レースでは出馬表 (shutuba) は取得不可（ページが削除される）
# race_odds, race_barometer も過去データでは取得困難
# 予測モデルの学習には race_result + horse_result が最重要

_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    logger.info("シャットダウン要求受信 (signal=%d) — 現在のレース完了後に停止します", signum)
    _shutdown_requested = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


class BackfillProgress:
    """進捗状態の永続管理。"""

    def __init__(self, path: Path = PROGRESS_PATH):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def save(self):
        self._path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def is_date_done(self, date: str, phase: str) -> bool:
        return date in self._data.get(f"done_{phase}", [])

    def mark_date_done(self, date: str, phase: str, stats: dict | None = None):
        key = f"done_{phase}"
        if key not in self._data:
            self._data[key] = []
        if date not in self._data[key]:
            self._data[key].append(date)
            self._data[key] = sorted(self._data[key])

        stats_key = f"stats_{phase}"
        if stats_key not in self._data:
            self._data[stats_key] = {}
        if stats:
            self._data[stats_key][date] = stats

        self._data["last_updated"] = datetime.now().isoformat()
        self.save()

    def is_horse_done(self, horse_id: str) -> bool:
        return horse_id in self._data.get("done_horses", set())

    def mark_horses_done(self, horse_ids: list[str]):
        if "done_horses" not in self._data:
            self._data["done_horses"] = []
        existing = set(self._data["done_horses"])
        existing.update(horse_ids)
        self._data["done_horses"] = sorted(existing)
        self._data["last_updated"] = datetime.now().isoformat()
        self.save()

    def get_summary(self) -> dict:
        summary = {
            "last_updated": self._data.get("last_updated", "—"),
        }
        for phase in ["fast", "horse", "full"]:
            key = f"done_{phase}"
            dates = self._data.get(key, [])
            summary[f"{phase}_dates"] = len(dates)
            if dates:
                summary[f"{phase}_oldest"] = dates[0]
                summary[f"{phase}_newest"] = dates[-1]
        summary["horses_done"] = len(self._data.get("done_horses", []))

        total_races = 0
        for phase in ["fast", "full"]:
            for date, stats in self._data.get(f"stats_{phase}", {}).items():
                total_races = max(total_races,
                                  total_races + stats.get("races_new", 0))
        summary["total_races_approx"] = total_races
        return summary


class BackfillLock:
    """ファイルベースの排他ロック。同じ年度の重複実行を防止。"""

    def __init__(self, year: int | str):
        LOCK_DIR.mkdir(parents=True, exist_ok=True)
        self._path = LOCK_DIR / f"backfill_{year}.lock"
        self._acquired = False

    def acquire(self) -> bool:
        if self._path.exists():
            try:
                info = json.loads(self._path.read_text())
                pid = info.get("pid", 0)
                started = info.get("started", "")
                if pid and _pid_alive(pid):
                    logger.error(
                        "年度 %s は既に実行中です (PID=%d, 開始=%s)",
                        self._path.stem, pid, started,
                    )
                    return False
                else:
                    logger.warning("古いロックを削除: %s (PID=%d は停止済み)", self._path, pid)
            except Exception:
                pass

        self._path.write_text(json.dumps({
            "pid": os.getpid(),
            "started": datetime.now().isoformat(),
        }))
        self._acquired = True
        return True

    def release(self):
        if self._acquired and self._path.exists():
            self._path.unlink(missing_ok=True)
            self._acquired = False

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError("ロック取得失敗")
        return self

    def __exit__(self, *args):
        self.release()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _generate_race_dates(year: int) -> list[str]:
    """指定年の JRA 開催可能日（土日祝＋正月）を生成する。"""
    dates = []
    start = datetime(year, 1, 1)
    end_date = min(datetime(year, 12, 31), datetime.now() - timedelta(days=1))

    current = start
    while current <= end_date:
        if current.weekday() in (5, 6):
            dates.append(current.strftime("%Y%m%d"))
        elif current.month == 1 and current.day <= 6:
            dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)

    return dates


def _run_phase_fast(runner, progress: BackfillProgress,
                    dates: list[str], max_dates: int) -> dict:
    """Phase 1: race_list + race_result + race_shutuba を取得。"""
    stats = {"dates_processed": 0, "dates_skipped": 0, "races_total": 0, "errors": 0}
    start_time = time.time()

    pending = [d for d in dates if not progress.is_date_done(d, "fast")]
    logger.info("Phase fast: %d/%d 日が未処理", len(pending), len(dates))

    for i, date in enumerate(pending[:max_dates]):
        if _shutdown_requested:
            logger.info("シャットダウン要求 — Phase fast 中断")
            break

        if time.time() - start_time > MAX_RUNTIME_HOURS * 3600:
            logger.info("最大実行時間 (%dh) 超過 — 中断", MAX_RUNTIME_HOURS)
            break

        logger.info("=== [fast %d/%d] %s ===", i + 1, min(len(pending), max_dates), date)

        try:
            races = runner.scrape_race_list(date)
            if not races:
                logger.info("レースなし: %s — スキップ", date)
                progress.mark_date_done(date, "fast", {"races": 0, "races_new": 0})
                stats["dates_skipped"] += 1
                continue

            races_new = 0
            for j, race in enumerate(races):
                if _shutdown_requested:
                    break
                rid = race["race_id"]

                if runner.storage.exists("race_result", rid):
                    continue

                logger.info("  [%d/%d] %s %s", j + 1, len(races), rid,
                            race.get("race_name", ""))
                try:
                    runner.scrape_race_result(rid, skip_existing=False)
                    races_new += 1
                except Exception as e:
                    logger.warning("  race_result/%s 失敗: %s", rid, e)

            if not _shutdown_requested:
                progress.mark_date_done(date, "fast", {
                    "races": len(races),
                    "races_new": races_new,
                })
                stats["dates_processed"] += 1
                stats["races_total"] += len(races)

            if i < min(len(pending), max_dates) - 1:
                pause = random.uniform(10.0, 30.0)
                logger.info("日付間クールダウン: %.0f 秒", pause)
                time.sleep(pause)

        except Exception as e:
            logger.error("Phase fast エラー [%s]: %s", date, e)
            stats["errors"] += 1
            time.sleep(random.uniform(30, 60))

    return stats


def _run_phase_horse(runner, progress: BackfillProgress,
                     max_horses: int = 500) -> dict:
    """Phase 2: 取得済みレースの出走馬情報を取得。"""
    stats = {"horses_processed": 0, "horses_skipped": 0, "errors": 0}
    start_time = time.time()

    horse_ids: set[str] = set()
    for category in ["race_result", "race_shutuba"]:
        for key in runner.storage.list_keys(category):
            data = runner.storage.load(category, key)
            if data:
                for entry in data.get("entries", []):
                    hid = entry.get("horse_id", "")
                    if hid:
                        horse_ids.add(hid)

    done_horses = set(progress._data.get("done_horses", []))
    pending = sorted(horse_ids - done_horses)

    logger.info("Phase horse: %d/%d 頭が未処理", len(pending), len(horse_ids))

    batch_done: list[str] = []

    for i, hid in enumerate(pending[:max_horses]):
        if _shutdown_requested:
            break
        if time.time() - start_time > MAX_RUNTIME_HOURS * 3600:
            logger.info("最大実行時間超過 — Phase horse 中断")
            break

        existing = runner.storage.load("horse_result", hid)
        if existing and existing.get("sire"):
            batch_done.append(hid)
            stats["horses_skipped"] += 1
            continue

        try:
            runner.scrape_horse(hid, skip_existing=False, with_history=True)
            batch_done.append(hid)
            stats["horses_processed"] += 1

            if len(batch_done) % 50 == 0:
                progress.mark_horses_done(batch_done)
                batch_done = []
                logger.info("  進捗保存: %d頭完了", stats["horses_processed"] + stats["horses_skipped"])

        except Exception as e:
            logger.error("馬情報取得エラー [%s]: %s", hid, e)
            stats["errors"] += 1
            time.sleep(random.uniform(10, 30))

    if batch_done:
        progress.mark_horses_done(batch_done)

    return stats


def _run_phase_full(runner, progress: BackfillProgress,
                    dates: list[str], max_dates: int) -> dict:
    """Phase 3: 補助データ (index, odds, oikiri 等) を取得。"""
    stats = {"dates_processed": 0, "races_total": 0, "errors": 0}
    start_time = time.time()

    fast_done = set(progress._data.get("done_fast", []))
    full_done = set(progress._data.get("done_full", []))
    pending = [d for d in dates if d in fast_done and d not in full_done]

    logger.info("Phase full: %d 日が未処理 (fast完了: %d)", len(pending), len(fast_done))

    categories = _PHASE_CATEGORIES["full"]

    for i, date in enumerate(pending[:max_dates]):
        if _shutdown_requested:
            break
        if time.time() - start_time > MAX_RUNTIME_HOURS * 3600:
            break

        logger.info("=== [full %d/%d] %s ===", i + 1, min(len(pending), max_dates), date)

        race_list = runner.storage.load("race_lists", date)
        if not race_list:
            progress.mark_date_done(date, "full", {"races": 0})
            continue

        races = race_list.get("races", [])
        for j, race in enumerate(races):
            if _shutdown_requested:
                break
            rid = race["race_id"]

            for cat in categories:
                method_map = {
                    "race_index": "scrape_speed_index",
                    "race_shutuba_past": "scrape_shutuba_past",
                    "race_odds": "scrape_odds",
                    "race_paddock": "scrape_paddock",
                    "race_barometer": "scrape_barometer",
                    "race_oikiri": "scrape_oikiri",
                }
                method_name = method_map.get(cat)
                if not method_name:
                    continue
                if runner.storage.exists(cat, rid):
                    continue
                try:
                    getattr(runner, method_name)(rid, skip_existing=False)
                except Exception as e:
                    logger.warning("  %s/%s エラー: %s", cat, rid, e)

        if not _shutdown_requested:
            progress.mark_date_done(date, "full", {"races": len(races)})
            stats["dates_processed"] += 1
            stats["races_total"] += len(races)

        if i < min(len(pending), max_dates) - 1:
            pause = random.uniform(10.0, 30.0)
            time.sleep(pause)

    return stats


def run_backfill(
    year: int | None = None,
    phase: str = "auto",
    max_dates: int = MAX_DATES_PER_RUN,
    dry_run: bool = False,
):
    """バックフィルのメインエントリポイント。"""
    from scraper.run import ScraperRunner

    now = datetime.now()
    years = [year] if year else list(range(DEFAULT_START_YEAR, now.year + 1))

    progress = BackfillProgress()

    all_dates: list[str] = []
    for y in years:
        all_dates.extend(_generate_race_dates(y))

    logger.info("=== Backfill 開始 ===")
    logger.info("対象年度: %s", years)
    logger.info("対象日数: %d 日", len(all_dates))
    logger.info("Phase: %s", phase)
    logger.info("max_dates/run: %d", max_dates)

    if dry_run:
        fast_pending = [d for d in all_dates if not progress.is_date_done(d, "fast")]
        full_pending = [d for d in all_dates
                        if progress.is_date_done(d, "fast") and not progress.is_date_done(d, "full")]
        print(f"\n[Dry Run] fast 未処理: {len(fast_pending)} 日")
        print(f"[Dry Run] full 未処理: {len(full_pending)} 日")
        print(f"[Dry Run] 馬情報: {len(progress._data.get('done_horses', []))} 頭取得済み")
        if fast_pending:
            print(f"[Dry Run] 次回 fast 対象: {fast_pending[:max_dates]}")
        return

    lock_key = year or "all"
    lock = BackfillLock(lock_key)
    if not lock.acquire():
        return

    try:
        runner = ScraperRunner(interval=2.0, cache=False, auto_login=True)

        if phase in ("auto", "fast"):
            logger.info("--- Phase fast 開始 ---")
            fast_stats = _run_phase_fast(runner, progress, all_dates, max_dates)
            logger.info("Phase fast 結果: %s", fast_stats)

            if not _shutdown_requested and phase == "auto":
                logger.info("--- Phase horse 開始 ---")
                horse_stats = _run_phase_horse(runner, progress, max_horses=300)
                logger.info("Phase horse 結果: %s", horse_stats)

        elif phase == "horse":
            logger.info("--- Phase horse 開始 ---")
            horse_stats = _run_phase_horse(runner, progress, max_horses=500)
            logger.info("Phase horse 結果: %s", horse_stats)

        elif phase == "full":
            logger.info("--- Phase full 開始 ---")
            full_stats = _run_phase_full(runner, progress, all_dates, max_dates)
            logger.info("Phase full 結果: %s", full_stats)

        runner.client.close()

    finally:
        lock.release()

    logger.info("=== Backfill 終了 ===")
    summary = progress.get_summary()
    for k, v in summary.items():
        logger.info("  %s: %s", k, v)


def show_status():
    """現在の進捗状況を表示する。"""
    progress = BackfillProgress()
    summary = progress.get_summary()

    print("\n=== Backfill 進捗状況 ===\n")
    print(f"最終更新: {summary.get('last_updated', '—')}")
    print()
    for phase in ["fast", "horse", "full"]:
        n = summary.get(f"{phase}_dates", 0)
        oldest = summary.get(f"{phase}_oldest", "—")
        newest = summary.get(f"{phase}_newest", "—")
        if phase == "horse":
            n = summary.get("horses_done", 0)
            print(f"  [{phase:>5}] {n:>5} 頭取得済み")
        else:
            print(f"  [{phase:>5}] {n:>5} 日完了  ({oldest} 〜 {newest})")

    now = datetime.now()
    total_possible = 0
    for y in range(DEFAULT_START_YEAR, now.year + 1):
        total_possible += len(_generate_race_dates(y))

    fast_done = summary.get("fast_dates", 0)
    pct = (fast_done / total_possible * 100) if total_possible else 0
    print(f"\n  全体進捗 (fast): {fast_done}/{total_possible} 日 ({pct:.1f}%)")

    lock_files = list(LOCK_DIR.glob("backfill_*.lock")) if LOCK_DIR.exists() else []
    if lock_files:
        print("\n  実行中のジョブ:")
        for lf in lock_files:
            try:
                info = json.loads(lf.read_text())
                alive = "●" if _pid_alive(info.get("pid", 0)) else "✗"
                print(f"    {alive} {lf.stem} (PID={info.get('pid')}, 開始={info.get('started', '?')})")
            except Exception:
                print(f"    ? {lf.stem}")
    print()


def main():
    parser = argparse.ArgumentParser(description="過去データ一括取得 (Backfill)")
    parser.add_argument("--year", type=int, help="対象年度 (例: 2024)")
    parser.add_argument("--phase", default="auto",
                        choices=["auto", "fast", "horse", "full"],
                        help="実行フェーズ (default: auto)")
    parser.add_argument("--max-dates", type=int, default=MAX_DATES_PER_RUN,
                        help=f"1回の実行で処理する最大日数 (default: {MAX_DATES_PER_RUN})")
    parser.add_argument("--dry-run", action="store_true",
                        help="取得せず計画のみ表示")
    parser.add_argument("--status", action="store_true",
                        help="進捗状況を表示して終了")

    args = parser.parse_args()

    os.chdir(Path(__file__).resolve().parent.parent)

    if args.status:
        show_status()
        return

    run_backfill(
        year=args.year,
        phase=args.phase,
        max_dates=args.max_dates,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
