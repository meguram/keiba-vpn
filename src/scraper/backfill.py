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
  python -m src.scraper.backfill

  # 特定年度のみ (並列用)
  python -m src.scraper.backfill --year 2024

  # Phase 指定
  python -m src.scraper.backfill --phase fast
  python -m src.scraper.backfill --phase horse
  python -m src.scraper.backfill --phase full

  # ドライラン (取得せず計画のみ表示)
  python -m src.scraper.backfill --dry-run

  # 進捗確認
  python -m src.scraper.backfill --status

cron設定例:
  # 4並列で年度分割 (毎日 AM 2:00〜)
  0 2 * * * cd /home/user/project/myproject/keiba && python -m src.scraper.backfill --year 2024 >> logs/backfill_2024.log 2>&1
  0 2 * * * cd /home/user/project/myproject/keiba && python -m src.scraper.backfill --year 2025 >> logs/backfill_2025.log 2>&1
  15 2 * * * cd /home/user/project/myproject/keiba && python -m src.scraper.backfill --year 2022 >> logs/backfill_2022.log 2>&1
  15 2 * * * cd /home/user/project/myproject/keiba && python -m src.scraper.backfill --year 2023 >> logs/backfill_2023.log 2>&1
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

from src.utils.keiba_logging import script_basic_config

script_basic_config(handlers=[logging.StreamHandler()])
logger = logging.getLogger("scraper.backfill")


def _update_coverage_safe(date: str, runner: Any) -> None:
    """phase 完了後に date_coverage を安全に更新する（失敗しても続行）。"""
    try:
        from src.scraper.date_coverage import update_date_coverage
        update_date_coverage(date, runner.storage)
    except Exception as e:
        logger.debug("coverage 更新スキップ [%s]: %s", date, e)

PROGRESS_PATH = Path("data/local/meta/backfill_progress.json")
LOCK_DIR = Path("data/local/meta/backfill_locks")

DEFAULT_START_YEAR = 2020
MAX_DATES_PER_RUN = 10
MAX_RUNTIME_HOURS = 6

_PHASE_CATEGORIES = {
    "fast": ["race_result"],
    "horse": [],
    "full": [
        "race_index", "race_shutuba_past",
        "race_paddock", "race_oikiri",
        "race_result_on_time",
    ],
}

# 過去レースでは出馬表 (shutuba) は取得不可（ページが削除される）
# race_odds, race_barometer も過去データでは取得困難
# 予測モデルの学習には race_result + horse_result が最重要

_shutdown_requested = False
_GUARD = None   # 過去分補填（--phase past）のときだけ設定する NightGuard（件数の上限・悪化の検知・時間窓）


def _should_stop() -> bool:
    """シャットダウン要求、または過去分補填の夜ごとの安全装置が止めるべきと言っているか。"""
    return _shutdown_requested or bool(_GUARD is not None and _GUARD.should_stop())


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
        if _should_stop():
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
                if _should_stop():
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

            if not _should_stop():
                progress.mark_date_done(date, "fast", {
                    "races": len(races),
                    "races_new": races_new,
                })
                stats["dates_processed"] += 1
                stats["races_total"] += len(races)
                _update_coverage_safe(date, runner)

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
        if _should_stop():
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
        if _should_stop():
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
            if _should_stop():
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
                    "race_result_on_time": "scrape_race_result_on_time",
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

        if not _should_stop():
            progress.mark_date_done(date, "full", {"races": len(races)})
            stats["dates_processed"] += 1
            stats["races_total"] += len(races)
            _update_coverage_safe(date, runner)

        if i < min(len(pending), max_dates) - 1:
            pause = random.uniform(10.0, 30.0)
            time.sleep(pause)

    return stats


# ── 過去分（2016〜2019 年）の補填 ─────────────────────────────────────────────
# 2020 年以降と同じカテゴリ・スキーマ・保存経路（HybridStorage.save → GCS）で揃える。取得の強さは backfill_guard が制限する。
# 段階（上から順に、未完了の段階を進める）。年は新しいほうから（2019 → 2016）。
PAST_YEARS = [2019, 2018, 2017, 2016]
PAST_TIERS = ["fast", "horse", "cards", "lap", "odds", "pair", "index", "past", "on_time", "anchors_f"]
# 2019 年のレースで実際に取れることを確認したカテゴリだけ。パドック・厩舎コメント・調子偏差値は 2019 年のページに内容が無い（空の文書になる）ので取らない。
_PAST_CARD_TIERS = {
    "cards": [("race_shutuba", "scrape_race_card"), ("race_oikiri", "scrape_oikiri")],
    "lap": [("race_result_lap", "scrape_race_result_lap")],
    "odds": [("race_odds", "scrape_odds")],
    "pair": [("race_pair_odds", "scrape_pair_odds")],
    "index": [("race_index", "scrape_speed_index")],
    "past": [("race_shutuba_past", "scrape_shutuba_past")],
    "on_time": [("race_result_on_time", "scrape_race_result_on_time")],
}
_API_SESSION_CATS = {"race_odds", "race_pair_odds"}
_PAST_UNAVAILABLE = {
    "race_paddock": "2019 年のページに内容が無い（空）",
    "race_trainer_comment": "2019 年のページに内容が無い（空）",
    "race_barometer": "2019 年は取得できない（空）",
}
CALENDAR_URL = "https://race.netkeiba.com/top/calendar.html?year={year}&month={month}"


def _past_race_dates(runner, progress: BackfillProgress, year: int) -> list[str]:
    """その年の JRA 開催日。月間カレンダー（12 リクエスト/年）から取り、進捗ファイルに保存して使い回す。"""
    import re

    key = f"calendar_{year}"
    cached = progress._data.get(key)
    if cached:
        return cached
    dates: set[str] = set()
    for month in range(1, 13):
        if _should_stop():
            return []
        html = runner.client.fetch(CALENDAR_URL.format(year=year, month=month))
        dates.update(d for d in re.findall(r"kaisai_date=(\d{8})", html) if d.startswith(str(year)))
    out = sorted(dates)
    progress._data[key] = out
    progress.save()
    return out


def _run_phase_cards_past(runner, progress: BackfillProgress, dates: list[str], tier: str) -> dict:
    """出馬表・追い切り・オッズ・タイム指数を、結果が揃った開催日について取る（既にあるカテゴリは取らない）。"""
    cats = _PAST_CARD_TIERS[tier]
    key = f"done_{tier}"
    stats = {"dates_processed": 0, "fetched": 0, "missing": 0}
    done = set(progress._data.get(key, []))
    fast_done = set(progress._data.get("done_fast", []))
    pending = [d for d in dates if d in fast_done and d not in done]
    logger.info("Phase %s(past): %d 日が未処理", tier, len(pending))
    for date in pending:
        if _should_stop():
            break
        race_list = runner.storage.load("race_lists", date) or {}
        races = race_list.get("races", [])
        got = miss = 0
        consecutive_miss = 0
        for race in races:
            if _should_stop():
                break
            rid = race["race_id"]
            for cat, method in cats:
                if _should_stop():
                    break
                if runner.storage.exists(cat, rid):
                    continue
                try:
                    if cat == "race_result_on_time":
                        data = getattr(runner, method)(rid, skip_existing=False, opening_date=date)
                    else:
                        data = getattr(runner, method)(rid, skip_existing=False)
                except Exception as e:  # noqa: BLE001
                    logger.warning("  %s/%s 失敗: %s", cat, rid, e)
                    data = None
                if cat in _API_SESSION_CATS:
                    # API セッションの取得は NetkeibaClient のスロットルとオブザーバを通らない。間隔を空け、件数に足す
                    if _GUARD is not None:
                        _GUARD.add_external(1)
                    time.sleep(random.uniform(2.2, 4.0))
                if data:
                    got += 1
                    consecutive_miss = 0
                else:
                    miss += 1
                    consecutive_miss += 1
                    if consecutive_miss >= 5 and _GUARD is not None:
                        _GUARD.fail(f"{cat} が連続して取得できない（{consecutive_miss} 件）")
        if not _should_stop():
            progress.mark_date_done(date, tier, {"races": len(races), "fetched": got, "missing": miss})
            stats["dates_processed"] += 1
            stats["fetched"] += got
            stats["missing"] += miss
            _update_coverage_safe(date, runner)
    return stats


def _collect_past_horses(runner, progress: BackfillProgress, dates: list[str]) -> int:
    """結果が揃った開催日の出走馬を集める（ストレージの読み込みだけで、netkeiba へは取りに行かない）。"""
    collected = set(progress._data.get("past_horse_dates", []))
    horses = set(progress._data.get("past_horse_ids", []))
    fast_done = set(progress._data.get("done_fast", []))
    n_new = 0
    for date in dates:
        if date in collected or date not in fast_done:
            continue
        for race in (runner.storage.load("race_lists", date) or {}).get("races", []):
            res = runner.storage.load("race_result", race["race_id"]) or {}
            for e in res.get("entries", []):
                hid = e.get("horse_id")
                if hid and hid not in horses:
                    horses.add(hid)
                    n_new += 1
        collected.add(date)
    progress._data["past_horse_ids"] = sorted(horses)
    progress._data["past_horse_dates"] = sorted(collected)
    progress.save()
    return n_new


def _run_phase_horse_past(runner, progress: BackfillProgress, max_horses: int | None = None) -> dict:
    """過去分の出走馬の戦績・プロフィール・血統（5 世代）を、まだ無い馬だけ取る。"""
    stats = {"horses_processed": 0, "horses_skipped": 0, "errors": 0}
    done = set(progress._data.get("past_horses_done", []))
    pending = [h for h in progress._data.get("past_horse_ids", []) if h not in done]
    logger.info("Phase horse(past): %d 頭が未処理", len(pending))
    batch: list[str] = []
    for hid in pending:
        if _should_stop() or (max_horses is not None and stats["horses_processed"] >= max_horses):
            break
        try:
            if not runner.storage.exists("horse_result", hid):
                runner.scrape_horse(hid, skip_existing=False, with_history=True)
                stats["horses_processed"] += 1
            else:
                stats["horses_skipped"] += 1
            if not _should_stop() and not runner.storage.exists("horse_pedigree_5gen", hid):
                runner.scrape_horse_pedigree_5gen(hid, skip_existing=True)
            # 完全な 10 世代のための 5 世代目のアンカー。母方（path_fm が M で始まる 16 位置）を先に取り、父方は後回し（anchors_f 段階）
            doc = runner.storage.load("horse_pedigree_5gen", hid) or {}
            mat, pat = [], []
            for a in doc.get("ancestors") or []:
                aid = (a.get("horse_id") or "").strip()
                if a.get("generation") == 5 and aid:
                    (mat if (int(a.get("position", 0)) >> 4) & 1 else pat).append(aid)
            for aid in mat:
                if _should_stop():
                    break
                if not runner.storage.exists("horse_pedigree_5gen", aid):
                    runner.scrape_horse_pedigree_5gen(aid, skip_existing=True)
                    stats["anchors_m"] = stats.get("anchors_m", 0) + 1
            pend = progress._data.setdefault("paternal_anchor_pending", [])
            pend.extend(a for a in pat if a not in pend)
            batch.append(hid)
        except Exception as e:  # noqa: BLE001
            logger.error("馬情報取得エラー [%s]: %s", hid, e)
            stats["errors"] += 1
        if len(batch) >= 25:
            progress._data["past_horses_done"] = sorted(done | set(batch))
            progress.save()
            done |= set(batch)
            batch = []
    if batch:
        progress._data["past_horses_done"] = sorted(done | set(batch))
        progress.save()
    return stats


def _run_phase_anchors_f(runner, progress: BackfillProgress) -> dict:
    """父方の 5 世代目のアンカーの 5 世代表（優先度が低い。母方・他の段階の後に、余力で）。既にあるものは取らない。"""
    stats = {"fetched": 0, "skipped": 0}
    pend = list(progress._data.get("paternal_anchor_pending", []))
    left = []
    for aid in pend:
        if _should_stop():
            left.append(aid)
            continue
        if runner.storage.exists("horse_pedigree_5gen", aid):
            stats["skipped"] += 1
            continue
        runner.scrape_horse_pedigree_5gen(aid, skip_existing=True)
        stats["fetched"] += 1
    progress._data["paternal_anchor_pending"] = left
    progress.save()
    return stats


def past_status(progress: BackfillProgress, years: list[int]) -> dict:
    """過去分の進捗（段階ごとの完了日数と未完了日数）。"""
    out: dict[str, Any] = {}
    for y in years:
        dates = progress._data.get(f"calendar_{y}", [])
        row: dict[str, Any] = {"kaisai_days": len(dates) or None}
        for tier in [t for t in PAST_TIERS if t not in ("horse", "anchors_f")]:
            done = set(progress._data.get(f"done_{tier}", []))
            row[tier] = f"{sum(1 for d in dates if d in done)}/{len(dates)}" if dates else "—"
        out[str(y)] = row
    hs = progress._data.get("past_horse_ids", [])
    out["horses"] = {"collected": len(hs), "done": len(progress._data.get("past_horses_done", []))}
    return out


def run_past(
    years: list[int] | None = None,
    cap: int | None = None,
    only_dates: list[str] | None = None,
    max_horses: int | None = None,
    ignore_window: bool = False,
    ignore_raceday: bool = False,
    tiers: list[str] | None = None,
    dry_run: bool = False,
) -> int:
    """過去分（既定 2019→2016）を、夜ごとの上限・悪化の検知・時間窓の内側で進める。終了コード: 0=正常（上限・時間窓での終了を含む）、2=始めなかった、3=アクセス制限の検知で全停止。"""
    global _GUARD
    from src.scraper import backfill_guard as bg
    from src.scraper.access_guard import AccessGuard, AccessRestrictionDetected
    from src.scraper.scrape_access_pause import read_access_pause

    years = years or PAST_YEARS
    tiers = tiers or PAST_TIERS
    progress = BackfillProgress()
    state = bg.load_state()
    from src.scraper.request_counter import count_today

    budget = bg.daily_budget(count_today())
    cap = min(cap or bg.stage_cap(state), budget) if not cap else cap
    if not cap:
        logger.info("今日の予算がありません（日の総上限 %d − 数えた件数 %d − 通常運用の見込み %d）", bg.TOTAL_DAILY_LIMIT, count_today(), bg.EXPECTED_WEEKDAY_OPS)
        return 2

    if dry_run:
        print(json.dumps({"years": years, "cap_this_night": cap, "cap_max": bg.CAP_MAX, "daily_budget_now": budget, "counted_today": count_today(),
                          "total_daily_limit": bg.TOTAL_DAILY_LIMIT, "window": bg.WINDOW,
                          "status": past_status(progress, years), "nights_recorded": len(state.get("nights", []))},
                         ensure_ascii=False, indent=1))
        return 0

    h = bg.halted()
    if h:
        logger.error("エラーで中断した印があります（%s）。確認して %s を削除するまで始めません", h.get("reason") or h.get("status_counts"), bg.HALT_PATH)
        return 2
    if (read_access_pause() or {}).get("active"):
        logger.error("アクセスの保留が有効です（人が確認して解除するまで始めません）")
        return 2
    now = bg._now()
    if bg.SKIP_RACEDAY and bg.is_raceday(now) and not ignore_raceday:
        logger.info("土日（開催日）は補填しません")
        return 2
    if not ignore_window and not bg.in_window(now):
        logger.info("時間窓 %s（JST）の外です", bg.WINDOW)
        return 2
    if not bg.wait_for_quiet():
        logger.info("他の取得が終わらないため、この夜は始めません")
        return 2

    lock = BackfillLock("past")
    if not lock.acquire():
        return 2
    from src.scraper import client as _client
    from src.scraper.run import ScraperRunner

    guard = bg.NightGuard(cap=cap, window="00:00-23:59" if ignore_window else bg.WINDOW)
    _GUARD = guard
    rc = 0
    ended_by = ""
    stats: dict[str, Any] = {}
    try:
        runner = ScraperRunner(interval=2.0, cache=False, auto_login=True)
        _client.add_response_observer(guard)
        try:
            with AccessGuard(statuses={403, 429}, watch_transport_errors=False):
                all_dates: list[str] = []
                for y in years:
                    if only_dates:
                        all_dates.extend(d for d in only_dates if d.startswith(str(y)))
                    else:
                        all_dates.extend(_past_race_dates(runner, progress, y))
                logger.info("過去分: 年=%s 開催日=%d 上限=%d 件/夜", years, len(all_dates), cap)
                for tier in tiers:
                    if _should_stop():
                        break
                    if tier == "fast":
                        stats["fast"] = _run_phase_fast(runner, progress, all_dates, max_dates=10_000)
                    elif tier == "anchors_f":
                        stats["anchors_f"] = _run_phase_anchors_f(runner, progress)
                    elif tier == "horse":
                        stats["horse_collected"] = _collect_past_horses(runner, progress, all_dates)
                        stats["horse"] = _run_phase_horse_past(runner, progress, max_horses=max_horses)
                    else:
                        stats[tier] = _run_phase_cards_past(runner, progress, all_dates, tier)
        except AccessRestrictionDetected as e:
            ended_by = f"アクセス制限の検知: {e}"
            rc = 3
            logger.error("%s — 全停止（人が確認して解除するまで再開しません）", ended_by)
            try:  # 取得の実行役の保留にも反映する（モジュールがあれば）
                from src.ops import access_hold

                access_hold.set_hold(ended_by, task_id="backfill-past")
            except Exception:  # noqa: BLE001
                pass
        finally:
            _client.remove_response_observer(guard)
        runner.client.close()
    finally:
        ended_by = ended_by or (guard.should_stop() or "対象を処理し終えた")
        rec = guard.record({"ended_by": ended_by, "years": years, "stats": stats})
        _GUARD = None
        lock.release()
    logger.info("=== 過去分補填 終了: %s ===", json.dumps(rec, ensure_ascii=False))
    return rc


def run_backfill(
    year: int | None = None,
    phase: str = "auto",
    max_dates: int = MAX_DATES_PER_RUN,
    dry_run: bool = False,
):
    """バックフィルのメインエントリポイント。"""
    from src.scraper.run import ScraperRunner

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


def past_report() -> None:
    """過去分補填の設定・夜ごとの結果・実効のペース・想定終了を表示する（朝の報告用）。"""
    from src.scraper import backfill_guard as bg
    from src.scraper.request_counter import count_today

    progress = BackfillProgress()
    state = bg.load_state()
    nights = state.get("nights", [])
    print("== 設定 ==")
    print(f"  1日の総上限(全取得) {bg.TOTAL_DAILY_LIMIT} / 補填の上限 最大 {bg.CAP_MAX}（初夜 {bg.CAP_MAX // 2}）/ 通常運用の見込み {bg.EXPECTED_WEEKDAY_OPS}")
    print(f"  時間窓 {bg.WINDOW} JST（月〜金の朝）/ 同時 1 本 / 中断: 403・429・5xx・遅延・通信エラー / 速度調整: AIMD（悪化で待ちを倍、悪化なしで戻す）")
    print(f"  今日（JST）に数えた件数 {count_today()}")
    print("== 夜ごとの結果 ==")
    for n in nights[-14:]:
        print(f"  {n.get('date')} 上限{n.get('cap')} 件数{n.get('requests')} 実効 {n.get('effective_requests_per_hour')}件/時 5xx等{n.get('bad_5xx_403_429')} 通信エラー{n.get('transport_errors')} "
              f"中央値応答{n.get('median_latency_sec')}s 追加待ち{n.get('extra_delay_end_sec')}s 健全={n.get('healthy')} 終了理由={n.get('ended_by')}")
    done = sum(int(n.get("requests") or 0) for n in nights)
    print("== 進捗 ==")
    print(json.dumps(past_status(progress, PAST_YEARS), ensure_ascii=False))
    total_lo, total_hi = 153_000, 170_000
    per_night = (sum(int(n.get("requests") or 0) for n in nights[-10:]) / max(1, len(nights[-10:]))) if nights else float(bg.CAP_MAX)
    per_week = max(1.0, per_night * 5)
    print(f"  累計 {done} 件（目安の総量 {total_lo}〜{total_hi} 件）。直近の1夜あたり {per_night:.0f} 件 → 週 {per_week:.0f} 件")
    print(f"  想定の残り {max(0, total_lo - done) / per_week:.1f}〜{max(0, total_hi - done) / per_week:.1f} 週")


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
                        choices=["auto", "fast", "horse", "full", "past"],
                        help="実行フェーズ (default: auto)")
    parser.add_argument("--max-dates", type=int, default=MAX_DATES_PER_RUN,
                        help=f"1回の実行で処理する最大日数 (default: {MAX_DATES_PER_RUN})")
    parser.add_argument("--dry-run", action="store_true",
                        help="取得せず計画のみ表示")
    parser.add_argument("--status", action="store_true",
                        help="進捗状況を表示して終了")
    parser.add_argument("--report", action="store_true", help="--phase past の設定・夜ごとの結果・実効ペース・想定終了を表示して終了")
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        help="--phase past の対象年（既定: 2019 2018 2017 2016。新しいほうから）")
    parser.add_argument("--cap", type=int, default=None,
                        help="--phase past の 1 夜の取得件数の上限（既定: 段階的。最初の夜は半分）")
    parser.add_argument("--only-dates", nargs="+", default=None, help="--phase past をこの開催日（YYYYMMDD）だけにする（少量の確認用）")
    parser.add_argument("--max-horses", type=int, default=None, help="--phase past で取る馬の数の上限（少量の確認用）")
    parser.add_argument("--tiers", nargs="+", default=None, choices=PAST_TIERS, help="--phase past で進める段階（既定: すべて、上から順）")
    parser.add_argument("--ignore-window", action="store_true", help="手動実行用: 夜間の時間窓を無視する（件数の上限と悪化の検知は有効）")
    parser.add_argument("--ignore-raceday", action="store_true", help="手動実行用: 土日でも実行する")

    args = parser.parse_args()

    os.chdir(Path(__file__).resolve().parents[2])

    if args.status:
        show_status()
        return

    if args.report:
        past_report()
        return

    if args.phase == "past":
        sys.exit(run_past(
            years=args.years, cap=args.cap, only_dates=args.only_dates, max_horses=args.max_horses,
            ignore_window=args.ignore_window, ignore_raceday=args.ignore_raceday, tiers=args.tiers, dry_run=args.dry_run,
        ))

    run_backfill(
        year=args.year,
        phase=args.phase,
        max_dates=args.max_dates,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
