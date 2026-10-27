"""ユーザーが訪れやすい画面（直近の開催日の レース一覧・各レースの AI 予測・レース詳細・レース選択の期待値表示）を、
アクセスされる前に作ってディスク(``src.api.prerender``)に置いておく。ユーザーはそれを読むだけで表示できる。

- 有効化は ``KEIBA_PREWARM=1``（既定オフ）。設定は ``config/prewarm.json``、仕様は ``docs/operations/prewarm.md``。
- 動かすのは API(FastAPI) プロセスのうち 1 つだけ（``prewarm.lock`` を取れたワーカー。落ちたら別ワーカーが引き継ぐ）。
- **変更に反応する**: データ保存（``HybridStorage.save``）と、予測・追走難度・ラップ予想・出馬表の状態などのファイル出力があると、
  そのレース/開催日に目印が付く。数秒（debounce）まとめてから、そのレースの画面（予測・詳細）と、その開催日の一覧・期待値表示だけを作り直す。
- **周回（安全網）**: ``cycle_sec`` ごとに窓内の全開催日を見て、無効・期限の半分を過ぎたものだけ作り直す。配備・設定変更・手動クリアで世代が進んだときも同じ。
- 重い処理なので、ホストの負荷が高い周は見送る(``max_load_per_core``)。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from src.api import prerender
from src.utils import cache_epoch

logger = logging.getLogger(__name__)
_JST = timezone(timedelta(hours=9))
_ROOT = Path(__file__).resolve().parents[2]
_DEFAULTS: dict[str, Any] = {
    "past_days": 14, "future_days": 7, "hot_past_days": 3, "hot_max_age_sec": 300, "cold_max_age_sec": 21600,
    "cycle_sec": 60, "race_gap_sec": 0.2, "max_load_per_core": 0.5, "value_views": ["paid", "full"],
    "debounce_sec": 3, "scan_sec": 15, "tick_sec": 2,
}


def load_config() -> dict[str, Any]:
    path = Path(os.environ.get("KEIBA_PREWARM_CONFIG") or _ROOT / "config" / "prewarm.json")
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cfg = {}
    return {**_DEFAULTS, **{k: v for k, v in cfg.items() if not k.startswith("_")}}


def select_dates(stems: list[str], today: datetime, cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    """事前生成する開催日を (hot, cold) で返す。hot=今週のこれから(未来側は窓の全部)と直近 hot_past_days 日、cold=それより前の過去分。どちらも今日に近い順。"""
    t = today.date()
    lo, hi = t - timedelta(days=int(cfg["past_days"])), t + timedelta(days=int(cfg["future_days"]))
    rows: list[tuple[int, int, str]] = []
    for s in {str(x) for x in stems}:
        try:
            d = datetime.strptime(s, "%Y%m%d").date()
        except ValueError:
            continue
        if lo <= d <= hi:
            rows.append((abs((d - t).days), (d - t).days, s))
    rows.sort()
    hot = [s for _, off, s in rows if off >= -int(cfg["hot_past_days"])]
    cold = [s for _, off, s in rows if off < -int(cfg["hot_past_days"])]
    return hot, cold


@dataclass
class Hooks:
    """app.py が渡す、実際の計算。計算の置き場(app.py)と事前生成の流れを分けるため。"""

    stems: Callable[[], list[str]]
    race_list: Callable[[str, float], Awaitable[list[str]]]          # (date, max_age) → race_id の一覧。無効・期限切れなら作り直して置く
    warm_race: Callable[[str, float, bool], Awaitable[int]]            # (race_id, max_age, only_if_stale) → 作り直した件数
    warm_values: Callable[[str, str, float, bool], int]                # (date, view, max_age, only_if_stale) → 作り直した件数
    now: Callable[[], datetime] = lambda: datetime.now(_JST)


@dataclass
class CycleStats:
    started: float = 0.0
    duration: float = 0.0
    dates: int = 0
    races: int = 0
    rebuilt: int = 0
    skipped: str = ""
    errors: int = 0
    events: int = 0


def overloaded(cfg: dict[str, Any]) -> bool:
    try:
        return os.getloadavg()[0] > (os.cpu_count() or 1) * float(cfg["max_load_per_core"])
    except (OSError, AttributeError):
        return False


def stale(kind: str, key: str, max_age: float, only_if_stale: bool) -> bool:
    """作り直すべきか。無効・無し、または期限の半分を過ぎたもの。"""
    if not only_if_stale:
        return True
    a = prerender.age(kind, key)
    return a is None or a > max_age / 2


async def _nosleep(_: float) -> None:
    return None


class Warmer:
    def __init__(self, hooks: Hooks, sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep) -> None:
        self.hooks = hooks
        self.sleep = sleep
        self.race_dates: dict[str, str] = {}      # 窓内の race_id → 開催日（一覧を作るたびに更新）
        self.hot: set[str] = set()
        self.window: set[str] = set()

    def max_age(self, date: str | None, cfg: dict[str, Any]) -> float:
        return float(cfg["cold_max_age_sec"] if (date in self.window and date not in self.hot) else cfg["hot_max_age_sec"])

    async def _list(self, date: str, max_age: float) -> list[str]:
        ids = await self.hooks.race_list(date, max_age)
        for rid in ids:
            self.race_dates[rid] = date
        return ids

    async def _values(self, date: str, cfg: dict[str, Any], max_age: float, st: CycleStats) -> None:
        for view in cfg["value_views"]:
            st.rebuilt += await asyncio.to_thread(self.hooks.warm_values, date, view, max_age, True)

    async def run_cycle(self, cfg: dict[str, Any], *, include_cold: bool = True) -> CycleStats:
        """1 周ぶんの事前生成（安全網）。有効（変更の目印が無く、期限の半分を過ぎていない）ものは作り直さない。"""
        st = CycleStats(started=time.time())
        hot, cold = select_dates(await asyncio.to_thread(self.hooks.stems), self.hooks.now(), cfg)
        self.hot, self.window = set(hot), set(hot) | set(cold)
        plan = [(d, float(cfg["hot_max_age_sec"])) for d in hot]
        if include_cold:
            plan += [(d, float(cfg["cold_max_age_sec"])) for d in cold]
        gap = float(cfg["race_gap_sec"])
        for date, max_age in plan:
            try:
                ids = await self._list(date, max_age)
                st.dates += 1
                await self._values(date, cfg, max_age, st)
                for rid in ids:
                    st.races += 1
                    st.rebuilt += await self.hooks.warm_race(rid, max_age, True)
                    await self.sleep(gap)
            except Exception:  # 1 日の失敗で他の日を止めない
                st.errors += 1
                logger.warning("事前生成に失敗: %s", date, exc_info=True)
        st.duration = round(time.time() - st.started, 2)
        return st

    async def handle_events(self, keys: list[str], cfg: dict[str, Any]) -> CycleStats:
        """変更の目印が付いたレース/開催日だけ作り直す。レースの画面を先に、そのあとで開催日の一覧・期待値表示。"""
        st = CycleStats(started=time.time(), events=len(keys))
        races = [k for k in dict.fromkeys(keys) if len(k) == 12]
        dates = [k for k in dict.fromkeys(keys) if len(k) == 8 and k in self.window]
        affected: dict[str, None] = {d: None for d in dates}
        try:
            for d in dates:                               # 一覧が変わった（レースの増減・取消など）: 先に一覧を作り直して、新しいレースも拾う
                ids = await self._list(d, self.max_age(d, cfg))
                races.extend(r for r in ids if r not in races)
            for rid in races:
                d = self.race_dates.get(rid)
                if d is None:                             # 窓の外のレース（過去分の補填など）は対象外
                    continue
                affected[d] = None
                st.races += 1
                st.rebuilt += await self.hooks.warm_race(rid, self.max_age(d, cfg), True)
            for d in affected:
                ma = self.max_age(d, cfg)
                await self._list(d, ma)
                await self._values(d, cfg, ma, st)
                st.dates += 1
        except Exception:
            st.errors += 1
            logger.warning("変更に伴う事前生成に失敗", exc_info=True)
        st.duration = round(time.time() - st.started, 2)
        return st

    def scan_outputs(self, since_ns: int) -> int:
        """予測・追走難度・ラップ予想・出馬表の状態など、ファイルで出力される側の更新を見つけて目印を付ける。付けた数を返す。"""
        keys: list[str] = []
        days = {d for d in self.hot}
        try:
            from src.api.local_predictions import inference_dir

            keys += _scan_inference(inference_dir(), days, since_ns)
        except Exception:
            pass
        try:
            from src.scraper.entries_sync import state_dir

            keys += _scan_flat(state_dir(), self.race_dates, since_ns)
        except Exception:
            pass
        return prerender.mark_dirty(keys)


_RID = re.compile(r"^(\d{12})(?:\.|$)")


def _day_name(name: str) -> str:
    return name.replace("-", "")


def _scan_day_dir(path: Path, since_ns: int, out: list[str]) -> None:
    try:
        with os.scandir(path) as it:
            for e in it:
                m = _RID.match(e.name)
                if m and e.stat().st_mtime_ns > since_ns:
                    out.append(m.group(1))
    except OSError:
        pass


def _scan_inference(root: Path, days: set[str], since_ns: int) -> list[str]:
    """``<root>/<種類>/<日付>/<race_id>.*`` と ``<root>/predictions_backfill/<モデル>/<日付>/<race_id>.*`` のうち、since 以降に更新されたもの。"""
    out: list[str] = []
    try:
        with os.scandir(root) as top:
            for t in top:
                if not t.is_dir():
                    continue
                try:
                    with os.scandir(t.path) as kids:
                        for k in kids:
                            if not k.is_dir():
                                continue
                            if _day_name(k.name) in days:
                                _scan_day_dir(Path(k.path), since_ns, out)
                            elif t.name == "predictions_backfill":
                                with os.scandir(k.path) as gk:
                                    for g in gk:
                                        if g.is_dir() and _day_name(g.name) in days:
                                            _scan_day_dir(Path(g.path), since_ns, out)
                except OSError:
                    continue
    except OSError:
        pass
    return out


def _scan_flat(path: Path, known: dict[str, str], since_ns: int) -> list[str]:
    """``<dir>/<race_id>.json`` が平らに並ぶ置き場。窓内のレースのファイルだけ stat する。"""
    out: list[str] = []
    try:
        with os.scandir(path) as it:
            for e in it:
                m = _RID.match(e.name)
                if m and m.group(1) in known and e.stat().st_mtime_ns > since_ns:
                    out.append(m.group(1))
    except OSError:
        pass
    return out


_lock_fd: int | None = None


def try_lead() -> bool:
    """事前生成を動かす 1 プロセスになれたら True（プロセスが終わるまで保持）。"""
    global _lock_fd
    if _lock_fd is not None:
        return True
    try:
        import fcntl

        path = Path(os.environ.get("KEIBA_PREWARM_LOCK") or prerender.state_dir() / "prewarm.lock")
        prerender._mkdirs(path.parent)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        _lock_fd = fd
        return True
    except (ImportError, OSError):
        return False


def _write_status(doc: dict[str, Any]) -> None:
    p = Path(os.environ.get("KEIBA_PREWARM_STATUS") or prerender.state_dir() / "prewarm_status.json")
    try:
        prerender._mkdirs(p.parent)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"pid": os.getpid(), "at": datetime.now(_JST).isoformat(timespec="seconds"), **doc}, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)
    except OSError:
        pass


async def run_forever(hooks: Hooks) -> None:
    """周回（安全網）と、変更の目印への反応を 1 つのループで回す。"""
    await asyncio.sleep(5)
    w = Warmer(hooks)
    since_ns = time.time_ns()          # これより前の変更は最初の周回で拾う
    scan_ns = since_ns
    last_cycle = 0.0
    last_scan = 0.0
    last_hard = cache_epoch.hard_current()
    status: dict[str, Any] = {"cycles": 0, "event_batches": 0, "events": 0}
    last_prune = 0.0
    while True:
        try:
            cfg = load_config()
            if not try_lead():
                await asyncio.sleep(30)
                continue
            now = time.time()
            hard = cache_epoch.hard_current()
            if now - last_cycle >= float(cfg["cycle_sec"]) or hard != last_hard:
                last_hard = hard
                if overloaded(cfg):
                    status["last_cycle"] = {"skipped": "load"}
                else:
                    st = await w.run_cycle(cfg)
                    status["cycles"] += 1
                    status["last_cycle"] = {"dates": st.dates, "races": st.races, "rebuilt": st.rebuilt, "errors": st.errors, "duration_sec": st.duration}
                    logger.info("事前生成(周回): %d 日・%d レース・作り直し %d 件・%.1f 秒", st.dates, st.races, st.rebuilt, st.duration)
                last_cycle = time.time()
                _write_status(status)
            if now - last_scan >= float(cfg["scan_sec"]) and w.race_dates:
                cut = time.time_ns()
                n = await asyncio.to_thread(w.scan_outputs, scan_ns)
                scan_ns, last_scan = cut, now
                if n:
                    logger.info("事前生成: ファイル出力の更新 %d 件を検知", n)
            cut = time.time_ns()
            keys = prerender.dirty_between(since_ns, cut)
            since_ns = cut
            if keys and w.race_dates:
                await asyncio.sleep(float(cfg["debounce_sec"]))           # 続けて来る保存をまとめる
                cut = time.time_ns()
                keys += prerender.dirty_between(since_ns, cut)
                since_ns = cut
                st = await w.handle_events(keys, cfg)
                status["event_batches"] += 1
                status["events"] += st.events
                status["last_event_batch"] = {"keys": st.events, "races": st.races, "rebuilt": st.rebuilt, "errors": st.errors, "duration_sec": st.duration}
                logger.info("事前生成(変更): 目印 %d 件 → %d レース・作り直し %d 件・%.1f 秒", st.events, st.races, st.rebuilt, st.duration)
                _write_status(status)
            if now - last_prune > 6 * 3600:
                await asyncio.to_thread(prerender.prune_dirty)
                last_prune = now
            await asyncio.sleep(float(cfg["tick_sec"]))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("事前生成のループで例外", exc_info=True)
            await asyncio.sleep(30)
