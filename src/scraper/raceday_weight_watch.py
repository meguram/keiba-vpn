"""開催日の馬体重検知ランナー用の純粋ロジック（時刻計画・検知判定）。

馬体重は基本的に発走の約 1 時間前に発表される。固定時刻（T-60）に取りに行くと発表の直前にあたりうるため、
各レースの T-65〜T-30 の窓で出馬表を 1〜2 分間隔で取り直し、9 割以上の馬に馬体重が入った時点で予測を起動する。
期限（T-30）までに入らなければ、馬体重なし（前走の馬体重で代替）として起動する。

ランナー本体は ``auto_scrape_queue.task_raceday_weight_runner``（``--task raceday-weight-runner``）。stg ホストの cron（23:00 UTC = 08:00 JST）で起動する。
環境変数:
  KEIBA_WEIGHT_WATCH_START_MIN     窓の開始（発走の何分前から。既定 65）
  KEIBA_WEIGHT_WATCH_DEADLINE_MIN  期限（発走の何分前まで。既定 30）
  KEIBA_WEIGHT_WATCH_POLL_SEC      取得の間隔（秒。既定 90）
  KEIBA_WEIGHT_WATCH_MIN_RATIO     馬体重が入った馬の割合の閾値（既定 0.9）
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

WEIGHT_WATCH_TASKS: list[str] = ["race_shutuba"]


def _env_num(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def start_min() -> float:
    return _env_num("KEIBA_WEIGHT_WATCH_START_MIN", 65.0)


def deadline_min() -> float:
    return _env_num("KEIBA_WEIGHT_WATCH_DEADLINE_MIN", 30.0)


def poll_sec() -> float:
    return max(30.0, _env_num("KEIBA_WEIGHT_WATCH_POLL_SEC", 90.0))


def min_ratio() -> float:
    return _env_num("KEIBA_WEIGHT_WATCH_MIN_RATIO", 0.9)


@dataclass(frozen=True)
class Window:
    race_id: str
    start: datetime
    deadline: datetime
    race: dict[str, Any]


def plan_windows(schedule: list[dict], start_min_: float | None = None, deadline_min_: float | None = None) -> list[Window]:
    """各レースの取得窓 [発走-start, 発走-deadline] を、開始時刻順に返す（純粋関数）。"""
    s = start_min() if start_min_ is None else start_min_
    d = deadline_min() if deadline_min_ is None else deadline_min_
    out = [Window(r["race_id"], r["post_time"] - timedelta(minutes=s), r["post_time"] - timedelta(minutes=d), r) for r in schedule]
    out.sort(key=lambda w: (w.start, w.race_id))
    return out


def open_windows(windows: list[Window], done: set[str], now: datetime) -> list[Window]:
    """now に取得窓が開いている（未処理の）レース。期限を過ぎたものも、起動のために返す。"""
    return [w for w in windows if w.race_id not in done and w.start <= now]


def next_wake(windows: list[Window], done: set[str], now: datetime, poll: float) -> datetime | None:
    """次に動く時刻。窓が開いているレースがあれば now + poll、なければ次の窓の開始。全部終われば None。"""
    pending = [w for w in windows if w.race_id not in done]
    if not pending:
        return None
    if any(w.start <= now for w in pending):
        return now + timedelta(seconds=poll)
    return min(w.start for w in pending)


def weights_ratio(card: dict | None) -> float:
    """出馬表の出走馬のうち、馬体重（正の数）が入っている馬の割合。"""
    entries = [e for e in ((card or {}).get("entries") or []) if isinstance(e, dict)]
    if not entries:
        return 0.0
    n = 0
    for e in entries:
        try:
            if float(e.get("weight") or 0) > 0:
                n += 1
        except (TypeError, ValueError):
            pass
    return n / len(entries)


def weights_announced(card: dict | None, ratio: float | None = None) -> bool:
    return weights_ratio(card) >= (min_ratio() if ratio is None else ratio)


def decide(card: dict | None, window: Window, now: datetime) -> str:
    """'announced'（馬体重が揃った）/ 'deadline'（期限切れ。馬体重なしで起動）/ 'wait'（次の取得まで待つ）。"""
    if weights_announced(card):
        return "announced"
    if now >= window.deadline:
        return "deadline"
    return "wait"
