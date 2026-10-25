"""データが変わったときに、プロセス内のキャッシュをまとめて無効化する仕組み（複数プロセス間は「世代ファイル」で伝える）。

- ``bump(reason)``: 世代ファイルを更新する（スクレイパ・配備・運用 CLI が呼ぶ）。
- ``sync()``: API が各要求の前に呼ぶ。世代ファイルが変わっていれば、登録済みの無効化関数をすべて実行する。
- ``register(name, fn)``: 各モジュールが、自分のキャッシュを空にする関数を登録する。
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

_FILE = Path(os.environ.get("KEIBA_CACHE_EPOCH_FILE") or Path(__file__).resolve().parents[2] / "data" / "local" / "meta" / "cache_epoch.json")
_clearers: dict[str, Callable[[], object]] = {}
_lock = threading.Lock()
_seen: dict[str, float] = {"mtime": -1.0, "checked": 0.0, "bumped": 0.0}
_CHECK_EVERY_SEC = 1.0
_BUMP_MIN_INTERVAL_SEC = 20.0

# 画面に出る内容が変わるカテゴリ（保存のたびに世代を進める対象）
UI_CATEGORIES = frozenset({
    "race_lists", "race_shutuba", "race_odds", "race_pair_odds", "race_trifecta_odds", "race_result",
    "race_result_on_time", "race_horse_weight", "horse_race_history", "horse_profile",
})


def register(name: str, fn: Callable[[], object]) -> None:
    with _lock:
        _clearers[name] = fn


def clear_all(reason: str = "") -> list[str]:
    """登録済みの無効化関数をすべて実行する。失敗したものの名前を返す。"""
    with _lock:
        items = list(_clearers.items())
    failed: list[str] = []
    for name, fn in items:
        try:
            fn()
        except Exception as exc:  # 1 つの失敗で他の無効化を止めない
            failed.append(name)
            logger.warning("キャッシュ無効化に失敗 %s: %s", name, exc)
    if reason:
        logger.info("キャッシュを無効化しました（%s, %d 件）", reason, len(items))
    return failed


def bump(reason: str = "") -> None:
    """世代ファイルを更新する。次に ``sync()`` を呼んだ全プロセスがキャッシュを空にする。"""
    try:
        _FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps({"at": time.time(), "reason": reason}, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, _FILE)
    except OSError as exc:
        logger.warning("世代ファイルを書けません: %s", exc)


def bump_for_save(category: str) -> None:
    """保存のたびに呼ぶ。画面に関わるカテゴリだけ、プロセスあたり 20 秒に 1 回まで世代を進める。"""
    if category not in UI_CATEGORIES:
        return
    now = time.monotonic()
    if now - _seen["bumped"] < _BUMP_MIN_INTERVAL_SEC:
        _schedule_trailing_bump(category, _BUMP_MIN_INTERVAL_SEC - (now - _seen["bumped"]))  # 間引いた保存も、間隔が空いたあとに 1 回反映する
        return
    _seen["bumped"] = now
    bump(f"save:{category}")


_pending: dict[str, threading.Timer | None] = {"timer": None}


def _fire_trailing(category: str) -> None:
    with _lock:
        _pending["timer"] = None
    _seen["bumped"] = time.monotonic()
    bump(f"save:{category}:trailing")


def _schedule_trailing_bump(category: str, delay: float) -> None:
    with _lock:
        if _pending["timer"] is not None:
            return
        t = threading.Timer(max(delay, 0.0) + 0.5, _fire_trailing, args=(category,))
        t.daemon = True
        _pending["timer"] = t
        t.start()


def flush_pending() -> None:
    """間引かれた保存の世代更新が残っていれば、今すぐ 1 回行う（プロセス終了時に呼ばれる）。"""
    with _lock:
        t, _pending["timer"] = _pending["timer"], None
    if t is not None:
        t.cancel()
        bump("save:flush")


atexit.register(flush_pending)


def sync(force: bool = False) -> bool:
    """世代ファイルが変わっていれば無効化を実行して True。1 秒に 1 回しか stat しない。"""
    now = time.monotonic()
    if not force and now - _seen["checked"] < _CHECK_EVERY_SEC:
        return False
    _seen["checked"] = now
    try:
        m = _FILE.stat().st_mtime
    except OSError:
        return False
    if _seen["mtime"] < 0:      # 初回は基準を覚えるだけ（起動直後のキャッシュは空）
        _seen["mtime"] = m
        return False
    if m == _seen["mtime"]:
        return False
    _seen["mtime"] = m
    clear_all("cache_epoch")
    return True
