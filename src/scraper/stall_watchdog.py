"""取得プロセスの固まり検知。HTTP 1 回が上限秒を超えて返らない／ログが長時間まったく出ない場合に、全スレッドのスタックを残して強制終了する。

requests の timeout は 1 回の通信操作ごとなので、名前解決や少しずつ届く応答では全体が縛られない（2026-10-29 に 8 時間余り固まった）。
SIGTERM は「現在のレース完了後に停止」のフラグを立てるだけで、固まった呼び出しは止められないため、ここでは os._exit する。
終了コードは EXIT_STALLED。cron の起動スクリプトが短い間隔で再起動する。
"""

from __future__ import annotations

import faulthandler
import logging
import os
import signal
import sys
import threading
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)

EXIT_STALLED = 75

_lock = threading.Lock()
_inflight: dict[int, tuple[float, str]] = {}
_last_log = time.monotonic()
_started = False


def _install_log_clock() -> None:
    # ハンドラではなくレコード生成に掛ける（propagate=False のロガーや basicConfig の有無に影響されない）
    orig = logging.getLogRecordFactory()

    def factory(*a, **kw):
        global _last_log
        _last_log = time.monotonic()
        return orig(*a, **kw)

    logging.setLogRecordFactory(factory)


@contextmanager
def track_http(url: str):
    """HTTP 呼び出しを囲む。監視が始まっていなくても害はない。"""
    tid = threading.get_ident()
    with _lock:
        _inflight[tid] = (time.monotonic(), url)
    try:
        yield
    finally:
        with _lock:
            _inflight.pop(tid, None)


def _stalled(http_sec: float, idle_sec: float, now: float) -> str | None:
    with _lock:
        items = list(_inflight.values())
    for t0, url in items:
        if now - t0 > http_sec:
            return f"HTTP が {now - t0:.0f} 秒返らない: {url}"
    if now - _last_log > idle_sec:
        return f"ログが {now - _last_log:.0f} 秒出ていない"
    return None


def _die(reason: str) -> None:
    msg = f"固まりを検知（{reason}）— スタックを出力して強制終了します (exit={EXIT_STALLED})"
    try:
        logger.critical(msg)
        sys.stderr.write(msg + "\n")
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        sys.stderr.flush()
    finally:
        os._exit(EXIT_STALLED)


def start(http_sec: float = 240.0, idle_sec: float = 2400.0, poll_sec: float = 5.0) -> None:
    """監視スレッドを 1 回だけ起動する。idle_sec は他の取得待ち（最大 1800 秒、ログなし）より長くする。"""
    global _started, _last_log
    if _started:
        return
    _started = True
    _last_log = time.monotonic()
    _install_log_clock()
    try:
        faulthandler.register(signal.SIGUSR1, all_threads=True)   # kill -USR1 <pid> でいつでもスタックを出せる
    except (AttributeError, ValueError, RuntimeError):
        pass

    def loop() -> None:
        while True:
            time.sleep(poll_sec)
            reason = _stalled(http_sec, idle_sec, time.monotonic())
            if reason:
                _die(reason)

    threading.Thread(target=loop, name="stall-watchdog", daemon=True).start()
    logger.info("固まり検知を開始: HTTP %.0f 秒 / ログなし %.0f 秒で強制終了", http_sec, idle_sec)
