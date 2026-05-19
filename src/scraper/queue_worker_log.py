"""
ファイルキュー経由のスクレイピング実行中だけ、ログ行をメモリに蓄積する。
/queue-status や API からワーカー相当のログをポーリング表示する用途。
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any

_lock = threading.Lock()
_buffer: deque[dict[str, Any]] = deque(maxlen=2500)
_seq = 0
_handler: logging.Handler | None = None
# スレッドローカルではなくグローバルフラグ — process_queue() 実行中はすべての
# スレッド (rdata_N / phase1_N / phase2_N 含む) のログを補足するため。
_queue_globally_active = threading.Event()


def mark_queue_worker_active(active: bool) -> None:
    """process_queue 開始/終了時に呼ぶ。"""
    if active:
        _queue_globally_active.set()
    else:
        _queue_globally_active.clear()


def is_queue_worker_active() -> bool:
    return _queue_globally_active.is_set()


class _QueueWorkerContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        n = record.name
        return (
            n.startswith("scraper")
            or n.startswith("queue")
            or n.startswith("src.scraper")
        )


class QueueWorkerRingHandler(logging.Handler):
    """root に1つだけ付け、フィルタでキューワーカー中の scraper.* / queue.* のみ記録。"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            if len(msg) > 4000:
                msg = msg[:3997] + "..."
            global _seq
            with _lock:
                _seq += 1
                eid = _seq
                _buffer.append({
                    "id": eid,
                    "ts": time.time(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": msg,
                })
        except Exception:
            pass


def ensure_queue_worker_log_handler() -> None:
    """冪等。root にハンドラを1つ追加。"""
    global _handler
    root = logging.getLogger()
    if getattr(root, "_queue_worker_ring_handler_installed", False):
        return
    h = QueueWorkerRingHandler(level=logging.DEBUG)
    from src.utils.keiba_logging import (
        STANDARD_DATE_FMT,
        STANDARD_LOG_FORMAT,
    )

    h.setFormatter(
        logging.Formatter(STANDARD_LOG_FORMAT, datefmt=STANDARD_DATE_FMT)
    )
    h.addFilter(_QueueWorkerContextFilter())
    root.addHandler(h)
    root._queue_worker_ring_handler_installed = True  # type: ignore[attr-defined]
    _handler = h

    # scraper.* / src.scraper.* はデフォルトで root レベルを継承するが、
    # root が WARNING のままだと INFO ログが作られず ring handler に届かない。
    # queue.worker は明示的に setLevel(INFO) しているので届くが、
    # scraper.run 等は NOTSET のため root の WARNING を継承してしまう。
    for _ns in ("scraper", "src.scraper"):
        _lg = logging.getLogger(_ns)
        if _lg.level == logging.NOTSET or _lg.level > logging.INFO:
            _lg.setLevel(logging.INFO)


def get_worker_logs(*, after: int = -1, limit: int = 300) -> dict[str, Any]:
    """
    after < 0: バッファ末尾から limit 件（初回ロード用）。
    after >= 0: id > after のエントリを先頭から最大 limit 件（増分ポーリング）。
    """
    lim = max(1, min(int(limit), 800))
    with _lock:
        snap = list(_buffer)
    if not snap:
        return {"entries": [], "max_id": -1, "total_buffered": 0}
    max_id = snap[-1]["id"]
    if after < 0:
        chunk = snap[-lim:]
    else:
        chunk = [e for e in snap if e["id"] > after][:lim]
    return {"entries": chunk, "max_id": max_id, "total_buffered": len(snap)}


def clear_worker_logs() -> None:
    global _seq
    with _lock:
        _buffer.clear()
        _seq = 0
