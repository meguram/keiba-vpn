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
    """冪等。root にハンドラを1つ追加。

    属性フラグではなく型チェックで判定する（fork/reload 後も正しく動作させるため）。
    ハンドラが既存でも scraper.*/src.scraper.* のレベルは毎回 INFO に保証する。
    """
    global _handler
    root = logging.getLogger()

    # scraper.* / src.scraper.* が WARNING 以上の場合 INFO ログが作られない。
    # root が WARNING のままでも個別 logger を INFO にすることで record が生成される。
    for _ns in ("scraper", "src.scraper"):
        _lg = logging.getLogger(_ns)
        if _lg.level == logging.NOTSET or _lg.level > logging.INFO:
            _lg.setLevel(logging.INFO)

    # 既にこのプロセスの root に ring handler が入っていれば追加しない
    if any(isinstance(h, QueueWorkerRingHandler) for h in root.handlers):
        return

    from src.utils.keiba_logging import (
        STANDARD_DATE_FMT,
        STANDARD_LOG_FORMAT,
    )

    h = QueueWorkerRingHandler(level=logging.DEBUG)
    h.setFormatter(
        logging.Formatter(STANDARD_LOG_FORMAT, datefmt=STANDARD_DATE_FMT)
    )
    h.addFilter(_QueueWorkerContextFilter())
    root.addHandler(h)
    root._queue_worker_ring_handler_installed = True  # type: ignore[attr-defined]
    _handler = h


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
