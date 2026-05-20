"""
ファイルキュー経由のスクレイピング実行中だけ、ログ行をメモリ＋ファイルに蓄積する。
/queue-status や API からワーカー相当のログをポーリング表示する用途。

マルチプロセス対応:
  uvicorn は --workers N で複数プロセスを持つ。スクレイパースレッドを持つプロセスだけが
  メモリ _buffer を持つため、別プロセスからの API リクエストは空を返す問題が起きる。
  → emit 時に data/queue/.worker_log_ring.jsonl にも追記し、全プロセスから読めるようにした。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

_lock = threading.Lock()
_buffer: deque[dict[str, Any]] = deque(maxlen=2500)
_seq = 0
_handler: logging.Handler | None = None
# スレッドローカルではなくグローバルフラグ — process_queue() 実行中はすべての
# スレッド (rdata_N / phase1_N / phase2_N 含む) のログを補足するため。
_queue_globally_active = threading.Event()

# ファイルリングバッファの設定
_LOG_FILE_MAX_LINES = 2500
_LOG_RING_FILE: Path | None = None
_log_file_lock = threading.Lock()


def _get_log_ring_file() -> Path:
    global _LOG_RING_FILE
    if _LOG_RING_FILE is None:
        _LOG_RING_FILE = Path(__file__).parents[2] / "data" / "queue" / ".worker_log_ring.jsonl"
        _LOG_RING_FILE.parent.mkdir(parents=True, exist_ok=True)
    return _LOG_RING_FILE


def _append_to_file(entry: dict[str, Any]) -> None:
    """ログエントリをファイルに追記する（プロセス間共有用）。"""
    try:
        p = _get_log_ring_file()
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with _log_file_lock:
            with open(p, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception:
        pass


def _trim_log_file_if_needed() -> None:
    """ファイルが最大行数を超えたら古い行を削除（定期的に呼ぶ）。"""
    try:
        p = _get_log_ring_file()
        if not p.exists():
            return
        size = p.stat().st_size
        # 1行最大 ~200B と仮定: 2500行 = 500KB 超えたらトリム
        if size < 500 * 1024:
            return
        with _log_file_lock:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
            if len(lines) > _LOG_FILE_MAX_LINES:
                keep = lines[-_LOG_FILE_MAX_LINES:]
                p.write_text("".join(keep), encoding="utf-8")
    except Exception:
        pass


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

    _trim_counter = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            if len(msg) > 4000:
                msg = msg[:3997] + "..."
            global _seq
            entry: dict[str, Any]
            with _lock:
                _seq += 1
                eid = _seq
                entry = {
                    "id": eid,
                    "ts": time.time(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": msg,
                }
                _buffer.append(entry)

            # ファイルにも書き出す（プロセス間共有）
            _append_to_file(entry)

            # 100件ごとにファイルトリム
            QueueWorkerRingHandler._trim_counter += 1
            if QueueWorkerRingHandler._trim_counter % 100 == 0:
                _trim_log_file_if_needed()
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


def _read_file_entries() -> list[dict[str, Any]]:
    """ファイルリングバッファからエントリを読み込む。"""
    try:
        p = _get_log_ring_file()
        if not p.exists():
            return []
        text = p.read_text(encoding="utf-8", errors="replace")
        entries: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
        return entries[-_LOG_FILE_MAX_LINES:]
    except Exception:
        return []


def get_worker_logs(*, after: int = -1, limit: int = 300) -> dict[str, Any]:
    """
    after < 0: バッファ末尾から limit 件（初回ロード用）。
    after >= 0: id > after のエントリを先頭から最大 limit 件（増分ポーリング）。

    メモリバッファが空の場合はファイルから読み込む（マルチプロセス対応）。
    """
    lim = max(1, min(int(limit), 800))

    with _lock:
        snap = list(_buffer)

    # メモリバッファが空 → ファイルから読む（別プロセスが書いたログ）
    if not snap:
        snap = _read_file_entries()

    if not snap:
        return {"entries": [], "max_id": -1, "total_buffered": 0}

    max_id = snap[-1].get("id", len(snap))
    if after < 0:
        chunk = snap[-lim:]
    else:
        chunk = [e for e in snap if e.get("id", 0) > after][:lim]
    return {"entries": chunk, "max_id": max_id, "total_buffered": len(snap)}


def clear_worker_logs() -> None:
    global _seq
    with _lock:
        _buffer.clear()
        _seq = 0
    try:
        p = _get_log_ring_file()
        if p.exists():
            p.write_text("", encoding="utf-8")
    except Exception:
        pass
