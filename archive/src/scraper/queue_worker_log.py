"""
ファイルキュー経由のスクレイピング実行中だけ、ログ行をメモリ＋ファイルに蓄積する。
/queue-status や API からワーカー相当のログをポーリング表示する用途。

各エントリの ``ts`` はログレコードの発生時刻（Formatter の %(asctime)s と同一基準のエポック秒）。
``ts_iso_jst`` は同瞬間を ``Asia/Tokyo`` の ISO8601（ミリ秒）で明示したもの（API/ブラウザのTZ差の切り分け用）。
API 応答に ``display_timezone`` を付与する。

マルチプロセス対応:
  uvicorn は --workers N で複数プロセスを持つ。スクレイパースレッドを持つプロセスだけが
  実ログをメモリ _buffer に持ち、API が当たるワーカーは別プロセスで _buffer が空のことが多い。
  → emit 時に data/queue/.worker_log_ring.jsonl にも追記し、全プロセスから読めるようにした。
  さらに get_worker_logs は **ファイルとメモリを常にマージ**する。API プロセスだけが
  RingHandler で数件メモリに載せた場合に「メモリのみを返してファイルを無視する」と
  本番スクレイプログが一切出ないバグがあったため。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# ログ行の人可読日時（Formatter の asctime）と揃える。netkeiba 運用は JST 前提。
_JST = timezone(timedelta(hours=9))

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
        # HTML アーカイブの定期クリーンアップ等はノイズが多く UI 向けリングには載せない
        if n == "scraper.html_archive":
            return False
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
            created = float(getattr(record, "created", None) or time.time())
            ts_iso_jst = datetime.fromtimestamp(created, tz=_JST).isoformat(
                timespec="milliseconds"
            )
            with _lock:
                _seq += 1
                eid = _seq
                entry = {
                    "id": eid,
                    "ts": created,
                    "ts_iso_jst": ts_iso_jst,
                    "tz": "Asia/Tokyo",
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
    uvicorn 等で propagate が False のとき root に届かないため、主要ロガーは propagate=True に戻す。
    """
    global _handler
    root = logging.getLogger()

    for _lg_name in ("scraper", "src.scraper", "queue", "queue.worker"):
        _lg = logging.getLogger(_lg_name)
        _lg.propagate = True
        if _lg.level == logging.NOTSET or _lg.level > logging.INFO:
            _lg.setLevel(logging.INFO)

    if root.level > logging.INFO:
        root.setLevel(logging.INFO)

    # 既にこのプロセスの root に ring handler が入っていれば追加しない
    if any(isinstance(h, QueueWorkerRingHandler) for h in root.handlers):
        return

    from src.utils.keiba_logging import (
        STANDARD_DATE_FMT,
        STANDARD_LOG_FORMAT,
        JstFormatter,
    )

    h = QueueWorkerRingHandler(level=logging.DEBUG)
    h.setFormatter(
        JstFormatter(STANDARD_LOG_FORMAT, datefmt=STANDARD_DATE_FMT)
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


def _dedupe_key(e: dict[str, Any]) -> tuple[float, str, str]:
    return (
        float(e.get("ts") or 0.0),
        str(e.get("logger") or ""),
        str(e.get("message") or ""),
    )


def _build_merged_log_snapshot() -> list[dict[str, Any]]:
    """
    ファイル（全プロセス共有）＋当プロセスメモリをマージし、時系列で id を 0..n-1 に振り直す。
    ストア済みの id は無視する（プロセスごとに _seq が重複し得るため）。
    """
    file_entries = _read_file_entries()
    with _lock:
        mem_entries = [dict(x) for x in _buffer]

    seen: set[tuple[float, str, str]] = set()
    merged: list[tuple[float, int, dict[str, Any]]] = []
    seq = 0
    for e in file_entries:
        ee = dict(e)
        k = _dedupe_key(ee)
        if k in seen:
            continue
        seen.add(k)
        merged.append((k[0], seq, ee))
        seq += 1
    for e in mem_entries:
        k = _dedupe_key(e)
        if k in seen:
            continue
        seen.add(k)
        merged.append((k[0], seq, e))
        seq += 1

    merged.sort(key=lambda t: (t[0], t[1]))
    out: list[dict[str, Any]] = []
    for i, (_, __, ee) in enumerate(merged):
        row = dict(ee)
        row["id"] = i
        out.append(row)
    return out


def get_worker_logs(*, after: int = -1, limit: int = 300) -> dict[str, Any]:
    """
    after < 0: マージ済みスナップショット末尾から limit 件（初回ロード用）。
    after >= 0: id > after のエントリを時系列順で最大 limit 件（増分ポーリング）。

    ファイルリングとメモリを常にマージする（uvicorn マルチワーカーで API 当たり先が
    スクレイプ実行プロセスと異なる場合でもログが欠けないようにする）。
    """
    lim = max(1, min(int(limit), 800))

    snap = _build_merged_log_snapshot()

    if not snap:
        return {
            "entries": [],
            "max_id": -1,
            "total_buffered": 0,
            "display_timezone": "Asia/Tokyo",
        }

    max_id = snap[-1].get("id", len(snap) - 1)
    if after < 0:
        chunk = snap[-lim:]
    else:
        chunk = [e for e in snap if int(e.get("id", 0)) > int(after)][:lim]
    return {
        "entries": chunk,
        "max_id": max_id,
        "total_buffered": len(snap),
        "display_timezone": "Asia/Tokyo",
    }


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
