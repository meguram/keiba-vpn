"""
ページ取得不可・タイムアウト等のトランスポート系エラー時にキュー処理を止めるための一時停止フラグ。

状態は data/queue/scrape_access_pause.json に保存（プロセス間で共有）。

HTTP 400（ブロック疑い）が N 回連続した場合はキューを自動全消去し、
UI 向けに queue_auto_cleared を記録する。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PAUSE_FILE = Path(__file__).parents[2] / "data" / "queue" / "scrape_access_pause.json"


def _default_state() -> dict[str, Any]:
    return {
        "active": False,
        "reason": None,
        "paused_at": None,
        "block_400_consecutive": 0,
        "queue_auto_cleared": {"active": False},
    }


def _load_state() -> dict[str, Any]:
    if not PAUSE_FILE.exists():
        return _default_state()
    try:
        with open(PAUSE_FILE, "r", encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict):
            return _default_state()
        out = _default_state()
        out.update(d)
        qac = d.get("queue_auto_cleared")
        if isinstance(qac, dict):
            out["queue_auto_cleared"] = {**{"active": False}, **qac}
        return out
    except (OSError, json.JSONDecodeError, TypeError) as e:
        logger.warning("一時停止ファイルの読み込みに失敗、無効扱い: %s", e)
        return _default_state()


def _save_state(state: dict[str, Any]) -> None:
    PAUSE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = PAUSE_FILE.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    tmp.replace(PAUSE_FILE)


def block_400_clear_threshold() -> int:
    try:
        return max(1, int(os.environ.get("SCRAPE_BLOCK_400_CLEAR_THRESHOLD", "5")))
    except (TypeError, ValueError):
        return 5


def read_access_pause() -> dict[str, Any]:
    state = _load_state()
    qac = state.get("queue_auto_cleared") or {}
    if not isinstance(qac, dict):
        qac = {"active": False}
    return {
        "active": bool(state.get("active")),
        "reason": state.get("reason"),
        "paused_at": state.get("paused_at"),
        "block_400_consecutive": int(state.get("block_400_consecutive") or 0),
        "block_400_clear_threshold": block_400_clear_threshold(),
        "queue_auto_cleared": {
            "active": bool(qac.get("active")),
            "cleared_at": qac.get("cleared_at"),
            "removed_jobs": qac.get("removed_jobs"),
            "threshold": qac.get("threshold"),
            "consecutive_count": qac.get("consecutive_count"),
            "message": qac.get("message"),
        },
    }


def write_access_pause(*, reason: str) -> None:
    state = _load_state()
    state["active"] = True
    state["reason"] = (reason or "")[:4000]
    state["paused_at"] = datetime.now().isoformat()
    _save_state(state)


def clear_access_pause() -> None:
    try:
        if PAUSE_FILE.exists():
            PAUSE_FILE.unlink()
    except OSError as e:
        logger.warning("一時停止ファイルの削除に失敗: %s", e)


def dismiss_queue_auto_cleared_notice() -> None:
    """自動全消去の告知のみ閉じる（一時停止は維持）。"""
    state = _load_state()
    qac = state.get("queue_auto_cleared")
    if isinstance(qac, dict):
        qac["active"] = False
        state["queue_auto_cleared"] = qac
    else:
        state["queue_auto_cleared"] = {"active": False}
    _save_state(state)


def reset_block_400_consecutive() -> None:
    state = _load_state()
    if int(state.get("block_400_consecutive") or 0) == 0:
        return
    state["block_400_consecutive"] = 0
    _save_state(state)


def _exception_chain(exc: BaseException) -> list[BaseException]:
    out: list[BaseException] = []
    seen: set[int] = set()
    e: BaseException | None = exc
    while e is not None and id(e) not in seen and len(out) < 24:
        seen.add(id(e))
        out.append(e)
        nxt = e.__cause__
        if nxt is None:
            nxt = getattr(e, "__context__", None)
        e = nxt
    return out


def is_block_suspect_http_400(exc: BaseException) -> bool:
    """ブロック疑いとして扱う HTTP 400 のみ。"""
    try:
        import requests
    except ImportError:
        return False

    for e in _exception_chain(exc):
        if isinstance(e, requests.exceptions.HTTPError):
            resp = e.response
            if resp is not None and resp.status_code == 400:
                return True
    return False


def is_access_or_transport_error(exc: BaseException) -> bool:
    """
    ページに届かない・サーバエラー・ブロック疑い等でキュー全体を止めたい例外。
    パースエラーやロジックの ValueError は含めない。
    """
    try:
        import requests
    except ImportError:
        return False

    skip_types: tuple[type, ...] = (
        requests.exceptions.InvalidURL,
        requests.exceptions.MissingSchema,
        requests.exceptions.InvalidSchema,
    )

    try:
        import urllib3.exceptions as u3e
    except ImportError:
        u3e = None  # type: ignore[assignment]

    for e in _exception_chain(exc):
        if isinstance(e, skip_types):
            return False

        if u3e is not None and isinstance(
            e,
            (
                u3e.MaxRetryError,
                u3e.NewConnectionError,
                u3e.ConnectTimeoutError,
                u3e.ReadTimeoutError,
                u3e.ProtocolError,
            ),
        ):
            return True

        if isinstance(e, requests.exceptions.HTTPError):
            resp = e.response
            if resp is not None:
                code = resp.status_code
                if code >= 500:
                    return True
                if code in (400, 401, 403, 408, 429):
                    return True
            continue

        if isinstance(
            e,
            (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.TooManyRedirects,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ContentDecodingError,
            ),
        ):
            return True

        if isinstance(e, requests.exceptions.InvalidJSONError):
            return False

        if isinstance(e, requests.exceptions.RequestException):
            return True

    return False


def handle_queue_transport_error(queue: Any, exc: BaseException) -> bool:
    """
    ジョブ失敗時のトランスポート系エラー処理。

    HTTP 400（ブロック疑い）が発生した場合、即座に pending/running/precheck の
    全ジョブを failed に移動してスクレイピングを停止する。
    ユーザーが UI から「再開」を押すことで failed ジョブが pending に戻り、
    スクレイピングが再開される。

    Returns:
        True なら全ジョブを failed に移動した（重大停止）。
    """
    from src.scraper.job_queue import ScrapeJobQueue

    if not isinstance(queue, ScrapeJobQueue):
        queue = ScrapeJobQueue()

    if is_block_suspect_http_400(exc):
        now = datetime.now()
        msg = (
            f"HTTP 400（ブロック疑い）が発生したため、待機中・実行中のジョブをすべて失敗に移動しました。"
            f" netkeiba へのアクセスを控え、しばらくしてから UI の「再開」ボタンでスクレイピングを再開してください。"
            f" 元のエラー: {str(exc)[:500]}"
        )
        failed_count = queue.fail_all_pending_and_running(reason=msg)
        state = _load_state()
        state["active"] = True
        state["reason"] = msg
        state["paused_at"] = now.isoformat()
        state["block_400_timestamps"] = []
        state["block_400_consecutive"] = 0
        state["queue_auto_cleared"] = {
            "active": True,
            "cleared_at": now.isoformat(),
            "removed_jobs": failed_count,
            "threshold": 1,
            "consecutive_count": 1,
            "message": msg,
        }
        _save_state(state)
        logger.error(
            "HTTP 400 ブロック疑い — pending/running %d 件を失敗に移動・スクレイピング停止",
            failed_count,
        )
        return True

    if is_access_or_transport_error(exc):
        queue.pause_queue_for_access_error(str(exc))
        return False

    return False

    return False


def on_queue_job_completed_successfully() -> None:
    """ジョブ成功時に 400 連続カウンタをリセット。"""
    reset_block_400_consecutive()
