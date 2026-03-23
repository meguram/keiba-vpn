"""
ページ取得不可・タイムアウト等のトランスポート系エラー時にキュー処理を止めるための一時停止フラグ。

状態は data/queue/scrape_access_pause.json に保存（プロセス間で共有）。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PAUSE_FILE = Path(__file__).parent.parent / "data" / "queue" / "scrape_access_pause.json"


def _default_pause() -> dict[str, Any]:
    return {"active": False, "reason": None, "paused_at": None}


def read_access_pause() -> dict[str, Any]:
    if not PAUSE_FILE.exists():
        return _default_pause()
    try:
        with open(PAUSE_FILE, "r", encoding="utf-8") as f:
            d = json.load(f)
        return {
            "active": bool(d.get("active")),
            "reason": d.get("reason"),
            "paused_at": d.get("paused_at"),
        }
    except (OSError, json.JSONDecodeError, TypeError) as e:
        logger.warning("一時停止ファイルの読み込みに失敗、無効扱い: %s", e)
        return _default_pause()


def write_access_pause(*, reason: str) -> None:
    PAUSE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "active": True,
        "reason": (reason or "")[:4000],
        "paused_at": datetime.now().isoformat(),
    }
    tmp = PAUSE_FILE.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(PAUSE_FILE)


def clear_access_pause() -> None:
    try:
        if PAUSE_FILE.exists():
            PAUSE_FILE.unlink()
    except OSError as e:
        logger.warning("一時停止ファイルの削除に失敗: %s", e)


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
                if code in (401, 403, 404, 408, 429):
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

        # レスポンス本文の JSON 破損等はページ到達後の問題 — キュー全体は止めない
        if isinstance(e, requests.exceptions.InvalidJSONError):
            return False

        if isinstance(e, requests.exceptions.RequestException):
            return True

    return False
