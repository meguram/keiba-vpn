"""スクレイピング対策のミドルウェア（FastAPI / Flask）。判定は src/utils/anti_scraping.py。

適用順（外側 → 内側）: KeibaDeployment（ヘッダ付与）→ メンテナンス判定 → AntiScraping（ブロック判定・レート制限）→ Auth → GZip → ルート。
内部ジョブ（転送ヘッダの無いループバックの直接アクセス）と許可リストは常に素通し。mode=detect のときは記録だけで止めない。
"""

from __future__ import annotations

import logging
from typing import Any

from src.utils.anti_scraping import AntiScraping, Decision, get_engine

logger = logging.getLogger(__name__)

ROBOTS_TAG = "noindex, nofollow, noarchive, noai, noimageai"
_MESSAGES = {
    401: "認証が必要です。ログインしてください。",
    403: "アクセスが制限されています。",
    429: "リクエストが多すぎます。しばらくしてからお試しください。",
}
_REASON_PUBLIC = {401: "login_required", 403: "blocked", 429: "rate_limited"}


def deny_body(dec: Decision) -> dict[str, Any]:
    """拒否時の本文。内部の理由（しきい値・スコア・検知した信号）は返さない。"""
    body: dict[str, Any] = {"error": _MESSAGES.get(dec.status, "アクセスできません"), "reason": _REASON_PUBLIC.get(dec.status, "blocked")}
    if dec.status == 401:
        body["login_url"] = "/login"
    if dec.retry_after:
        body["retry_after"] = int(dec.retry_after)
    return body


def _headers_for(dec: Decision) -> dict[str, str]:
    h = {"X-Robots-Tag": ROBOTS_TAG, "Cache-Control": "no-store"}
    if dec.retry_after:
        h["Retry-After"] = str(int(dec.retry_after))
    return h


# ───────────────────────── FastAPI ─────────────────────────

def register_fastapi(app: Any, engine: AntiScraping | None = None) -> None:
    """app.add_middleware を 1 回呼ぶ（AuthMiddleware を足した直後に呼び、Auth の外側に置く）。"""
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse, PlainTextResponse

    eng = engine or get_engine()

    class AntiScrapingMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            try:
                path = request.url.path
                peer = request.client.host if request.client else None
                hdrs = request.headers
                ip = eng.client_ip(peer, hdrs)
                if request.method == "POST" and eng.is_login_path(path):
                    ok, wait = eng.login_check(ip)
                    if not ok:
                        dec = Decision("block", 429, "login_locked", wait)
                        return JSONResponse(deny_body(dec), status_code=429, headers=_headers_for(dec))
                authed = False
                token = ""
                if eng.path_kind(path) == "data":
                    try:
                        from src.api.auth import COOKIE_NAME, is_developer

                        authed = bool(is_developer(request))
                        token = request.cookies.get(COOKIE_NAME, "")
                    except Exception:  # noqa: BLE001
                        authed = False
                dec = eng.check(peer=peer, headers=hdrs, method=request.method, path=path, authed=authed, session_token=token)
            except Exception as exc:  # noqa: BLE001 — 対策の不具合でサイトを止めない
                logger.warning("anti_scraping 判定に失敗（素通し）: %s", exc)
                return await call_next(request)
            if dec.blocked:
                if path.startswith("/api/"):
                    return JSONResponse(deny_body(dec), status_code=dec.status, headers=_headers_for(dec))
                return PlainTextResponse(_MESSAGES.get(dec.status, "アクセスできません"), status_code=dec.status, headers=_headers_for(dec))
            response = await call_next(request)
            try:
                response.headers.setdefault("X-Robots-Tag", ROBOTS_TAG)
                if request.method == "POST" and eng.is_login_path(path):
                    eng.login_record(ip, response.status_code in (200, 302, 303) and bool(response.headers.get("set-cookie")))
            except Exception:  # noqa: BLE001
                pass
            return response

    app.add_middleware(AntiScrapingMiddleware)


# ───────────────────────── Flask ─────────────────────────

def register_flask(app: Any, engine: AntiScraping | None = None) -> None:
    """Flask の before_request / after_request に接続する（create_app の中で 1 回呼ぶ）。"""
    from flask import g, jsonify, request

    eng = engine or get_engine()

    @app.before_request
    def _anti_scraping_before():  # type: ignore[no-untyped-def]
        try:
            path = request.path
            peer = request.remote_addr
            ip = eng.client_ip(peer, request.headers)
            g._as_ip = ip
            if request.method == "POST" and eng.is_login_path(path):
                ok, wait = eng.login_check(ip)
                if not ok:
                    dec = Decision("block", 429, "login_locked", wait)
                    r = jsonify(deny_body(dec))
                    r.status_code = 429
                    r.headers.update(_headers_for(dec))
                    return r
            authed = False
            token = ""
            if eng.path_kind(path) == "data":
                try:
                    from src.api.auth import COOKIE_NAME, dev_auto_login_applies, _verify_token

                    token = request.cookies.get(COOKIE_NAME, "")
                    authed = bool(dev_auto_login_applies(peer, request.headers) or _verify_token(token))
                except Exception:  # noqa: BLE001
                    authed = False
            dec = eng.check(peer=peer, headers=request.headers, method=request.method, path=path, authed=authed, session_token=token)
            if dec.blocked:
                r = jsonify(deny_body(dec))
                r.status_code = dec.status
                r.headers.update(_headers_for(dec))
                return r
        except Exception as exc:  # noqa: BLE001
            logger.warning("anti_scraping 判定に失敗（素通し）: %s", exc)
        return None

    @app.after_request
    def _anti_scraping_after(resp):  # type: ignore[no-untyped-def]
        try:
            resp.headers.setdefault("X-Robots-Tag", ROBOTS_TAG)
            if request.method == "POST" and eng.is_login_path(request.path):
                ip = getattr(g, "_as_ip", None) or eng.client_ip(request.remote_addr, request.headers)
                eng.login_record(ip, resp.status_code == 200)
        except Exception:  # noqa: BLE001
            pass
        return resp
