"""共有ヘッダ認証（X-Keiba-Proxy-Key）。Cloud Run の API を VPS の nginx 経由でだけ使えるようにする。

- 既定オフ（``KEIBA_PROXY_KEY_REQUIRED`` が 1 でない間は何もしない）。ホスト運用には影響しない。
- 鍵は環境変数（Cloud Run では Secret Manager からのマウント）。``KEIBA_PROXY_KEY`` と ``KEIBA_PROXY_KEY_NEXT`` のどちらでも通すので、
  新しい鍵を NEXT に入れて VPS を切り替え、最後に古い鍵を外す、というローテーションができる。
- 比較は定数時間（hmac.compare_digest）。鍵の値はログ・応答に出さない。
- 鍵が通った要求に限り、``X-Keiba-Client-IP``（VPS の nginx が付ける本当のクライアント IP）を接続元として採用する
  （Cloud Run では接続元が Google のフロントになるため。鍵を持つ相手しかこのヘッダを使えない）。
- ヘッダ無し・不一致は 401。必須なのに鍵が 1 つも設定されていないときは全拒否（開けっぱなしにしない）。
- ヘルスチェック（Cloud Run の起動確認が鍵を持たない）などは ``exempt_paths`` で除外する。
"""

from __future__ import annotations

import hmac
import ipaddress
import logging
import os
from typing import Any, Awaitable, Callable, Iterable

logger = logging.getLogger("api.proxy_key")

HEADER = "x-keiba-proxy-key"
DEFAULT_EXEMPT = ("/api/health", "/api/v1/health")
CLIENT_IP_HEADER = b"x-keiba-client-ip"

_DENY_BODY = b'{"error":"unauthorized","reason":"proxy_key"}'


def is_required(env: dict[str, str] | None = None) -> bool:
    e = os.environ if env is None else env
    return (e.get("KEIBA_PROXY_KEY_REQUIRED") or "").strip().lower() in ("1", "true", "yes", "on")


def configured_keys(env: dict[str, str] | None = None) -> list[bytes]:
    e = os.environ if env is None else env
    keys = []
    for name in ("KEIBA_PROXY_KEY", "KEIBA_PROXY_KEY_NEXT"):
        v = (e.get(name) or "").strip()
        if v:
            keys.append(v.encode("utf-8"))
    return keys


def key_matches(presented: str | bytes | None, keys: Iterable[bytes]) -> bool:
    """どの鍵とも比較を省略しない（先に一致した時点で打ち切らない）定数時間比較。"""
    if not presented:
        return False
    p = presented.encode("utf-8") if isinstance(presented, str) else presented
    ok = False
    for k in keys:
        ok |= hmac.compare_digest(p, k)
    return ok


def is_exempt(path: str, exempt_paths: Iterable[str]) -> bool:
    return any(path == p or path.rstrip("/") == p for p in exempt_paths)


def _client_ip(scope: dict[str, Any]) -> str | None:
    for k, v in scope.get("headers") or []:
        if k.lower() == CLIENT_IP_HEADER:
            try:
                return str(ipaddress.ip_address(v.decode("latin-1").split(",")[0].strip()))
            except ValueError:
                return None
    return None


class ProxyKeyMiddleware:
    """純 ASGI ミドルウェア。HTTP / WebSocket の最初に鍵を確認し、通らなければ下流を呼ばず 401 を返す。"""

    def __init__(self, app: Callable[..., Awaitable[None]], *, exempt_paths: Iterable[str] = DEFAULT_EXEMPT,
                 env: dict[str, str] | None = None) -> None:
        self.app = app
        self.exempt_paths = tuple(exempt_paths)
        self._env = env

    async def __call__(self, scope: dict[str, Any], receive: Callable, send: Callable) -> None:
        if scope.get("type") not in ("http", "websocket") or not is_required(self._env):
            await self.app(scope, receive, send)
            return
        path = scope.get("path") or ""
        if is_exempt(path, self.exempt_paths):
            await self.app(scope, receive, send)
            return
        keys = configured_keys(self._env)
        presented = None
        for k, v in scope.get("headers") or []:
            if k.lower() == HEADER.encode():
                presented = v.decode("latin-1")
                break
        if keys and key_matches(presented, keys):
            ip = _client_ip(scope)
            if ip:
                scope = {**scope, "client": (ip, 0)}
            await self.app(scope, receive, send)
            return
        if not keys:
            logger.error("KEIBA_PROXY_KEY_REQUIRED=1 なのに鍵が未設定です。全て拒否します（Secret Manager のマウントを確認）")
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        await send({"type": "http.response.start", "status": 401, "headers": [
            (b"content-type", b"application/json"), (b"cache-control", b"no-store"),
            (b"content-length", str(len(_DENY_BODY)).encode()),
        ]})
        await send({"type": "http.response.body", "body": _DENY_BODY})
