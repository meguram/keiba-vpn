"""API の GET 応答にブラウザ向けの短いキャッシュ指示（Cache-Control: private, max-age=N と Vary: Cookie）を付ける。

許可リストは config/http_cache.json。共有キャッシュには使えない private なので、会員と匿名の応答が混ざらない。
Cookie が変わる（ログイン・ログアウト）と別の応答として取り直される。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

_CFG = Path(os.environ.get("KEIBA_HTTP_CACHE_CONFIG") or Path(__file__).resolve().parents[2] / "config" / "http_cache.json")
_cache: dict[str, Any] = {"at": 0.0, "cfg": None}


def config() -> dict[str, Any]:
    now = time.monotonic()
    if _cache["cfg"] is not None and now - _cache["at"] < 30:
        return _cache["cfg"]
    try:
        cfg = json.loads(_CFG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cfg = {"enabled": False}
    _cache.update(at=now, cfg=cfg)
    return cfg


def max_age_for(path: str, method: str, status: int, has_cache_control: bool, cfg: dict[str, Any] | None = None) -> int | None:
    """付けるべき max-age(秒)。付けない場合は None。"""
    cfg = cfg if cfg is not None else config()
    if not cfg.get("enabled") or method.upper() != "GET" or status != 200 or has_cache_control:
        return None
    if any(path == n or path.startswith(n.rstrip("/") + "/") or path.startswith(n) for n in cfg.get("never") or []):
        return None
    for r in cfg.get("rules") or []:
        if path.startswith(r["prefix"]):
            ma = int(r.get("max_age") or 0)
            return ma if ma > 0 else None
    return None


def _vary_add(existing: str) -> str:
    parts = [p.strip() for p in existing.split(",") if p.strip()]
    if not any(p.lower() == "cookie" for p in parts):
        parts.append("Cookie")
    return ", ".join(parts)


def install_fastapi(app) -> None:
    from starlette.middleware.base import BaseHTTPMiddleware

    class _HttpCacheMW(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            resp = await call_next(request)
            ma = max_age_for(request.url.path, request.method, resp.status_code, "cache-control" in resp.headers)
            if ma:
                resp.headers["Cache-Control"] = f"private, max-age={ma}"
                resp.headers["Vary"] = _vary_add(resp.headers.get("Vary", ""))
            return resp

    app.add_middleware(_HttpCacheMW)


def install_flask(app) -> None:
    from flask import request

    @app.after_request
    def _http_cache(response):
        ma = max_age_for(request.path, request.method, response.status_code, "Cache-Control" in response.headers)
        if ma:
            response.headers["Cache-Control"] = f"private, max-age={ma}"
            response.headers["Vary"] = _vary_add(response.headers.get("Vary", ""))
        return response
