"""API の GET 応答にブラウザ向けの短いキャッシュ指示（Cache-Control: private, max-age=N と Vary: Cookie）を付ける。

許可リストは config/http_cache.json。既定は private なので、会員と匿名の応答が混ざらない。
Cookie が変わる（ログイン・ログアウト）と別の応答として取り直される。
例外: ルールに ``s_maxage`` があり、リクエストに Cookie / Authorization が無い（会員区分で応答が変わらない匿名）ときだけ
``public, max-age=N, s-maxage=S`` にして CDN に置かせる（閲覧者が増えてもオリジンに届くのは S 秒に 1 回）。
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


def shared_s_maxage_for(path: str, cfg: dict[str, Any] | None = None) -> int | None:
    """CDN など共有キャッシュに置いてよい秒数（``s_maxage``）。ルールに無ければ None。"""
    cfg = cfg if cfg is not None else config()
    for r in cfg.get("rules") or []:
        if path.startswith(r["prefix"]):
            s = int(r.get("s_maxage") or 0)
            return s if s > 0 else None
    return None


def is_anonymous_request(cookie_header: str, authorization: str, remote_addr: str | None, headers) -> bool:
    """会員区分で応答が変わらない（誰が見ても同じ）匿名リクエストか。Cookie・Authorization が一切無く、dev の自動ログインでもないときだけ True。"""
    if (cookie_header or "").strip() or (authorization or "").strip():
        return False
    from src.api.auth import dev_auto_login_applies

    return not dev_auto_login_applies(remote_addr, headers)


def cache_control_value(ma: int, s_maxage: int | None, anonymous: bool) -> str:
    """匿名で ``s_maxage`` のあるパスだけ共有キャッシュ可（public）。それ以外は従来どおり private（会員の応答が他人に渡らない）。"""
    if anonymous and s_maxage:
        return f"public, max-age={ma}, s-maxage={s_maxage}"
    return f"private, max-age={ma}"


def _vary_add(existing: str) -> str:
    parts = [p.strip() for p in existing.split(",") if p.strip()]
    if not any(p.lower() == "cookie" for p in parts):
        parts.append("Cookie")
    return ", ".join(parts)


def install_fastapi(app) -> None:
    from starlette.middleware.base import BaseHTTPMiddleware

    class _HttpCacheMW(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            from src.utils import cache_epoch

            cache_epoch.sync()
            resp = await call_next(request)
            ma = max_age_for(request.url.path, request.method, resp.status_code, "cache-control" in resp.headers)
            if ma:
                anon = is_anonymous_request(
                    request.headers.get("cookie", ""), request.headers.get("authorization", ""),
                    getattr(request.client, "host", None), request.headers,
                )
                resp.headers["Cache-Control"] = cache_control_value(ma, shared_s_maxage_for(request.url.path), anon)
                resp.headers["Vary"] = _vary_add(resp.headers.get("Vary", ""))
            return resp

    app.add_middleware(_HttpCacheMW)


def install_flask(app) -> None:
    from flask import request

    @app.before_request
    def _cache_epoch_sync():
        from src.utils import cache_epoch

        cache_epoch.sync()

    @app.after_request
    def _http_cache(response):
        ma = max_age_for(request.path, request.method, response.status_code, "Cache-Control" in response.headers)
        if ma:
            anon = is_anonymous_request(
                request.headers.get("Cookie", ""), request.headers.get("Authorization", ""), request.remote_addr, request.headers,
            )
            response.headers["Cache-Control"] = cache_control_value(ma, shared_s_maxage_for(request.path), anon)
            response.headers["Vary"] = _vary_add(response.headers.get("Vary", ""))
        return response
