"""
認証・セッション管理モジュール

開発者とビジターのページアクセスを分離する。
- 開発者: ログイン済みクッキーで全ページアクセス可能
- ビジター: 公開ページのみアクセス可能

セッションは署名付きクッキーで管理し、長期間キャッシュ可能。
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from typing import Optional

from fastapi import Request
from fastapi.responses import RedirectResponse

logger = logging.getLogger("api.auth")

COOKIE_NAME = "keiba_dev_session"
COOKIE_MAX_AGE = 30 * 24 * 3600  # 30日


def _get_secret_key() -> str:
    return os.environ.get("DEV_SECRET_KEY", "keiba-dev-default-secret-2026")


def _get_dev_password() -> str:
    return os.environ.get("DEV_PASSWORD", "")


def _sign(payload: str) -> str:
    key = _get_secret_key().encode()
    return hmac.new(key, payload.encode(), hashlib.sha256).hexdigest()


def _make_token(timestamp: int | None = None) -> str:
    ts = timestamp or int(time.time())
    payload = f"dev:{ts}"
    sig = _sign(payload)
    return f"{payload}:{sig}"


def _verify_token(token: str) -> bool:
    try:
        parts = token.split(":")
        if len(parts) != 3:
            return False
        role, ts_str, sig = parts
        if role != "dev":
            return False
        ts = int(ts_str)
        if time.time() - ts > COOKIE_MAX_AGE:
            return False
        expected = _sign(f"{role}:{ts_str}")
        return hmac.compare_digest(sig, expected)
    except Exception:
        return False


def is_developer(request: Request) -> bool:
    token = request.cookies.get(COOKIE_NAME, "")
    return _verify_token(token)


def create_session_response(redirect_to: str = "/") -> RedirectResponse:
    token = _make_token()
    response = RedirectResponse(url=redirect_to, status_code=303)
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        path="/",
    )
    return response


def clear_session_response(redirect_to: str = "/") -> RedirectResponse:
    response = RedirectResponse(url=redirect_to, status_code=303)
    response.delete_cookie(key=COOKIE_NAME, path="/")
    return response


def verify_password(password: str) -> bool:
    dev_pw = _get_dev_password()
    if not dev_pw:
        logger.warning("DEV_PASSWORD が設定されていません (.env に追加してください)")
        return False
    return hmac.compare_digest(password, dev_pw)


# ── ページ分類 ──

PUBLIC_PAGES: set[str] = {
    "/",
    "/login",
    "/race/{race_id}",
    "/tracking-difficulty",
    "/bloodline",
    "/bloodline-vector",
    "/pedigree-map",
    "/note-aptitude-race",
    "/course-bloodline",
    "/track-speed",
    "/myostatin",
}

DEV_ONLY_PAGES: set[str] = {
    "/monitor",
    "/data-viewer",
    "/betting",
}

PUBLIC_API_PREFIXES: list[str] = [
    "/api/health",
    "/api/predictions",
    "/api/race/",
    "/api/race-list/",
    "/api/scrape-dates",
    "/api/upcoming-races",
    "/api/scrape-status",
    "/api/data/",
    "/api/horse/",
    "/api/person/",
    "/api/bloodline",
    "/api/course-bloodline",
    "/api/track-speed",
    "/api/myostatin",
    "/api/pedigree-map",
    "/api/pedigree/",
    "/api/cushion",
    "/static/",
]

DEV_ONLY_API_PREFIXES: list[str] = [
    "/api/monitor/",
    "/api/scrape-trigger",
    "/api/scrape-jobs",
    "/api/train",
    "/api/backfill",
    "/api/race-lists-backfill",
    "/api/html-archive",
    "/api/gcs-stats",
    "/api/structure",
    "/api/auto-scrape",
    "/api/betting",
    "/api/odds/train",
    "/api/simulation",
    "/api/model/",
]


def is_public_path(path: str) -> bool:
    if path in PUBLIC_PAGES:
        return True
    if path == "/login":
        return True
    if path.startswith("/static/"):
        return True
    if path.startswith("/race/"):
        return True
    for prefix in PUBLIC_API_PREFIXES:
        if path.startswith(prefix):
            return True
    return False


def is_dev_only_path(path: str) -> bool:
    if path in DEV_ONLY_PAGES:
        return True
    for prefix in DEV_ONLY_API_PREFIXES:
        if path.startswith(prefix):
            return True
    return False


def requires_auth(path: str) -> bool:
    if is_public_path(path):
        return False
    return is_dev_only_path(path)
