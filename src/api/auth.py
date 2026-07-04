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


def _request_is_secure(request: Request) -> bool:
    if request.url.scheme == "https":
        return True
    forwarded = request.headers.get("x-forwarded-proto", "")
    return forwarded.split(",")[0].strip().lower() == "https"


def create_session_response(redirect_to: str = "/", request: Request | None = None) -> RedirectResponse:
    token = _make_token()
    response = RedirectResponse(url=redirect_to, status_code=303)
    secure = _request_is_secure(request) if request is not None else False
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=secure,
        path="/",
    )
    return response


def clear_session_response(redirect_to: str = "/", request: Request | None = None) -> RedirectResponse:
    response = RedirectResponse(url=redirect_to, status_code=303)
    secure = _request_is_secure(request) if request is not None else False
    response.delete_cookie(key=COOKIE_NAME, path="/", secure=secure)
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
    # ④ AI 予測
    "/tracking-difficulty",
    # ③ 血統
    "/bloodline",
    "/bloodline-vector",
    "/pedigree-map",
    "/bloodline-cluster",
    "/course-bloodline",
    "/pedigree-race-stats",
    "/myostatin",
    # ⑤ データ分析
    "/note-aptitude-race",
    "/track-speed",
    "/growth-curve",
}

DEV_ONLY_PAGES: set[str] = {
    # ① 開発者モード（データチェック・スクレイピング）
    "/monitor",
    "/data-viewer",
    "/queue-status",
    "/server-logs",
    "/scrape-upcoming",
    # ② 馬券の最適化
    "/betting",
    # ③ 馬場速度 計算ロジック解説
    "/track-speed/dev",
}

PUBLIC_API_PREFIXES: list[str] = [
    "/api/v1/health",
    "/api/v1/races/",
    "/api/v1/race-list/",
    "/api/v1/scrape-dates",
    "/api/v1/upcoming-races",
    "/api/v1/scrape-status",
    "/api/v1/data/",
    "/api/v1/horse/",
    "/api/v1/horse-names/",
    "/api/v1/person/",
    # ③ 血統
    "/api/v1/bloodline",
    "/api/v1/course-bloodline",
    "/api/v1/myostatin",
    "/api/v1/pedigree-map",
    "/api/v1/pedigree/",
    "/api/v1/pedigree-race-stats",
    "/api/v1/stallion-sire-tree",
    # ⑤ データ分析
    "/api/v1/track-speed",
    "/api/v1/growth-curve",
    "/api/v1/growth-curve/status",
    "/api/v1/cushion",
    "/api/v1/auth/status",
    "/static/",
]

DEV_ONLY_API_PREFIXES: list[str] = [
    "/api/v1/admin/",
    "/api/v1/monitor/",
    # ① 開発者モード（スクレイピング・データチェック）
    "/api/v1/scrape-trigger",
    "/api/v1/scrape-jobs",
    "/api/v1/scrape-queue",
    "/api/v1/check-scraped-status",
    "/api/v1/fetch-future-calendar",
    "/api/v1/structure",
    "/api/v1/html-archive",
    "/api/v1/auto-scrape",
    "/api/v1/gcs-stats",
    # ④ AI 予測（モデル学習・追走難度の訓練）
    "/api/v1/tracking-difficulty/train",
    "/api/v1/train",
    "/api/v1/model/",
    # ② 馬券の最適化
    "/api/v1/betting",
    "/api/v1/odds/train",
    "/api/v1/odds/snapshot",
    "/api/v1/simulation",
    # データバックフィル
    "/api/v1/backfill",
    "/api/v1/race-lists-backfill",
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
