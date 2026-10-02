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

from src.config.deployment import keiba_env_raw

logger = logging.getLogger("api.auth")

COOKIE_NAME = "keiba_dev_session"
COOKIE_MAX_AGE = 30 * 24 * 3600  # 30日


_INSECURE_DEV_SECRET = "keiba-dev-default-secret-2026"
_warned_insecure_secret = False


def _get_secret_key() -> str:
    """セッション Cookie の署名鍵。stg/prod では DEV_SECRET_KEY 必須（既定値だと Cookie を偽造できる）。"""
    global _warned_insecure_secret
    key = os.environ.get("DEV_SECRET_KEY", "")
    if key:
        return key
    if keiba_env_raw() in ("stg", "staging", "prod", "production"):
        raise RuntimeError("DEV_SECRET_KEY が未設定です。stg/prod ではランダムな値を .env.<env> に設定してください")
    if not _warned_insecure_secret:
        logger.warning("DEV_SECRET_KEY 未設定: 開発用の固定鍵を使用します（dev 専用。公開環境では使わないこと）")
        _warned_insecure_secret = True
    return _INSECURE_DEV_SECRET


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


def classify_session(request: Request) -> str:
    """セッションクッキーの状態を分類する（ユーザ向けメッセージ分岐用）。

    - "valid": 有効なセッション
    - "none": クッキー自体が無い（未ログイン、初回アクセス等）
    - "expired": 署名は正しいが有効期限（30日）切れ
    - "invalid": クッキーが存在するが形式不正・署名検証失敗（改ざん・別環境のクッキー等）
    """
    token = request.cookies.get(COOKIE_NAME, "")
    if not token:
        return "none"
    try:
        parts = token.split(":")
        if len(parts) != 3:
            return "invalid"
        role, ts_str, sig = parts
        if role != "dev":
            return "invalid"
        ts = int(ts_str)
        expected = _sign(f"{role}:{ts_str}")
        if not hmac.compare_digest(sig, expected):
            return "invalid"
        if time.time() - ts > COOKIE_MAX_AGE:
            return "expired"
        return "valid"
    except Exception:
        return "invalid"


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


_WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _to_v1_path(path: str) -> str | None:
    """FastAPI の旧ルート `/api/X` を、許可リストの基準である `/api/v1/X` に読み替える。"""
    if path.startswith("/api/") and not path.startswith("/api/v1/"):
        return "/api/v1/" + path[len("/api/"):]
    return None


def requires_auth(path: str, method: str = "GET") -> bool:
    """開発者ログインが必要か。

    許可リストは `/api/v1/...` 前置きで書かれているが、FastAPI の実ルートは `/api/...`。
    旧ルートの読み取り（監視ポータル等が Cookie 無しで呼ぶ）は従来どおり通し、
    書き込み系メソッドだけ `/api/v1/` 相当の判定を適用する。
    """
    if is_public_path(path):
        return False
    if is_dev_only_path(path):
        return True
    legacy = _to_v1_path(path)
    if legacy is not None and method.upper() in _WRITE_METHODS:
        return not is_public_path(legacy) and is_dev_only_path(legacy)
    return False
