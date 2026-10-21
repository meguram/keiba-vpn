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


ROLE_DEV = "dev"
ROLE_MEMBER = "member"


def _make_token(timestamp: int | None = None, role: str = ROLE_DEV) -> str:
    ts = timestamp or int(time.time())
    payload = f"{role}:{ts}"
    sig = _sign(payload)
    return f"{payload}:{sig}"


def token_role(token: str) -> str | None:
    """署名と有効期限が正しいトークンのロール（dev / member）。無効なら None。"""
    try:
        parts = token.split(":")
        if len(parts) != 3:
            return None
        role, ts_str, sig = parts
        if role not in (ROLE_DEV, ROLE_MEMBER):
            return None
        ts = int(ts_str)
        if time.time() - ts > COOKIE_MAX_AGE:
            return None
        expected = _sign(f"{role}:{ts_str}")
        return role if hmac.compare_digest(sig, expected) else None
    except Exception:
        return None


def _verify_token(token: str) -> bool:
    """開発者（dev）のトークンか。会員（member）のトークンは False（開発者向け機能を開かない）。"""
    return token_role(token) == ROLE_DEV


_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})


def _hostname(value: str) -> str:
    v = (value or "").strip().lower()
    if v.startswith("["):  # [::1]:23100
        return v.split("]")[0] + "]"
    return v.split(":")[0]


def _is_loopback(addr: str | None) -> bool:
    """127.0.0.0/8・::1・IPv4 射影（::ffff:127.0.0.1）をループバックとみなす。解釈できなければ False。"""
    import ipaddress

    s = (addr or "").strip().strip("[]")
    if not s:
        return False
    try:
        ip = ipaddress.ip_address(s)
    except ValueError:
        return False
    if getattr(ip, "ipv4_mapped", None) is not None:
        ip = ip.ipv4_mapped
    return ip.is_loopback


def dev_auto_login_applies(remote_addr: str | None, headers) -> bool:
    """dev 環境で、この PC 上のブラウザからのリクエストなら自動的にログイン済みとして扱う。

    次を全て満たすときだけ True（それ以外は通常の Cookie 認証）:
      - KEIBA_ENV が明示的に dev（未設定は prod 扱いなので対象外）かつ KEIBA_DEV_AUTO_LOGIN が 0 でない
      - 接続元がループバック（127.0.0.1 / ::1）
      - Host / X-Forwarded-Host が localhost 系、X-Forwarded-For があればループバックのみ
        （tcpexposer 等の公開トンネル経由は Host が公開ドメインになるため対象外）
    """
    import os

    if (os.environ.get("KEIBA_ENV") or "").strip().lower() not in ("dev", "development", "local"):
        return False
    if (os.environ.get("KEIBA_DEV_AUTO_LOGIN") or "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    if not _is_loopback(remote_addr):
        return False

    def h(name: str) -> str:
        return (headers.get(name) or "").strip()

    for name in ("x-forwarded-host", "host"):
        v = h(name)
        if v and _hostname(v.split(",")[0]) not in _LOCAL_HOSTS:
            return False
    if h("forwarded"):
        return False
    xff = h("x-forwarded-for")
    if xff and not all(_is_loopback(x) for x in xff.split(",")):
        return False
    return True


def is_developer(request: Request) -> bool:
    if dev_auto_login_applies(getattr(request.client, "host", None), request.headers):
        return True
    token = request.cookies.get(COOKIE_NAME, "")
    return _verify_token(token)


def is_member(request: Request) -> bool:
    """会員限定の項目を見られるか（開発者 dev は会員を兼ねる）。"""
    if dev_auto_login_applies(getattr(request.client, "host", None), request.headers):
        return True
    return token_role(request.cookies.get(COOKIE_NAME, "")) is not None


def classify_session(request: Request) -> str:
    """セッションクッキーの状態を分類する（ユーザ向けメッセージ分岐用）。

    - "valid": 有効なセッション
    - "none": クッキー自体が無い（未ログイン、初回アクセス等）
    - "expired": 署名は正しいが有効期限（30日）切れ
    - "invalid": クッキーが存在するが形式不正・署名検証失敗（改ざん・別環境のクッキー等）
    """
    if dev_auto_login_applies(getattr(request.client, "host", None), request.headers):
        return "valid"
    token = request.cookies.get(COOKIE_NAME, "")
    if not token:
        return "none"
    try:
        parts = token.split(":")
        if len(parts) != 3:
            return "invalid"
        role, ts_str, sig = parts
        if role not in (ROLE_DEV, ROLE_MEMBER):
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


def create_session_response(redirect_to: str = "/", request: Request | None = None, role: str = ROLE_DEV) -> RedirectResponse:
    token = _make_token(role=role)
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


# ── 会員アカウント（config/members.json。パスワードはハッシュだけを持つ） ──

_MEMBERS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config", "members.json")
_DUMMY_SALT = bytes(16)


def _env_on(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


def _load_members() -> list[dict]:
    import json

    try:
        with open(_MEMBERS_FILE, encoding="utf-8") as f:
            users = (json.load(f) or {}).get("users") or []
    except (OSError, ValueError):
        return []
    return [u for u in users if isinstance(u, dict) and u.get("id")]


def verify_member_login(user_id: str, password: str) -> bool:
    """会員の ID・パスワードを検証する。requires_env の環境変数が有効でないアカウントは常に不可（既定オフ）。"""
    user_id = (user_id or "").strip()
    password = password or ""
    rec = next((u for u in _load_members() if u.get("id") == user_id), None)
    enabled = bool(rec) and (not rec.get("requires_env") or _env_on(str(rec["requires_env"])))
    try:
        salt = bytes.fromhex(rec["salt"]) if rec else _DUMMY_SALT
        iters = int(rec.get("iterations") or 200000) if rec else 200000
        want = rec["hash"] if rec else ""
    except (KeyError, TypeError, ValueError):
        return False
    got = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iters).hex()  # 存在しない ID でも同じ計算量にする
    return bool(enabled) and rec.get("algo", "pbkdf2_sha256") == "pbkdf2_sha256" and hmac.compare_digest(got, want)


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
