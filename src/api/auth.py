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


def _make_token(timestamp: int | None = None, role: str = ROLE_DEV, uid: str = "") -> str:
    """セッショントークン。会員は uid（ログイン ID）も署名に含める（role:ts:uid16進:署名）。uid なしは従来形式（role:ts:署名）。"""
    ts = timestamp or int(time.time())
    payload = f"{role}:{ts}:{uid.encode().hex()}" if uid else f"{role}:{ts}"
    sig = _sign(payload)
    return f"{payload}:{sig}"


def _split_token(token: str) -> tuple[str, str, str, str] | None:
    """(role, ts, uid, sig)。形式が違えば None。uid は無ければ空。"""
    parts = (token or "").split(":")
    if len(parts) == 3:
        return parts[0], parts[1], "", parts[2]
    if len(parts) == 4:
        try:
            return parts[0], parts[1], bytes.fromhex(parts[2]).decode(), parts[3]
        except ValueError:
            return None
    return None


def token_role(token: str) -> str | None:
    """署名と有効期限が正しいトークンのロール（dev / member）。無効なら None。"""
    try:
        sp = _split_token(token)
        if sp is None:
            return None
        role, ts_str, uid, sig = sp
        if role not in (ROLE_DEV, ROLE_MEMBER):
            return None
        ts = int(ts_str)
        if time.time() - ts > COOKIE_MAX_AGE:
            return None
        payload = f"{role}:{ts_str}:{uid.encode().hex()}" if uid else f"{role}:{ts_str}"
        if not hmac.compare_digest(sig, _sign(payload)):
            return None
        if role == ROLE_MEMBER and uid and not member_active(uid):      # 無効化・削除したユーザーのセッションは使えなくする
            return None
        return role
    except Exception:
        return None


def token_user(token: str) -> str:
    """有効なトークンのログイン ID（会員のみ。開発者・従来形式・無効は空）。"""
    if token_role(token) != ROLE_MEMBER:
        return ""
    sp = _split_token(token)
    return sp[2] if sp else ""


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
        sp = _split_token(token)
        if sp is None:
            return "invalid"
        role, ts_str, uid, sig = sp
        if role not in (ROLE_DEV, ROLE_MEMBER):
            return "invalid"
        ts = int(ts_str)
        payload = f"{role}:{ts_str}:{uid.encode().hex()}" if uid else f"{role}:{ts_str}"
        if not hmac.compare_digest(sig, _sign(payload)):
            return "invalid"
        if time.time() - ts > COOKIE_MAX_AGE:
            return "expired"
        if role == ROLE_MEMBER and uid and not member_active(uid):
            return "invalid"
        return "valid"
    except Exception:
        return "invalid"


def _request_is_secure(request: Request) -> bool:
    if request.url.scheme == "https":
        return True
    forwarded = request.headers.get("x-forwarded-proto", "")
    return forwarded.split(",")[0].strip().lower() == "https"


def create_session_response(redirect_to: str = "/", request: Request | None = None, role: str = ROLE_DEV, uid: str = "") -> RedirectResponse:
    token = _make_token(role=role, uid=uid)
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


# ユーザー管理（python -m src.scripts.ops.manage_users）が書く、git に入れないファイル。config/members.json（動作確認用）と合わせて読み、同じ ID は後者が優先
MEMBERS_LOCAL_FILE = os.environ.get("KEIBA_MEMBERS_FILE") or os.path.join(os.path.dirname(_MEMBERS_FILE), "..", "data", "local", "auth", "members.json")
_members_cache: dict = {"key": None, "users": []}


def _read_users(path: str) -> list[dict]:
    import json

    try:
        with open(path, encoding="utf-8") as f:
            users = (json.load(f) or {}).get("users") or []
    except (OSError, ValueError):
        return []
    return [u for u in users if isinstance(u, dict) and (u.get("id") or u.get("idk"))]


def _load_members() -> list[dict]:
    """会員の一覧。ファイルの更新時刻が変わったときだけ読み直す（毎リクエストでも軽い）。"""
    paths = [_MEMBERS_FILE, MEMBERS_LOCAL_FILE]
    key = tuple((p, os.path.getmtime(p) if os.path.exists(p) else None) for p in paths)
    if _members_cache["key"] != key:
        merged: dict[str, dict] = {}
        for p in paths:
            for u in _read_users(p):
                merged[str(u.get("idk") or u.get("id"))] = u
        _members_cache.update(key=key, users=list(merged.values()))
    return _members_cache["users"]


# ── ID・パスワードを「漏れても読めない」形で持つ ──
# ログイン ID は平文で保存せず、サーバだけが持つ秘密（ペッパー）つきの HMAC（idk）だけを持つ。パスワードは PBKDF2 のあとにペッパーで HMAC をかける。
# 会員ファイルだけが漏れても、誰が登録されているか・パスワードが何かは分からず、推測の検証もできない。管理者本人も元の ID・パスワードは読み出せない（再設定と、ID を入力しての照合だけ）。
# ペッパーは KEIBA_MEMBERS_PEPPER、無ければ DEV_SECRET_KEY から導く（変えると既存ユーザーは使えなくなる＝作り直し）。


def _pepper() -> bytes:
    explicit = os.environ.get("KEIBA_MEMBERS_PEPPER", "")
    if explicit:
        return explicit.encode()
    return hmac.new(_get_secret_key().encode(), b"keiba-members-pepper-v1", hashlib.sha256).digest()


def normalize_id(user_id: str) -> str:
    return (user_id or "").strip().lower()


def id_key(user_id: str) -> str:
    """ログイン ID の検索キー（HMAC-SHA256、ペッパーつき）。元の ID には戻せない。"""
    return hmac.new(_pepper(), b"id:" + normalize_id(user_id).encode(), hashlib.sha256).hexdigest()


def password_digest(password: str, salt: bytes, iterations: int) -> str:
    """PBKDF2-SHA256 の結果にペッパーで HMAC をかけた値（algo = pbkdf2_sha256_pepper）。"""
    pb = hashlib.pbkdf2_hmac("sha256", (password or "").encode(), salt, iterations)
    return hmac.new(_pepper(), b"pw:" + pb, hashlib.sha256).hexdigest()


def member_record(user_id: str) -> dict | None:
    """ID（平文）に対応するアカウント。新形式は idk で、動作確認用の旧形式（config/members.json の test）は id で探す。"""
    if not normalize_id(user_id):
        return None
    k = id_key(user_id)
    for u in _load_members():
        if u.get("idk") == k or (u.get("id") and u.get("id") == (user_id or "").strip()):
            return u
    return None


def member_active(user_id: str) -> bool:
    """ログイン可能なユーザーか（存在し、enabled が false でなく、requires_env があればその環境変数が有効）。"""
    rec = member_record(user_id)
    return bool(rec) and rec.get("enabled", True) is not False and (not rec.get("requires_env") or _env_on(str(rec["requires_env"])))


def member_display_name(user_id: str) -> str:
    """画面に出す名前。ID は保存しないので、ログイン時に本人が入力した ID（セッションに入っている）をそのまま出す。"""
    return (user_id or "").strip()


def verify_member_login(user_id: str, password: str) -> bool:
    """会員の ID・パスワードを検証する。存在しない ID でも同じ計算量にする。無効・requires_env が未設定のアカウントは常に不可。"""
    password = password or ""
    rec = member_record(user_id)
    enabled = member_active(user_id)
    try:
        salt = bytes.fromhex(rec["salt"]) if rec else _DUMMY_SALT
        iters = int(rec.get("iterations") or 200000) if rec else 200000
        want = rec["hash"] if rec else ""
    except (KeyError, TypeError, ValueError):
        return False
    algo = (rec or {}).get("algo", "pbkdf2_sha256")
    if algo == "pbkdf2_sha256_pepper":
        got = password_digest(password, salt, iters)
    else:                                   # 旧形式（config/members.json の test）
        got = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iters).hex()
    return bool(enabled) and algo in ("pbkdf2_sha256", "pbkdf2_sha256_pepper") and hmac.compare_digest(got, want)


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
    # 会員の登録・管理（管理者のみ）
    "/admin/users",
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
