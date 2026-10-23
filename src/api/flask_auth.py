"""Flask 認証（AREA-02 / AREA-03）。"""

from __future__ import annotations

from flask import Request, jsonify, request

from src.api.auth import (
    dev_auto_login_applies,
    COOKIE_MAX_AGE,
    COOKIE_NAME,
    ROLE_DEV,
    ROLE_MEMBER,
    _make_token,
    _verify_token,
    token_role,
    token_tier,
    tier_fields,
    verify_member_login,
    verify_password,
)


def is_logged_in(req: Request | None = None) -> bool:
    req = req or request
    if dev_auto_login_applies(req.remote_addr, req.headers):
        return True
    token = req.cookies.get(COOKIE_NAME, "")
    return _verify_token(token)


def is_member_session(req: Request | None = None) -> bool:
    """有料ゾーン（会員限定の項目）を見られるか（有料会員・開発者。無料のログイン会員は含まない）。開発者向け機能の判定は is_logged_in のまま。"""
    req = req or request
    if dev_auto_login_applies(req.remote_addr, req.headers):
        return True
    return token_tier(req.cookies.get(COOKIE_NAME, "")) in ("dev", "paid")


def login_response(password: str, remember: bool = True, user_id: str = ""):
    """user_id が空なら開発者パスワード、あれば会員（config/members.json）として認証する。"""
    if (user_id or "").strip():
        if not verify_member_login(user_id, password):
            return jsonify({"error": "invalid credentials"}), 401
        role = ROLE_MEMBER
        uid = user_id.strip()
    else:
        uid = ""
        if not verify_password(password):
            return jsonify({"error": "invalid password"}), 401
        role = ROLE_DEV
    token = _make_token(role=role, uid=uid)
    resp = jsonify({"status": "ok", "logged_in": True, "role": role})
    max_age = COOKIE_MAX_AGE if remember else None
    resp.set_cookie(COOKIE_NAME, token, max_age=max_age, httponly=True, samesite="Lax", path="/")
    return resp


def logout_response():
    resp = jsonify({"status": "ok", "logged_in": False})
    resp.delete_cookie(COOKIE_NAME, path="/")
    return resp


def require_login(f):
    """デコレータ: ログイン必須 API。"""
    from functools import wraps

    @wraps(f)
    def wrapper(*args, **kwargs):
        if not is_logged_in():
            return jsonify({"error": "login required"}), 401
        return f(*args, **kwargs)

    return wrapper


def require_internal(f):
    """管理系: 127.0.0.1 / VPN 内のみ。"""
    from functools import wraps

    @wraps(f)
    def wrapper(*args, **kwargs):
        ip = request.remote_addr or ""
        if ip not in ("127.0.0.1", "::1") and not ip.startswith("10."):
            return jsonify({"error": "forbidden"}), 403
        return f(*args, **kwargs)

    return wrapper


def require_member(f):
    """デコレータ: 会員（または開発者）必須 API。匿名は 401。"""
    from functools import wraps

    @wraps(f)
    def wrapper(*args, **kwargs):
        if not is_member_session():
            return jsonify({"error": "この情報は会員限定です。ログインしてください。", "reason": "member_required", "login_url": "/login"}), 401
        return f(*args, **kwargs)

    return wrapper


def get_auth_status() -> dict:
    """現在のリクエストの認証状態。ゲスト / 無料のログイン会員（free）/ 有料会員（member。有料ゾーンを見られる）/ 管理者（dev）の 4 区分。"""
    from src.api.auth import member_display_name, token_user

    token = request.cookies.get(COOKIE_NAME, "")
    f = tier_fields(token, dev_auto_login_applies(request.remote_addr, request.headers))
    uid = token_user(token) if f["logged_in"] and not f["is_developer"] else ""
    name = "管理者" if f["is_developer"] else (member_display_name(uid) if uid else ("会員" if f["logged_in"] else ""))
    return {**f, "user_id": uid or None, "user_name": name or None}
