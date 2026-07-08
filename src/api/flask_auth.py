"""Flask 認証（AREA-02 / AREA-03）。"""

from __future__ import annotations

from flask import Request, jsonify, request

from src.api.auth import (
    COOKIE_MAX_AGE,
    COOKIE_NAME,
    _make_token,
    _verify_token,
    verify_password,
)


def is_logged_in(req: Request | None = None) -> bool:
    req = req or request
    token = req.cookies.get(COOKIE_NAME, "")
    return _verify_token(token)


def login_response(password: str, remember: bool = True):
    if not verify_password(password):
        return jsonify({"error": "invalid password"}), 401
    token = _make_token()
    resp = jsonify({"status": "ok", "logged_in": True})
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
    """デコレータ: メンバー必須 API。MVP では require_login と同等。"""
    from functools import wraps

    @wraps(f)
    def wrapper(*args, **kwargs):
        token = request.cookies.get(COOKIE_NAME, "")
        if not _verify_token(token):
            return jsonify({"error": "member required", "reason": "not_member"}), 401
        return f(*args, **kwargs)

    return wrapper


def get_auth_status() -> dict:
    """現在のリクエストの認証状態を返す。MVP では is_member == logged_in。"""
    logged_in = is_logged_in()
    return {
        "logged_in": logged_in,
        "is_member": logged_in,
    }
