"""決済（Stripe）の API。無効の間（KEIBA_BILLING 未設定など）はすべて 404 で、画面からは呼ばれない。仕様: docs/operations/billing.md"""

from __future__ import annotations

import os

from flask import Blueprint, jsonify, request

from src.api.flask_auth import COOKIE_NAME, require_member
from src.billing import service
from src.billing.stripe_api import BillingError, verify_webhook

bp = Blueprint("v1_billing", __name__)


def _off():
    return jsonify({"error": "not found"}), 404


def _origin_ok() -> bool:
    """別サイトからのフォーム送信（CSRF）を防ぐ。Origin があるときは許可リストにあること。"""
    origin = request.headers.get("Origin")
    return not origin or origin in (service.config().get("allowed_origins") or [])


def _ip() -> str:
    return request.headers.get("CF-Connecting-IP") or request.remote_addr or "?"


@bp.get("/billing/status")
def billing_status():
    if not service.enabled():
        return _off()
    from src.api.auth import token_user

    uid = token_user(request.cookies.get(COOKIE_NAME, ""))
    return jsonify(service.status_of(uid) if uid else {"managed": False})


@bp.post("/billing/signup")
def billing_signup():
    if not service.enabled():
        return _off()
    if not _origin_ok():
        return jsonify({"error": "forbidden"}), 403
    body = request.get_json(silent=True) or {}
    try:
        url = service.start_signup(str(body.get("user_id") or ""), str(body.get("password") or ""), _ip())
    except BillingError as e:
        return jsonify({"error": str(e)}), 429 if "多すぎ" in str(e) else 400
    except Exception as e:  # noqa: BLE001  UserError（ID・パスワードの規則）は文言をそのまま返す
        from src.scripts.ops.manage_users import UserError

        if isinstance(e, UserError):
            return jsonify({"error": str(e)}), 400
        return jsonify({"error": "登録を開始できませんでした"}), 502
    return jsonify({"checkout_url": url})


@bp.post("/billing/portal")
@require_member
def billing_portal():
    if not service.enabled():
        return _off()
    if not _origin_ok():
        return jsonify({"error": "forbidden"}), 403
    from src.api.auth import token_user

    uid = token_user(request.cookies.get(COOKIE_NAME, ""))
    try:
        return jsonify({"portal_url": service.portal_url(uid)})
    except BillingError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:  # noqa: BLE001
        return jsonify({"error": "管理画面を開けませんでした"}), 502


@bp.post("/billing/webhook")
def billing_webhook():
    if not service.enabled():
        return _off()
    try:
        event = verify_webhook(request.get_data(), request.headers.get("Stripe-Signature", ""), os.environ["STRIPE_WEBHOOK_SECRET"])
    except BillingError:
        return jsonify({"error": "bad signature"}), 400
    try:
        result = service.handle_event(event)
    except service.Unmatched:
        return jsonify({"error": "unmatched, retry"}), 409           # 先に届いたイベント。Stripe が再送する
    except Exception:  # noqa: BLE001
        return jsonify({"error": "failed"}), 500
    return jsonify({"result": result})
