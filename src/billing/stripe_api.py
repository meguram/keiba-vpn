"""Stripe の REST API と Webhook 署名検証（SDK なし。標準の requests だけ）。カード番号は一切扱わない（Stripe のホスト型ページで入力される）。"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import uuid
from typing import Any

import requests

API = "https://api.stripe.com/v1"
TIMEOUT = 15


class BillingError(RuntimeError):
    """決済まわりの失敗。メッセージに秘密（キー・署名）を含めない。"""


def _headers(idem: str | None = None) -> dict[str, str]:
    key = os.environ.get("STRIPE_SECRET_KEY", "")
    if not key:
        raise BillingError("STRIPE_SECRET_KEY が未設定です")
    h = {"Authorization": f"Bearer {key}"}
    if idem:
        h["Idempotency-Key"] = idem
    return h


def _post(path: str, data: dict[str, Any], idem: str | None = None) -> dict[str, Any]:
    try:
        r = requests.post(f"{API}{path}", data=data, headers=_headers(idem or str(uuid.uuid4())), timeout=TIMEOUT)
    except requests.RequestException as e:
        raise BillingError(f"Stripe への接続に失敗しました: {type(e).__name__}") from None
    if r.status_code >= 400:
        try:
            msg = (r.json().get("error") or {}).get("message", "")
        except ValueError:
            msg = ""
        raise BillingError(f"Stripe のエラー（HTTP {r.status_code}）: {msg[:120]}")
    return r.json()


def create_checkout_session(*, price_id: str, client_reference_id: str, success_url: str, cancel_url: str, trial_days: int = 0) -> dict[str, Any]:
    """月額の購読の Checkout セッションを作る。client_reference_id には、元に戻せないログインID のキー（idk）を渡す（メールなどは渡さない）。"""
    data: dict[str, Any] = {
        "mode": "subscription",
        "line_items[0][price]": price_id,
        "line_items[0][quantity]": 1,
        "client_reference_id": client_reference_id,
        "success_url": success_url,
        "cancel_url": cancel_url,
        "locale": "ja",
        "billing_address_collection": "auto",
        "payment_method_types[0]": "card",
    }
    if trial_days > 0:
        data["subscription_data[trial_period_days]"] = trial_days
    return _post("/checkout/sessions", data)


def create_portal_session(*, customer_id: str, return_url: str) -> dict[str, Any]:
    """解約・カード変更・請求履歴の Customer Portal。"""
    return _post("/billing_portal/sessions", {"customer": customer_id, "return_url": return_url})


def verify_webhook(payload: bytes, sig_header: str, secret: str, *, tolerance: int = 300, now: float | None = None) -> dict[str, Any]:
    """Stripe-Signature（t=…,v1=…）を検証してイベントを返す。署名の不一致・古すぎる/先すぎる時刻は BillingError。"""
    if not secret or not sig_header:
        raise BillingError("署名がありません")
    parts = [p.split("=", 1) for p in sig_header.split(",") if "=" in p]
    ts = next((v for k, v in parts if k == "t"), "")
    sigs = [v for k, v in parts if k == "v1"]
    try:
        t = int(ts)
    except ValueError:
        raise BillingError("署名の時刻が読めません") from None
    expected = hmac.new(secret.encode(), f"{t}.".encode() + payload, hashlib.sha256).hexdigest()
    if not any(hmac.compare_digest(expected, s) for s in sigs):
        raise BillingError("署名が一致しません")
    if abs((now if now is not None else time.time()) - t) > tolerance:
        raise BillingError("署名の時刻が許容範囲外です（再送攻撃の疑い）")
    try:
        return json.loads(payload)
    except ValueError:
        raise BillingError("本文が JSON ではありません") from None
