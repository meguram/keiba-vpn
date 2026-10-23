import hashlib
import hmac
import json
import time

import pytest

from src.api import auth
from src.billing import service, stripe_api
from src.billing.stripe_api import BillingError
from src.scripts.ops import manage_users as mu

SECRET = "whsec_test"
PW = "correct-horse-battery"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "MEMBERS_LOCAL_FILE", str(tmp_path / "auth" / "members.json"))
    monkeypatch.setattr(service, "EVENTS_PATH", tmp_path / "events.json")
    monkeypatch.setattr(service, "_SIGNUP_HITS", {})
    monkeypatch.setenv("DEV_SECRET_KEY", "test-secret")
    monkeypatch.delenv("KEIBA_MEMBERS_PEPPER", raising=False)
    for k in ("KEIBA_BILLING", "STRIPE_SECRET_KEY", "STRIPE_WEBHOOK_SECRET", "STRIPE_PRICE_ID"):
        monkeypatch.delenv(k, raising=False)
    auth._members_cache["key"] = None
    return monkeypatch


def turn_on(mp, key="sk_test_x"):
    mp.setenv("KEIBA_BILLING", "1")
    mp.setenv("STRIPE_SECRET_KEY", key)
    mp.setenv("STRIPE_WEBHOOK_SECRET", SECRET)
    mp.setenv("STRIPE_PRICE_ID", "price_1")
    auth._members_cache["key"] = None


def sign(payload: bytes, secret=SECRET, t=None):
    t = int(t if t is not None else time.time())
    return f"t={t},v1=" + hmac.new(secret.encode(), f"{t}.".encode() + payload, hashlib.sha256).hexdigest()


def test_disabled_by_default_and_live_key_needs_license(env):
    assert not service.enabled()
    turn_on(env)
    assert service.enabled()                                   # テストキーは確認前でも可
    turn_on(env, key="sk_live_x")
    assert not service.enabled()                               # 本番キーはライセンス確認まで動かない
    env.setattr(service, "config", lambda: {"license_confirmed": True})
    assert service.enabled()


def test_webhook_signature(env):
    body = b'{"id":"evt_1"}'
    assert stripe_api.verify_webhook(body, sign(body), SECRET)["id"] == "evt_1"
    with pytest.raises(BillingError):
        stripe_api.verify_webhook(body, sign(body, secret="other"), SECRET)
    with pytest.raises(BillingError):
        stripe_api.verify_webhook(body + b" ", sign(body), SECRET)            # 改ざん
    with pytest.raises(BillingError):
        stripe_api.verify_webhook(body, sign(body, t=time.time() - 3600), SECRET)       # 古い（リプレイ）


def test_billing_gate_on_login(env):
    mu.create_user("alice", PW, billing={"status": "pending", "created_ts": time.time()})
    rec = auth.member_record("alice")
    assert not auth.billing_allows(rec)
    for st, ok in (("active", True), ("trialing", True), ("canceled", False), ("incomplete", False)):
        assert auth.billing_allows({"billing": {"status": st}}) is ok
    now = time.time()
    assert auth.billing_allows({"billing": {"status": "past_due", "period_end": now - 86400}}, now=now)           # 猶予内
    assert not auth.billing_allows({"billing": {"status": "past_due", "period_end": now - 10 * 86400}}, now=now)
    assert auth.billing_allows({})                                                                                # 管理者が作ったアカウントは対象外


def _signup(env, user="bob"):
    turn_on(env)
    env.setattr(stripe_api, "create_checkout_session", lambda **kw: {"url": "https://checkout.stripe.com/x", **kw})
    return service.start_signup(user, PW, "1.2.3.4")


def test_signup_then_webhook_activates_once(env):
    assert _signup(env).startswith("https://checkout")
    idk = auth.id_key("bob")
    assert not auth.verify_member_login("bob", PW)                                    # 支払い前はログインできない
    ev = {"id": "evt_1", "type": "checkout.session.completed", "data": {"object": {"client_reference_id": idk, "customer": "cus_1", "subscription": "sub_1", "payment_status": "paid"}}}
    assert service.handle_event(ev) == "ok"
    assert auth.verify_member_login("bob", PW)
    assert service.handle_event(ev) == "duplicate"                                    # 同じイベントは二重に処理しない
    assert service.status_of("bob")["status"] == "active"
    service.handle_event({"id": "evt_2", "type": "customer.subscription.deleted", "data": {"object": {"customer": "cus_1"}}})
    assert not auth.verify_member_login("bob", PW)                                    # 解約で使えなくなる


def test_unmatched_event_is_retryable_and_not_marked_seen(env):
    turn_on(env)
    ev = {"id": "evt_9", "type": "customer.subscription.updated", "data": {"object": {"customer": "cus_none", "status": "active"}}}
    with pytest.raises(service.Unmatched):
        service.handle_event(ev)
    assert "evt_9" not in service._seen_ids()


def test_pending_cleanup_and_stripe_failure_leaves_no_account(env):
    turn_on(env)
    def boom(**kw):
        raise BillingError("down")
    env.setattr(stripe_api, "create_checkout_session", boom)
    with pytest.raises(BillingError):
        service.start_signup("carol", PW, "1.1.1.1")
    assert [u for u in mu.list_users() if u.get("fp") != "-"] == []
    mu.create_user("old", PW, billing={"status": "pending", "created_ts": time.time() - 48 * 3600})
    assert mu.remove_pending(24) == 1


def test_signup_rate_limit(env):
    turn_on(env)
    env.setattr(stripe_api, "create_checkout_session", lambda **kw: {"url": "u"})
    for i in range(5):
        service.start_signup(f"user{i}", PW, "9.9.9.9")
    with pytest.raises(BillingError):
        service.start_signup("user99", PW, "9.9.9.9")


def test_routes_404_when_off_and_webhook_end_to_end(env):
    from flask import Flask

    from src.api.v1.routes.billing import bp

    app = Flask(__name__)
    app.register_blueprint(bp, url_prefix="/api/v1")
    c = app.test_client()
    assert c.post("/api/v1/billing/signup", json={"user_id": "x", "password": PW}).status_code == 404
    assert c.post("/api/v1/billing/webhook", data=b"{}").status_code == 404
    _signup(env, "dave")
    body = json.dumps({"id": "evt_7", "type": "checkout.session.completed", "data": {"object": {"client_reference_id": auth.id_key("dave"), "customer": "cus_7", "payment_status": "paid"}}}).encode()
    assert c.post("/api/v1/billing/webhook", data=body, headers={"Stripe-Signature": "t=1,v1=bad"}).status_code == 400
    assert c.post("/api/v1/billing/webhook", data=body, headers={"Stripe-Signature": sign(body)}).status_code == 200
    assert auth.verify_member_login("dave", PW)
    assert c.post("/api/v1/billing/signup", json={"user_id": "e", "password": PW}, headers={"Origin": "https://evil.example"}).status_code == 403
