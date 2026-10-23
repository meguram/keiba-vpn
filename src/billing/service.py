"""決済の業務ロジック。権限（会員になれるか）は、画面の戻りではなく Stripe の Webhook を正にして反映する。

流れ（新規登録→支払い）:
  1. 訪問者がログインIDとパスワードを送る → 仮登録（billing.status=pending。支払いが済むまでログインできない）→ Stripe Checkout の URL を返す。
  2. 支払いが済むと Stripe が Webhook を送る → 署名を検証 → 会員ファイルの billing を更新（active など）→ ログインできるようになる。
  3. 解約・支払い失敗・期間終了も Webhook で反映（解約・カード変更は Stripe の Customer Portal）。
個人情報: 会員ファイルには、ログインIDの変換値（idk）と Stripe の顧客 ID だけを紐づける。メールアドレス等は Stripe 側にだけある。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from src.billing import stripe_api as sa
from src.billing.stripe_api import BillingError

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "billing.json"
EVENTS_PATH = Path(os.environ.get("KEIBA_BILLING_EVENTS") or ROOT / "data" / "local" / "billing" / "events.json")
_SIGNUP_HITS: dict[str, list[float]] = {}


class Unmatched(BillingError):
    """Webhook の対象ユーザーがまだ見つからない（先に届いた購読イベントなど）。Stripe に再送してもらう。"""


def config() -> dict[str, Any]:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _on(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


def live_key() -> bool:
    return (os.environ.get("STRIPE_SECRET_KEY") or "").startswith(("sk_live_", "rk_live_"))


def enabled() -> bool:
    """決済を動かしてよいか。KEIBA_BILLING=1 ＋ 必要な環境変数が揃い、本番キーなら config の license_confirmed が true であること（テストキーは確認前でも可）。"""
    if not _on("KEIBA_BILLING"):
        return False
    if not (os.environ.get("STRIPE_SECRET_KEY") and os.environ.get("STRIPE_WEBHOOK_SECRET") and os.environ.get("STRIPE_PRICE_ID")):
        return False
    return bool(config().get("license_confirmed")) if live_key() else True


def _rate_limited(ip: str, now: float) -> bool:
    limit = int(config().get("signup_per_ip_per_hour") or 5)
    hits = [t for t in _SIGNUP_HITS.get(ip, []) if now - t < 3600]
    if len(hits) >= limit:
        _SIGNUP_HITS[ip] = hits
        return True
    hits.append(now)
    _SIGNUP_HITS[ip] = hits
    return False


def start_signup(user_id: str, password: str, ip: str, now: float | None = None) -> str:
    """仮登録して Checkout の URL を返す。ID の規則・パスワードの規則は manage_users と同じ（UserError は呼び出し側でそのまま文言に使える）。"""
    from src.api import auth
    from src.scripts.ops import manage_users as mu

    now = now if now is not None else time.time()
    if _rate_limited(ip or "?", now):
        raise BillingError("短時間に試行が多すぎます。しばらくしてからお試しください")
    cfg = config()
    mu.remove_pending(float(cfg.get("pending_ttl_hours") or 24), now)
    mu.create_user(user_id, password, billing={"status": "pending", "created_ts": now})
    idk = auth.id_key(user_id)
    try:
        sess = sa.create_checkout_session(price_id=os.environ["STRIPE_PRICE_ID"], client_reference_id=idk, success_url=cfg["success_url"],
                                          cancel_url=cfg["cancel_url"], trial_days=int(cfg.get("trial_days") or 0))
    except Exception:
        mu.remove_user(idk[:8])                      # Stripe に繋がらなかった仮登録は残さない
        raise
    return str(sess["url"])


def portal_url(user_id: str) -> str:
    from src.api import auth

    rec = auth.member_record(user_id) or {}
    customer = (rec.get("billing") or {}).get("customer")
    if not customer:
        raise BillingError("支払い情報がありません")
    return str(sa.create_portal_session(customer_id=customer, return_url=config()["portal_return_url"])["url"])


def status_of(user_id: str) -> dict[str, Any]:
    from src.api import auth

    b = (auth.member_record(user_id) or {}).get("billing") or {}
    return {"managed": bool(b), "status": b.get("status"), "period_end": b.get("period_end"), "cancel_at_period_end": bool(b.get("cancel_at_period_end"))}


# ── Webhook ──
def _seen_ids() -> list[str]:
    try:
        return list(json.loads(EVENTS_PATH.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return []


def _mark_seen(event_id: str) -> None:
    ids = _seen_ids()
    ids.append(event_id)
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = EVENTS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(ids[-500:]), encoding="utf-8")
    os.replace(tmp, EVENTS_PATH)


def _period_end(sub: dict[str, Any]) -> int | None:
    if sub.get("current_period_end"):
        return int(sub["current_period_end"])
    items = ((sub.get("items") or {}).get("data")) or []        # 新しい API では購読アイテム側にある
    return int(items[0]["current_period_end"]) if items and items[0].get("current_period_end") else None


def handle_event(event: dict[str, Any]) -> str:
    """署名検証済みのイベントを反映する。同じイベントは二重に処理しない（"duplicate"）。対象が見つからなければ Unmatched（Stripe が再送する）。"""
    from src.scripts.ops import manage_users as mu

    eid = str(event.get("id") or "")
    if eid and eid in _seen_ids():
        return "duplicate"
    typ = event.get("type")
    obj = (event.get("data") or {}).get("object") or {}
    try:
        if typ == "checkout.session.completed":
            paid = obj.get("payment_status") in ("paid", "no_payment_required")
            mu.update_billing(idk=obj.get("client_reference_id"), customer=obj.get("customer"), subscription=obj.get("subscription"),
                              status="active" if paid else None)
        elif typ in ("customer.subscription.created", "customer.subscription.updated"):
            mu.update_billing(customer=obj.get("customer"), status=obj.get("status"), period_end=_period_end(obj),
                              cancel_at_period_end=bool(obj.get("cancel_at_period_end")), subscription=obj.get("id"))
        elif typ == "customer.subscription.deleted":
            mu.update_billing(customer=obj.get("customer"), status="canceled", period_end=_period_end(obj))
        elif typ == "invoice.payment_failed":
            mu.update_billing(customer=obj.get("customer"), status="past_due")
        else:
            return "ignored"
    except mu.UserError as e:
        raise Unmatched(str(e)) from None
    if eid:
        _mark_seen(eid)
    return "ok"
