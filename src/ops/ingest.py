"""VPS からの定期送信（ハートビート・アクセス集計）の受信と、異常の判定通知。フレームワークに依存しない純粋な部分。

受け先は監視ポータル（``src/monitor/cloud_app.py``、Cloud Run の keiba-monitor。リクエスト課金・最大 1）。**keiba-api には送らない**:
API は『CPU を常に割り当て』（インスタンス課金）でデプロイ済みで、15 分以内の間隔で ping が来ると最小 0 でも落ちず、月 約 7,800 円になる
（docs/operations/cost_recalc_20261028.md）。

- ``receive_heartbeat``  VPS が 5 分おきに送る。HMAC 署名で認証（共有ヘッダ鍵は不要）。``ops/heartbeat/latest.json`` を更新。
- ``run_check``          Scheduler が 15 分おきに呼ぶ（共有ヘッダ鍵で保護）。評価して、新しい異常・復旧だけを通知する。
- ``receive_traffic``    VPS が 15 分おきに送るアクセス集計（日別）。HMAC 署名で認証。``ops/traffic/bundle.json`` に置く。
署名の秘密は ``KEIBA_HEARTBEAT_SECRET``（新旧 2 本: ``KEIBA_HEARTBEAT_SECRET_NEXT``）。アクセス集計にも同じ秘密を使う。
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Callable

from src.ops import heartbeat_schema as S
from src.ops import heartbeat_store as store

TRAFFIC_OBJECT = "ops/traffic/bundle.json"
MAX_TRAFFIC_BYTES = 256 * 1024
MAX_DAYS = 130
_DAY = re.compile(r"^\d{8}$")
_INT_FIELDS = ("requests", "page_views", "rsc", "api", "assets", "visitors", "internal")
_MAP_FIELDS = ("status", "bot_kind", "bot_status", "referrer_kind", "devices", "seo_files")
_LIST_FIELDS = {"bots": 20, "pages": 60, "patterns": 30, "referrers": 30, "err_pages": 30, "err5xx": 20}


def secrets(env: dict[str, str] | None = None) -> list[bytes]:
    e = os.environ if env is None else env
    return [v.strip().encode() for n in ("KEIBA_HEARTBEAT_SECRET", "KEIBA_HEARTBEAT_SECRET_NEXT") if (v := e.get(n) or "").strip()]


def _auth(headers: Any, body: bytes, sec: list[bytes], now: float) -> str | None:
    return S.verify_signature(sec, headers.get(S.HEADER_TS), headers.get(S.HEADER_SIG), body, now=now)


def receive_heartbeat(headers: Any, body: bytes, backend: store.Backend, *, sec: list[bytes] | None = None, now: float | None = None) -> tuple[int, dict]:
    now = time.time() if now is None else now
    if len(body) > S.MAX_BODY_BYTES:
        return 413, {"error": "too_large"}
    why = _auth(headers, body, secrets() if sec is None else sec, now)
    if why:
        return 401, {"error": "unauthorized", "reason": why}
    try:
        payload = S.validate_payload(json.loads(body))
    except (ValueError, TypeError) as exc:
        return 400, {"error": "bad_payload", "detail": str(exc)[:200]}
    rec = store.update(backend, lambda r: S.append_history(r, payload, now))
    return 200, {"ok": True, "received_at": rec["received_at"]}


def run_check(backend: store.Backend, notify: Callable[..., Any], *, now: float | None = None) -> dict:
    """評価して、新しい異常・復旧だけを通知する（同じ異常は RENOTIFY_S まで再通知しない）。"""
    now = time.time() if now is None else now
    sent: list[dict] = []
    recovered: list[str] = []

    def _fn(rec: dict) -> dict:
        alerts = S.evaluate(rec, now)
        to_send, rec_keys, new_state = S.plan_notifications(alerts, rec.get("alert_state") or {}, now)
        sent.extend(to_send)
        recovered.extend(rec_keys)
        return {**rec, "alert_state": new_state}

    store.update(backend, _fn)
    for a in sent:
        notify(a["title"], [a["detail"]] if a["detail"] else [], a["severity"])
    if recovered:
        notify("VPS の異常が解消しました", [", ".join(recovered)], "normal")
    return {"ok": True, "sent": [a["key"] for a in sent], "recovered": recovered}


# ───────────────────────── アクセス集計 ─────────────────────────

def _int(v: Any, hi: int = 10**9) -> int:
    try:
        return max(0, min(int(v), hi))
    except (TypeError, ValueError):
        return 0


def sanitize_day(d: Any) -> dict:
    """1 日分を既知の形だけに絞る（署名済みでも、壊れた値・巨大な値で画面を壊さない）。"""
    if not isinstance(d, dict):
        return {}
    out: dict[str, Any] = {k: _int(d.get(k)) for k in _INT_FIELDS if k in d}
    for k in _MAP_FIELDS:
        if isinstance(d.get(k), dict):
            out[k] = {str(a)[:80]: _int(b) for a, b in list(d[k].items())[:40]}
    for k, n in _LIST_FIELDS.items():
        if isinstance(d.get(k), list):
            out[k] = [[str(x[0])[:200], _int(x[1])] for x in d[k][:n] if isinstance(x, (list, tuple)) and len(x) == 2]
    if isinstance(d.get("hours"), list) and len(d["hours"]) == 24:
        out["hours"] = [_int(h) for h in d["hours"]]
    return out


def validate_traffic(d: Any) -> dict:
    """検証して正規化した dict を返す。不正は ``S.SchemaError``。"""
    if not isinstance(d, dict) or d.get("v") != 1:
        raise S.SchemaError("v=1 の JSON オブジェクトが必要です")
    days = d.get("days")
    if not isinstance(days, dict) or len(days) > MAX_DAYS:
        raise S.SchemaError("days が不正です")
    out_days = {k: sanitize_day(v) for k, v in days.items() if _DAY.match(str(k))}
    windows = d.get("windows") if isinstance(d.get("windows"), dict) else {}
    return {"v": 1, "sent_at": _int(d.get("sent_at")), "readable": bool(d.get("readable")), "reason": str(d.get("reason") or "")[:300] or None,
            "files": _int(d.get("files")), "lines": _int(d.get("lines")),
            "windows": {k: _int(windows.get(k)) for k in ("today", "yesterday", "7d", "14d") if k in windows}, "days": out_days}


def receive_traffic(headers: Any, body: bytes, io: Any, *, sec: list[bytes] | None = None, now: float | None = None) -> tuple[int, dict]:
    now = time.time() if now is None else now
    if len(body) > MAX_TRAFFIC_BYTES:
        return 413, {"error": "too_large"}
    why = _auth(headers, body, secrets() if sec is None else sec, now)
    if why:
        return 401, {"error": "unauthorized", "reason": why}
    try:
        bundle = validate_traffic(json.loads(body))
    except (ValueError, TypeError) as exc:
        return 400, {"error": "bad_payload", "detail": str(exc)[:200]}
    bundle["received_at"] = now
    io.put_bytes(TRAFFIC_OBJECT, json.dumps(bundle, ensure_ascii=False).encode("utf-8"), content_type="application/json")
    return 200, {"ok": True, "days": len(bundle["days"]), "received_at": now}
