"""VPS ハートビートの受信 API（FastAPI ルーター）。Cloud Run のゲートウェイ（src/api/cloudrun_entry.py）に載せる。

- ``POST /ops/heartbeat``        VPS が 5 分おきに送る。HMAC 署名（X-Keiba-Heartbeat-Ts / -Sig）で認証。共有ヘッダ鍵は不要。
- ``GET  /ops/heartbeat/status`` 鮮度と現在の異常を返す（共有ヘッダ鍵で保護される）。秘密は含まない。
- ``POST /ops/heartbeat/check``  Cloud Scheduler が呼ぶ。評価して、新しい異常・復旧を Slack に通知する（共有ヘッダ鍵で保護）。

署名の秘密は ``KEIBA_HEARTBEAT_SECRET``（新旧 2 本: ``KEIBA_HEARTBEAT_SECRET_NEXT``）。VPS は Slack の秘密を持たない。
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.ops import heartbeat_schema as S
from src.ops import heartbeat_store as store

logger = logging.getLogger("ops.heartbeat")


def _secrets(env: dict[str, str] | None = None) -> list[bytes]:
    e = os.environ if env is None else env
    return [v.strip().encode() for n in ("KEIBA_HEARTBEAT_SECRET", "KEIBA_HEARTBEAT_SECRET_NEXT") if (v := e.get(n) or "").strip()]


def status_view(record: dict, now: float) -> dict:
    alerts = S.evaluate(record, now)
    rec_at = record.get("received_at")
    latest = record.get("latest") or {}
    state = "never" if rec_at is None else ("stale" if now - rec_at >= S.STALE_CRITICAL_S else "ok")
    return {
        "status": state,
        "age_s": None if rec_at is None else round(now - rec_at),
        "received_at": rec_at,
        "alerts": alerts,
        "latest": {
            "host": latest.get("host"), "release": latest.get("release"), "mem": latest.get("mem"), "swap": latest.get("swap"),
            "disk": latest.get("disk"), "load": latest.get("load"), "uptime_s": latest.get("uptime_s"),
            "units": latest.get("units"), "checks": latest.get("checks"),
        } if latest else None,
    }


def build_router(backend_factory: Callable[[], store.Backend] | None = None, *, clock: Callable[[], float] = time.time,
                 notify: Callable[..., Any] | None = None) -> APIRouter:
    """backend_factory / clock / notify はテストで差し替える。"""
    router = APIRouter()
    factory = backend_factory or store.default_backend

    def _notify(title: str, lines: list[str], severity: str) -> None:
        if notify is not None:
            notify(title, lines, severity=severity)
            return
        from src.utils.slack_notify import notify_event

        notify_event(title, lines, severity=severity, channel="alert")

    @router.post("/ops/heartbeat")
    async def receive(request: Request):
        body = await request.body()
        if len(body) > S.MAX_BODY_BYTES:
            return JSONResponse({"error": "too_large"}, status_code=413)
        why = S.verify_signature(_secrets(), request.headers.get(S.HEADER_TS), request.headers.get(S.HEADER_SIG), body, now=clock())
        if why:
            logger.warning("ハートビートの署名を拒否: %s", why)
            return JSONResponse({"error": "unauthorized", "reason": why}, status_code=401)
        try:
            payload = S.validate_payload(json.loads(body))
        except (ValueError, TypeError) as exc:
            return JSONResponse({"error": "bad_payload", "detail": str(exc)[:200]}, status_code=400)
        now = clock()
        rec = store.update(factory(), lambda r: S.append_history(r, payload, now))
        return JSONResponse({"ok": True, "received_at": rec["received_at"]})

    @router.get("/ops/heartbeat/status")
    async def status():
        rec, _ = factory().read()
        return JSONResponse(status_view(rec or S.new_record(), clock()), headers={"Cache-Control": "no-store"})

    @router.post("/ops/heartbeat/check")
    async def check():
        now = clock()
        sent: list[dict] = []
        recovered: list[str] = []

        def _fn(rec: dict) -> dict:
            alerts = S.evaluate(rec, now)
            to_send, rec_keys, new_state = S.plan_notifications(alerts, rec.get("alert_state") or {}, now)
            sent.extend(to_send)
            recovered.extend(rec_keys)
            return {**rec, "alert_state": new_state}

        store.update(factory(), _fn)
        for a in sent:
            _notify(a["title"], [a["detail"]] if a["detail"] else [], a["severity"])
        if recovered:
            _notify("VPS の異常が解消しました", [", ".join(recovered)], "normal")
        return JSONResponse({"ok": True, "sent": [a["key"] for a in sent], "recovered": recovered})

    return router
