"""VPS ハートビートの JSON スキーマ・署名・評価（純粋関数。外部接続なし）。

VPS（systemd タイマー）が 5 分おきに採取して送る ``payload`` を定義し、GCP 側（Cloud Run の /ops/heartbeat）が検証・評価する。
VPS 側の送信スクリプト（scripts/ops/vps/heartbeat_client.py）は標準ライブラリだけで同じ形を作る（テストが一致を確認する）。
閾値の根拠: docs/operations/architecture_target.md 3.2「監視」。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import time
from typing import Any

SCHEMA_VERSION = 1
MAX_BODY_BYTES = 16 * 1024
SIGNATURE_TOLERANCE_S = 300
HEADER_TS = "x-keiba-heartbeat-ts"
HEADER_SIG = "x-keiba-heartbeat-sig"

# 評価の閾値（architecture_target.md 3.2）
STALE_CRITICAL_S = 15 * 60
MEM_WARN_MB = 1024
MEM_WARN_CONSECUTIVE = 3
MEM_CRIT_MB = 600
SWAP_WARN_MB = 200
DISK_WARN_PCT = 80.0
RESTART_WARN_COUNT = 3
RESTART_WINDOW_S = 10 * 60
HISTORY_MAX = 12
RENOTIFY_S = 6 * 3600

_UNIT_RE = re.compile(r"^[A-Za-z0-9_.@:-]{1,64}$")
_ACTIVE_STATES = {"active", "inactive", "failed", "activating", "deactivating", "reloading", "unknown"}


class SchemaError(ValueError):
    pass


def _num(d: dict, key: str, lo: float, hi: float, *, required: bool = True) -> float | None:
    v = d.get(key)
    if v is None:
        if required:
            raise SchemaError(f"{key} がありません")
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise SchemaError(f"{key} は数値で指定してください")
    if not (lo <= float(v) <= hi):
        raise SchemaError(f"{key} が範囲外です ({lo}..{hi})")
    return float(v)


def validate_payload(d: Any) -> dict:
    """検証して正規化した dict を返す（未知のキーは捨てる）。不正は SchemaError。

    形式（v=1）::

        {"v":1, "host":"vps", "sent_at":1760000000, "release":"keiba-ui-...",
         "mem":{"total_mb":1980,"available_mb":1400,"used_mb":380},
         "swap":{"total_mb":1024,"used_mb":0},
         "disk":{"root_used_pct":41.5}, "load":{"l1":0.1,"l5":0.1}, "uptime_s":123456,
         "units":{"nginx":{"active":"active","restarts":0,"mem_mb":12}, ...},
         "checks":{"nextjs_http":200,"nginx_http":200}}
    """
    if not isinstance(d, dict):
        raise SchemaError("JSON オブジェクトではありません")
    if d.get("v") != SCHEMA_VERSION:
        raise SchemaError(f"v={SCHEMA_VERSION} のみ対応しています")
    host = d.get("host")
    if not isinstance(host, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,32}", host):
        raise SchemaError("host が不正です")
    sent_at = _num(d, "sent_at", 1_000_000_000, 4_000_000_000)
    mem = d.get("mem")
    if not isinstance(mem, dict):
        raise SchemaError("mem がありません")
    swap = d.get("swap") if isinstance(d.get("swap"), dict) else {}
    disk = d.get("disk") if isinstance(d.get("disk"), dict) else {}
    load = d.get("load") if isinstance(d.get("load"), dict) else {}
    out: dict[str, Any] = {
        "v": SCHEMA_VERSION, "host": host, "sent_at": sent_at,
        "release": str(d.get("release") or "")[:80],
        "mem": {
            "total_mb": _num(mem, "total_mb", 1, 1_000_000),
            "available_mb": _num(mem, "available_mb", 0, 1_000_000),
            "used_mb": _num(mem, "used_mb", 0, 1_000_000, required=False),
        },
        "swap": {"total_mb": _num(swap, "total_mb", 0, 1_000_000, required=False) or 0.0,
                 "used_mb": _num(swap, "used_mb", 0, 1_000_000, required=False) or 0.0},
        "disk": {"root_used_pct": _num(disk, "root_used_pct", 0, 100, required=False)},
        "load": {"l1": _num(load, "l1", 0, 10_000, required=False), "l5": _num(load, "l5", 0, 10_000, required=False)},
        "uptime_s": _num(d, "uptime_s", 0, 10**10, required=False),
        "units": {}, "checks": {},
    }
    units = d.get("units") or {}
    if not isinstance(units, dict) or len(units) > 32:
        raise SchemaError("units が不正です")
    for name, u in units.items():
        if not isinstance(name, str) or not _UNIT_RE.match(name) or not isinstance(u, dict):
            raise SchemaError("units の要素が不正です")
        active = str(u.get("active") or "unknown")
        out["units"][name] = {
            "active": active if active in _ACTIVE_STATES else "unknown",
            "restarts": int(_num(u, "restarts", 0, 10**9, required=False) or 0),
            "mem_mb": _num(u, "mem_mb", 0, 1_000_000, required=False),
        }
    checks = d.get("checks") or {}
    if not isinstance(checks, dict) or len(checks) > 16:
        raise SchemaError("checks が不正です")
    for name, v in checks.items():
        if not isinstance(name, str) or not _UNIT_RE.match(name) or isinstance(v, bool) or not isinstance(v, (int, float)):
            raise SchemaError("checks の要素が不正です")
        out["checks"][name] = int(v)
    return out


# ───────────────────────── 署名 ─────────────────────────

def sign(secret: bytes | str, ts: int | str, body: bytes) -> str:
    k = secret.encode("utf-8") if isinstance(secret, str) else secret
    return hmac.new(k, f"{ts}.".encode() + body, hashlib.sha256).hexdigest()


def verify_signature(secrets: list[bytes], ts: str | None, sig: str | None, body: bytes, *, now: float | None = None,
                     tolerance_s: int = SIGNATURE_TOLERANCE_S) -> str | None:
    """通れば None、拒否理由なら文字列（'no_secret' | 'missing' | 'stale_ts' | 'bad_signature'）。"""
    if not secrets:
        return "no_secret"
    if not ts or not sig:
        return "missing"
    try:
        t = int(ts)
    except ValueError:
        return "missing"
    if abs((time.time() if now is None else now) - t) > tolerance_s:
        return "stale_ts"
    ok = False
    for k in secrets:
        ok |= hmac.compare_digest(sign(k, t, body), sig.strip().lower())
    return None if ok else "bad_signature"


# ───────────────────────── 履歴と評価 ─────────────────────────

def new_record() -> dict:
    return {"latest": None, "received_at": None, "history": [], "alert_state": {}}


def append_history(record: dict, payload: dict, received_at: float) -> dict:
    """履歴（直近 HISTORY_MAX 件）に今回の値を足した新しい record を返す。"""
    hist = list(record.get("history") or [])
    hist.append({"t": received_at, "avail_mb": payload["mem"]["available_mb"], "swap_mb": payload["swap"]["used_mb"],
                 "restarts": {n: u["restarts"] for n, u in payload["units"].items()}})
    return {**record, "latest": payload, "received_at": received_at, "history": hist[-HISTORY_MAX:]}


def evaluate(record: dict, now: float) -> list[dict]:
    """現在の異常の一覧 ``[{"key","severity","title","detail"}]``。severity は warning | critical。"""
    alerts: list[dict] = []
    rec_at = record.get("received_at")
    latest = record.get("latest")
    if latest is None or rec_at is None:
        return [{"key": "no_heartbeat", "severity": "critical", "title": "VPS のハートビートが一度も届いていません", "detail": ""}]
    age = now - float(rec_at)
    if age >= STALE_CRITICAL_S:
        alerts.append({"key": "stale", "severity": "critical", "title": "VPS のハートビートが途絶えています",
                       "detail": f"最終受信から {int(age // 60)} 分（VPS・トンネル・ネットワークの停止の疑い）"})
        return alerts  # 古い値で他を評価しない
    avail = float(latest["mem"]["available_mb"])
    hist = record.get("history") or []
    if avail < MEM_CRIT_MB:
        alerts.append({"key": "mem_critical", "severity": "critical", "title": "VPS の空きメモリが非常に少ない",
                       "detail": f"MemAvailable {avail:.0f}MB（緊急の閾値 {MEM_CRIT_MB}MB 未満）"})
    else:
        last = [h.get("avail_mb") for h in hist[-MEM_WARN_CONSECUTIVE:]]
        if len(last) >= MEM_WARN_CONSECUTIVE and all(isinstance(x, (int, float)) and x < MEM_WARN_MB for x in last):
            alerts.append({"key": "mem_low", "severity": "warning", "title": "VPS の空きメモリが少ない状態が続いています",
                           "detail": f"MemAvailable {avail:.0f}MB（{MEM_WARN_MB}MB 未満が {MEM_WARN_CONSECUTIVE} 回連続）"})
    swap = float(latest["swap"]["used_mb"])
    if swap > SWAP_WARN_MB:
        alerts.append({"key": "swap_high", "severity": "warning", "title": "VPS の swap 使用量が多い", "detail": f"swap {swap:.0f}MB（{SWAP_WARN_MB}MB 超）"})
    disk = latest["disk"].get("root_used_pct")
    if disk is not None and disk > DISK_WARN_PCT:
        alerts.append({"key": "disk_high", "severity": "warning", "title": "VPS のディスク使用率が高い", "detail": f"{disk:.0f}%（{DISK_WARN_PCT:.0f}% 超）"})
    failed = sorted(n for n, u in latest["units"].items() if u["active"] == "failed")
    if failed:
        alerts.append({"key": "unit_failed", "severity": "warning", "title": "VPS の systemd ユニットが failed", "detail": ", ".join(failed)})
    # 再起動の多発: 直近 10 分の履歴で、累積カウンタの増加が閾値を超えたユニット
    recent = [h for h in hist if now - float(h["t"]) <= RESTART_WINDOW_S]
    if len(recent) >= 2:
        first, last_ = recent[0].get("restarts") or {}, recent[-1].get("restarts") or {}
        bad = sorted(n for n, v in last_.items() if v - first.get(n, v) > RESTART_WARN_COUNT)
        if bad:
            alerts.append({"key": "restarts", "severity": "warning", "title": "VPS のユニットが再起動を繰り返しています",
                           "detail": ", ".join(f"{n}(+{last_[n] - first.get(n, last_[n])})" for n in bad)})
    for name, code in sorted(latest["checks"].items()):
        if code != 200:
            alerts.append({"key": f"check_{name}", "severity": "critical" if name.startswith("nextjs") else "warning",
                           "title": f"VPS の確認 {name} が正常ではありません", "detail": f"HTTP {code}"})
    return alerts


def plan_notifications(alerts: list[dict], alert_state: dict, now: float) -> tuple[list[dict], list[str], dict]:
    """(送る異常, 復旧した key, 新しい alert_state)。同じ異常は RENOTIFY_S 経つまで再送しない。"""
    to_send: list[dict] = []
    new_state: dict[str, Any] = {}
    active_keys = {a["key"] for a in alerts}
    for a in alerts:
        prev = alert_state.get(a["key"])
        if prev is None or now - float(prev.get("last_sent", 0)) >= RENOTIFY_S or prev.get("severity") != a["severity"]:
            to_send.append(a)
            new_state[a["key"]] = {"first_seen": (prev or {}).get("first_seen", now), "last_sent": now, "severity": a["severity"]}
        else:
            new_state[a["key"]] = prev
    recovered = sorted(k for k in alert_state if k not in active_keys)
    return to_send, recovered, new_state


def dumps(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
