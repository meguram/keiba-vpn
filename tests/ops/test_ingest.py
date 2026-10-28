"""VPS からの定期送信の受信（ハートビート・アクセス集計）と点検通知。外部接続なし。"""

from __future__ import annotations

import json

import pytest

from src.ops import heartbeat_schema as S, heartbeat_store as store, ingest
from src.ops.gcs_io import MemoryIO
from tests.ops.test_heartbeat import NOW, payload

SECRET = b"k1"


def signed(body: bytes, secret=SECRET, ts: float = NOW) -> dict:
    return {S.HEADER_TS: str(int(ts)), S.HEADER_SIG: S.sign(secret, int(ts), body)}


def traffic_body(**over) -> bytes:
    d = {"v": 1, "sent_at": int(NOW), "readable": True, "reason": None, "files": 2, "lines": 10, "windows": {"today": 3, "7d": 5, "x": 9},
         "days": {"20261028": {"requests": 10, "page_views": 4, "visitors": 3, "status": {"2xx": 9}, "pages": [["/race/1", 3], ["/", 1]],
                               "hours": [0] * 24, "evil": "x"}, "bad-key": {"requests": 1}}}
    d.update(over)
    return json.dumps(d).encode()


# ── ハートビート ──
def test_receive_heartbeat_accepts_signed_and_updates_store(tmp_path):
    backend = store.FileBackend(tmp_path / "latest.json")
    body = json.dumps(payload()).encode()
    code, res = ingest.receive_heartbeat(signed(body), body, backend, sec=[SECRET], now=NOW)
    assert code == 200 and res["ok"]
    rec, _ = backend.read()
    assert rec["latest"]["mem"]["available_mb"] == 1400.0


@pytest.mark.parametrize("hdr,expect", [
    (lambda b: {}, "missing"),
    (lambda b: signed(b, secret=b"wrong"), "bad_signature"),
    (lambda b: signed(b, ts=NOW - 3600), "stale_ts"),
])
def test_receive_heartbeat_rejects(tmp_path, hdr, expect):
    backend = store.FileBackend(tmp_path / "latest.json")
    body = json.dumps(payload()).encode()
    code, res = ingest.receive_heartbeat(hdr(body), body, backend, sec=[SECRET], now=NOW)
    assert code == 401 and res["reason"] == expect and backend.read()[0] is None


def test_receive_heartbeat_bad_payload_and_size(tmp_path):
    backend = store.FileBackend(tmp_path / "latest.json")
    bad = b'{"v": 2}'
    assert ingest.receive_heartbeat(signed(bad), bad, backend, sec=[SECRET], now=NOW)[0] == 400
    big = b"x" * (S.MAX_BODY_BYTES + 1)
    assert ingest.receive_heartbeat(signed(big), big, backend, sec=[SECRET], now=NOW)[0] == 413


def test_secrets_reads_current_and_next():
    assert ingest.secrets({"KEIBA_HEARTBEAT_SECRET": " a ", "KEIBA_HEARTBEAT_SECRET_NEXT": "b"}) == [b"a", b"b"]
    assert ingest.secrets({}) == []


# ── 点検 ──
def test_run_check_notifies_once_then_recovers(tmp_path):
    backend = store.FileBackend(tmp_path / "latest.json")
    body = json.dumps(payload()).encode()
    ingest.receive_heartbeat(signed(body), body, backend, sec=[SECRET], now=NOW)
    sent: list = []
    notify = lambda title, lines, sev: sent.append((title, sev))  # noqa: E731
    late = NOW + S.STALE_CRITICAL_S + 60
    r1 = ingest.run_check(backend, notify, now=late)
    assert "stale" in " ".join(r1["sent"]) or r1["sent"], r1
    n = len(sent)
    assert n >= 1 and sent[0][1] == "critical"
    assert ingest.run_check(backend, notify, now=late + 60)["sent"] == []  # 同じ異常は再通知しない
    assert len(sent) == n
    ingest.receive_heartbeat(signed(body, ts=late + 120), body, backend, sec=[SECRET], now=late + 120)
    r3 = ingest.run_check(backend, notify, now=late + 130)
    assert r3["recovered"] and sent[-1][1] == "normal"


# ── アクセス集計 ──
def test_receive_traffic_stores_sanitized_bundle():
    io = MemoryIO()
    body = traffic_body()
    code, res = ingest.receive_traffic(signed(body), body, io, sec=[SECRET], now=NOW)
    assert code == 200 and res["days"] == 1
    b = json.loads(io.get_bytes(ingest.TRAFFIC_OBJECT))
    assert b["received_at"] == NOW and set(b["days"]) == {"20261028"}   # 日付でないキーは捨てる
    assert "evil" not in b["days"]["20261028"] and b["windows"] == {"today": 3, "7d": 5}
    assert b["days"]["20261028"]["pages"][0] == ["/race/1", 3]


def test_receive_traffic_rejects_bad_signature_size_and_shape():
    io = MemoryIO()
    body = traffic_body()
    assert ingest.receive_traffic(signed(body, secret=b"no"), body, io, sec=[SECRET], now=NOW)[0] == 401
    assert ingest.receive_traffic({}, body, io, sec=[SECRET], now=NOW)[0] == 401
    assert ingest.receive_traffic(signed(body), body, io, sec=[], now=NOW)[0] == 401   # 秘密が未設定なら誰も入れない
    big = b"x" * (ingest.MAX_TRAFFIC_BYTES + 1)
    assert ingest.receive_traffic(signed(big), big, io, sec=[SECRET], now=NOW)[0] == 413
    wrong = json.dumps({"v": 9}).encode()
    assert ingest.receive_traffic(signed(wrong), wrong, io, sec=[SECRET], now=NOW)[0] == 400
    assert io.get_bytes(ingest.TRAFFIC_OBJECT) is None


def test_sanitize_day_clamps_and_limits():
    d = ingest.sanitize_day({"requests": -5, "page_views": "x", "pages": [["/a", 1]] * 500, "status": {"2xx": 1, "k" * 200: 2}, "hours": [1, 2], "bots": "no"})
    assert d["requests"] == 0 and d["page_views"] == 0 and len(d["pages"]) == 60 and "hours" not in d and "bots" not in d
    assert all(len(k) <= 80 for k in d["status"])
    assert ingest.sanitize_day("nope") == {}
