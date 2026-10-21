"""VPS ハートビート: スキーマ・署名・評価・受信 API・VPS クライアントとの整合。外部接続なし。"""

import importlib.util
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.ops import heartbeat_api, heartbeat_schema as S, heartbeat_store as store

ROOT = Path(__file__).resolve().parents[2]
NOW = 1_800_000_000.0


def payload(avail=1400.0, swap=0.0, disk=40.0, units=None, checks=None, sent_at=NOW):
    return {"v": 1, "host": "vps", "sent_at": sent_at, "release": "keiba-ui-x",
            "mem": {"total_mb": 1980, "available_mb": avail, "used_mb": 380},
            "swap": {"total_mb": 1024, "used_mb": swap}, "disk": {"root_used_pct": disk}, "load": {"l1": 0.1, "l5": 0.1},
            "uptime_s": 1000, "units": units or {"nginx": {"active": "active", "restarts": 0, "mem_mb": 12}},
            "checks": checks or {"nextjs_http": 200, "api_upstream_http": 200}}


def feed(rec, avails, start=NOW, step=300, **kw):
    t = start
    for a in avails:
        rec = S.append_history(rec, S.validate_payload(payload(avail=a, **kw)), t)
        t += step
    return rec, t - step


# ── スキーマ ──
def test_validate_ok_and_drops_unknown_keys():
    p = payload()
    p["secret"] = "x"
    out = S.validate_payload(p)
    assert "secret" not in out and out["mem"]["available_mb"] == 1400.0


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(v=2), lambda p: p.update(host="a b"), lambda p: p["mem"].update(available_mb="x"),
    lambda p: p["mem"].update(available_mb=-1), lambda p: p["units"].update({"bad name!": {}}), lambda p: p.pop("mem"),
    lambda p: p["mem"].update(available_mb=True),
])
def test_validate_rejects(mutate):
    p = payload()
    mutate(p)
    with pytest.raises(S.SchemaError):
        S.validate_payload(p)


# ── 署名 ──
def test_signature():
    body = b'{"a":1}'
    sig = S.sign("k1", 1000, body)
    assert S.verify_signature([b"k1"], "1000", sig, body, now=1000) is None
    assert S.verify_signature([b"k0", b"k1"], "1000", sig, body, now=1000) is None   # 新旧 2 本
    assert S.verify_signature([b"k2"], "1000", sig, body, now=1000) == "bad_signature"
    assert S.verify_signature([b"k1"], "1000", sig, b'{"a":2}', now=1000) == "bad_signature"
    assert S.verify_signature([b"k1"], "1000", sig, body, now=1000 + 301) == "stale_ts"
    assert S.verify_signature([b"k1"], None, sig, body, now=1000) == "missing"
    assert S.verify_signature([], "1000", sig, body, now=1000) == "no_secret"


# ── 評価（architecture_target.md 3.2 の閾値） ──
def keys(alerts):
    return {a["key"]: a["severity"] for a in alerts}


def test_never_received_is_critical():
    assert keys(S.evaluate(S.new_record(), NOW)) == {"no_heartbeat": "critical"}


def test_stale_after_15_minutes_is_critical():
    rec, t = feed(S.new_record(), [1400.0])
    assert S.evaluate(rec, t + 14 * 60) == []
    assert keys(S.evaluate(rec, t + 15 * 60)) == {"stale": "critical"}


def test_memory_thresholds():
    rec, t = feed(S.new_record(), [900.0, 900.0])
    assert "mem_low" not in keys(S.evaluate(rec, t))               # 2 回連続ではまだ
    rec, t = feed(S.new_record(), [900.0, 900.0, 900.0])
    assert keys(S.evaluate(rec, t)) == {"mem_low": "warning"}      # 1GB 未満が 3 回連続
    rec, t = feed(S.new_record(), [900.0, 1500.0, 900.0])
    assert S.evaluate(rec, t) == []                                # 連続していない
    rec, t = feed(S.new_record(), [1400.0, 1400.0, 500.0])
    assert keys(S.evaluate(rec, t)) == {"mem_critical": "critical"}  # 600MB 未満は即時


def test_swap_disk_failed_units_checks():
    rec, t = feed(S.new_record(), [1400.0], swap=250.0, disk=85.0,
                  units={"nginx": {"active": "failed", "restarts": 0, "mem_mb": None}},
                  checks={"nextjs_http": 502, "api_upstream_http": 503})
    k = keys(S.evaluate(rec, t))
    assert k == {"swap_high": "warning", "disk_high": "warning", "unit_failed": "warning",
                 "check_nextjs_http": "critical", "check_api_upstream_http": "warning"}


def test_restart_burst_in_10_minutes():
    rec = S.new_record()
    t = NOW
    for n in (0, 2, 5):   # 10 分内に +5（> 3）
        rec = S.append_history(rec, S.validate_payload(payload(units={"keiba-ui": {"active": "active", "restarts": n, "mem_mb": 90}})), t)
        t += 240
    assert "restarts" in keys(S.evaluate(rec, t - 240))


def test_notification_dedup_and_recovery():
    a = [{"key": "swap_high", "severity": "warning", "title": "t", "detail": "d"}]
    send, rec, st = S.plan_notifications(a, {}, NOW)
    assert [x["key"] for x in send] == ["swap_high"] and rec == []
    send, rec, st2 = S.plan_notifications(a, st, NOW + 600)
    assert send == [] and st2["swap_high"]["first_seen"] == NOW           # 同じ異常は再送しない
    send, _, st3 = S.plan_notifications(a, st2, NOW + S.RENOTIFY_S + 1)
    assert len(send) == 1                                                 # 6 時間後は再送
    esc = [{**a[0], "severity": "critical"}]
    assert len(S.plan_notifications(esc, st, NOW + 10)[0]) == 1           # 重要度が上がったら即送る
    send, rec, st4 = S.plan_notifications([], st3, NOW + 9999)
    assert send == [] and rec == ["swap_high"] and st4 == {}              # 復旧


# ── 受信 API ──
@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("KEIBA_HEARTBEAT_SECRET", "sekret")
    monkeypatch.delenv("KEIBA_HEARTBEAT_SECRET_NEXT", raising=False)


def make_client(tmp_path, clock, sent):
    backend = store.FileBackend(tmp_path / "latest.json")
    app = FastAPI()
    app.include_router(heartbeat_api.build_router(lambda: backend, clock=lambda: clock[0],
                                                  notify=lambda title, lines, severity: sent.append((title, lines, severity))))
    return TestClient(app), backend


def post(c, body: dict, ts=NOW, secret="sekret", raw=None):
    b = raw if raw is not None else S.dumps(body)
    return c.post("/ops/heartbeat", content=b, headers={S.HEADER_TS: str(int(ts)), S.HEADER_SIG: S.sign(secret, int(ts), b)})


def test_receive_status_and_check_flow(tmp_path, env):
    clock, sent = [NOW], []
    c, backend = make_client(tmp_path, clock, sent)
    assert c.get("/ops/heartbeat/status").json()["status"] == "never"
    r = post(c, payload())
    assert r.status_code == 200
    s = c.get("/ops/heartbeat/status").json()
    assert s["status"] == "ok" and s["alerts"] == [] and s["latest"]["mem"]["available_mb"] == 1400.0
    assert "secret" not in json.dumps(s).lower()
    assert json.loads((tmp_path / "latest.json").read_text())["received_at"] == NOW
    # 途絶 → check が緊急通知を 1 回だけ出し、復旧で解消通知
    clock[0] = NOW + 16 * 60
    assert c.post("/ops/heartbeat/check").json()["sent"] == ["stale"]
    assert c.post("/ops/heartbeat/check").json()["sent"] == []
    assert sent[0][2] == "critical" and "途絶" in sent[0][0]
    clock[0] = NOW + 17 * 60
    assert post(c, payload(), ts=clock[0]).status_code == 200
    assert c.post("/ops/heartbeat/check").json()["recovered"] == ["stale"]
    assert sent[-1][2] == "normal"


def test_receive_rejects_bad_signature_stale_ts_and_bad_payload(tmp_path, env):
    clock, sent = [NOW], []
    c, _ = make_client(tmp_path, clock, sent)
    assert post(c, payload(), secret="wrong").status_code == 401
    assert post(c, payload(), ts=NOW - 1000).status_code == 401
    assert post(c, {"v": 9}).status_code == 400
    assert post(c, None, raw=b"x" * (S.MAX_BODY_BYTES + 1)).status_code == 413
    assert c.get("/ops/heartbeat/status").json()["status"] == "never"      # 何も保存されていない


def test_receive_without_secret_configured_is_401(tmp_path, monkeypatch):
    monkeypatch.delenv("KEIBA_HEARTBEAT_SECRET", raising=False)
    monkeypatch.delenv("KEIBA_HEARTBEAT_SECRET_NEXT", raising=False)
    c, _ = make_client(tmp_path, [NOW], [])
    assert post(c, payload()).json()["reason"] == "no_secret"


def test_secret_rotation_next_accepted(tmp_path, env, monkeypatch):
    monkeypatch.setenv("KEIBA_HEARTBEAT_SECRET_NEXT", "newer")
    c, _ = make_client(tmp_path, [NOW], [])
    assert post(c, payload(), secret="newer").status_code == 200


def test_store_retries_once_on_conflict():
    class Flaky:
        def __init__(self):
            self.n = 0

        def read(self):
            return None, 0

        def write(self, data, token):
            self.n += 1
            return self.n > 1

    b = Flaky()
    assert store.update(b, lambda r: {**r, "x": 1})["x"] == 1 and b.n == 2


# ── VPS クライアントとの整合 ──
def _load_client():
    spec = importlib.util.spec_from_file_location("hb_client", ROOT / "scripts/ops/vps/heartbeat_client.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_vps_client_payload_matches_schema_and_signature(monkeypatch):
    m = _load_client()
    monkeypatch.setattr(m, "http_code", lambda url, timeout=8.0: 200)
    monkeypatch.setattr(m, "unit_state", lambda name: {"active": "active", "restarts": 1, "mem_mb": 20.5})
    p = m.collect({"APP_PORT": "3000", "PROXY_PORT": "8080", "APP_ROOT": "/nonexistent"})
    out = S.validate_payload(p)                      # クライアントの出力が GCP 側の検証を通る
    assert set(out["units"]) == set(m.UNITS)
    body = m.encode(p)
    assert body == S.dumps(p)                        # 署名対象のバイト列が両方で同じ
    assert m.sign("k", 1234, body) == S.sign("k", 1234, body)
