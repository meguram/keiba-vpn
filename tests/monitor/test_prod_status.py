"""prod（VPS）の状況: 収集アダプタ（fake runner）・全体の信号・キャッシュの読み出し・画面。外部接続なし。"""
from __future__ import annotations

import json
import time

import pytest

from src.monitor import prod_status as P
from src.monitor import app as monitor_app

PROBE = {
    "probe_version": 1, "at": "2026-10-22T10:00:00+0900", "host": "vps",
    "mem": {"total_mb": 1962.0, "available_mb": 1413.0, "used_mb": 548.0, "swap_total_mb": 2048.0, "swap_used_mb": 2.0},
    "load": [0.0, 0.0, 0.0], "uptime_s": 5000.0, "disk": {"used_pct": 23, "avail_mb": 74000},
    "units": {"keiba-ui": {"installed": True, "active": "active", "sub": "running", "restarts": 0, "memory_mb": 70.0},
              "nginx": {"installed": True, "active": "active", "sub": "running", "restarts": 0, "memory_mb": 9.0},
              "cloudflared-prod": {"installed": True, "active": "active", "sub": "running", "restarts": 0, "memory_mb": 17.0},
              "keiba-fetch-agent": {"installed": False}},
    "ui": {"code": 200, "ms": 300, "json": {"release": "keiba-ui-abc", "env": "prod", "commit": "abcdef0123", "source_digest": "deadbeef"}},
    "upstream_api": {"code": 200, "ms": 340}, "current_release": "keiba-ui-abc", "releases": ["a", "b"],
    "health_timer_state": {"ok": True}, "tunnel": {"readable": False},
}


def fake_run(cmd, input_text=None, timeout=30.0):
    if cmd and cmd[0] == "ssh":
        return 0, json.dumps(PROBE) + "\n", ""
    return 1, "", "not available"


def test_vps_adapter_parses_probe_and_survives_failures():
    sec = P.vps_adapter(fake_run)
    assert sec["state"] == "ok" and sec["data"]["mem"]["available_mb"] == 1413.0
    bad = P.vps_adapter(lambda c, i=None, t=1.0: (255, "", "Connection timed out"))
    assert bad["state"] == "error" and "ssh" in bad["reason"]
    garbage = P.vps_adapter(lambda c, i=None, t=1.0: (0, "not json", ""))
    assert garbage["state"] == "error"


def test_overall_green_and_thresholds():
    sections = {"vps": P.ok(PROBE)}
    assert P.evaluate_overall(sections)["level"] == "green"
    low = json.loads(json.dumps(PROBE)); low["mem"]["available_mb"] = 900.0
    assert P.evaluate_overall({"vps": P.ok(low)})["level"] == "yellow"
    low["mem"]["available_mb"] = 500.0
    assert P.evaluate_overall({"vps": P.ok(low)})["level"] == "red"
    high_used = json.loads(json.dumps(PROBE)); high_used["mem"]["used_mb"] = 700.0
    assert P.evaluate_overall({"vps": P.ok(high_used)})["level"] == "yellow"
    down = json.loads(json.dumps(PROBE)); down["units"]["nginx"]["active"] = "failed"
    assert P.evaluate_overall({"vps": P.ok(down)})["level"] == "red"
    noui = json.loads(json.dumps(PROBE)); noui["ui"]["code"] = 502
    assert P.evaluate_overall({"vps": P.ok(noui)})["level"] == "red"
    noapi = json.loads(json.dumps(PROBE)); noapi["upstream_api"]["code"] = 403
    assert P.evaluate_overall({"vps": P.ok(noapi)})["level"] == "red"
    restarts = json.loads(json.dumps(PROBE)); restarts["units"]["keiba-ui"]["restarts"] = 266
    assert P.evaluate_overall({"vps": P.ok(restarts)})["level"] == "green"        # 累計だけでは警告しない（過去の失敗が残る）
    restarts["units"]["keiba-ui"]["restart_delta"] = 1
    assert P.evaluate_overall({"vps": P.ok(restarts)})["level"] == "yellow"
    restarts["units"]["keiba-ui"]["restart_delta"] = 5
    assert P.evaluate_overall({"vps": P.ok(restarts)})["level"] == "red"


def test_overall_when_vps_unavailable_is_not_red():
    ov = P.evaluate_overall({"vps": P.error("ssh できない")})
    assert ov["level"] == "yellow" and "取得できていません" in ov["reasons"][0]["message"]


def test_missing_heartbeat_and_shadow_are_not_errors():
    def run(cmd, input_text=None, timeout=30.0):
        if cmd and cmd[0] == "ssh":
            return 0, json.dumps(PROBE), ""
        return 1, "", "ERROR: No URLs matched: gs://x/ops/heartbeat/latest.json"
    import os
    os.environ["GCS_BUCKET"] = "bkt"
    try:
        hb = P.heartbeat_adapter(run, "/x/gcloud_ro.sh")
        sh = P.shadow_adapter(run, "/x/gcloud_ro.sh")
    finally:
        del os.environ["GCS_BUCKET"]
    assert hb["state"] == "missing" and "未導入" in hb["reason"]
    assert sh["state"] == "missing"
    assert P.evaluate_overall({"vps": P.ok(PROBE), "heartbeat": hb, "shadow": sh})["level"] == "green"


def test_heartbeat_stale_is_red():
    rec = {"received_at": 1000.0, "latest": {"mem": {"available_mb": 1500}, "swap": {"used_mb": 0}, "units": {}, "disk": {"used_pct": 10}}}

    def run(cmd, input_text=None, timeout=30.0):
        return 0, json.dumps(rec), ""
    import os
    os.environ["GCS_BUCKET"] = "bkt"
    try:
        hb = P.heartbeat_adapter(run, "/x", now=1000.0 + 3600)
    finally:
        del os.environ["GCS_BUCKET"]
    assert hb["state"] == "ok" and hb["data"]["alerts"][0]["key"] == "stale"
    assert P.evaluate_overall({"vps": P.ok(PROBE), "heartbeat": hb})["level"] == "red"


def test_release_history_and_missing(tmp_path):
    assert P.release_history(tmp_path / "none.jsonl")["state"] == "missing"
    f = tmp_path / "h.jsonl"
    f.write_text('{"at":"t1","release":"r1","commit":"c1","result":"ok"}\nbroken\n{"at":"t2","release":"r2","commit":"c2","result":"failed"}\n')
    rows = P.release_history(f)["data"]["recent"]
    assert [r["release"] for r in rows] == ["r2", "r1"]


def test_collect_all_isolates_adapter_failures(tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_PROD_STATUS_DIR", str(tmp_path))
    monkeypatch.setattr(P, "release_history", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(P, "public_entry_probe", lambda *a, **k: P.ok({"url": "u", "status": 302, "redirect": "x"}))
    res = P.collect_all(fake_run, gcloud_ro="/x/gcloud_ro.sh", only=["vps", "releases", "public_entry", "cost"])
    assert res["sections"]["vps"]["state"] == "ok"
    assert res["sections"]["releases"]["state"] == "error" and "boom" in res["sections"]["releases"]["reason"]
    assert res["overall"]["level"] == "green"
    path = P.write_cache(res)
    assert json.loads(path.read_text())["schema"] == 1


def test_load_missing_stale_and_ok(tmp_path):
    p = tmp_path / "latest.json"
    assert P.load(p)["status"] == "not_collected"
    p.write_text("{broken")
    assert P.load(p)["status"] == "not_collected"
    P.write_cache({"schema": 1, "collected_at": "t", "sections": {"vps": P.ok(PROBE)}, "overall": {"level": "green", "reasons": []}}, p)
    assert P.load(p)["status"] == "ok"
    old = P.load(p, now=time.time() + 3600)
    assert old["status"] == "stale" and old["overall"]["level"] == "yellow"
    s = P.summary(P.load(p))
    assert s["release"] == "keiba-ui-abc" and s["env"] == "prod" and s["mem_available_mb"] == 1413.0


def test_notify_is_off_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("KEIBA_PROD_STATUS_NOTIFY", raising=False)
    assert P.maybe_notify({"overall": {"level": "red", "reasons": [{"level": "red", "message": "x"}]}}, tmp_path / "s.json") is False


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_PROD_STATUS_DIR", str(tmp_path))
    monitor_app.app.config["TESTING"] = True
    c = monitor_app.app.test_client()
    with c.session_transaction() as s:
        s["monitor_auth"] = True
    return c, tmp_path


def test_pages_require_login():
    monitor_app.app.config["TESTING"] = True
    c = monitor_app.app.test_client()
    for url in ("/prod", "/api/internal/prod-status"):
        r = c.get(url)
        assert r.status_code == 302 and "/login" in r.headers["Location"]


def test_page_and_api_without_cache_do_not_crash(client):
    c, _ = client
    d = c.get("/api/internal/prod-status").get_json()
    assert d["status"] == "not_collected" and d["overall"]["level"] == "unknown"
    assert c.get("/api/internal/prod-status?summary=1").get_json()["level"] == "unknown"
    html = c.get("/prod").get_data(as_text=True)
    assert "prod（ConoHa VPS）の状況" in html and 'href="/prod"' in html
    assert "prod-summary" in c.get("/services").get_data(as_text=True)


def test_api_with_cache(client):
    c, tmp = client
    res = P.collect_all(fake_run, gcloud_ro="/x", only=["vps"])
    P.write_cache(res, tmp / "latest.json")
    d = c.get("/api/internal/prod-status").get_json()
    assert d["status"] == "ok" and d["sections"]["vps"]["data"]["ui"]["json"]["env"] == "prod"
    s = c.get("/api/internal/prod-status?summary=1").get_json()
    assert s["level"] == "green" and s["release"] == "keiba-ui-abc"


def test_restart_delta_uses_previous_collection():
    prev = json.loads(json.dumps(PROBE)); prev["units"]["keiba-ui"]["restarts"] = 10
    cur = json.loads(json.dumps(PROBE)); cur["units"]["keiba-ui"]["restarts"] = 12
    P.add_restart_delta(cur, prev)
    assert cur["units"]["keiba-ui"]["restart_delta"] == 2 and cur["units"]["nginx"]["restart_delta"] == 0
    first = json.loads(json.dumps(PROBE))
    P.add_restart_delta(first, None)
    assert first["units"]["keiba-ui"]["restart_delta"] is None
    reset = json.loads(json.dumps(PROBE)); reset["units"]["keiba-ui"]["restarts"] = 0
    P.add_restart_delta(reset, prev)
    assert reset["units"]["keiba-ui"]["restart_delta"] is None                      # カウンタが戻った（unit を作り直した）


def test_public_entry_probe_uses_injected_get():
    sec = P.public_entry_probe(lambda u: (302, "https://x.cloudflareaccess.com/"))
    assert sec["state"] == "ok" and sec["data"]["status"] == 302
    assert P.public_entry_probe(lambda u: (None, "URLError"))["state"] == "error"
