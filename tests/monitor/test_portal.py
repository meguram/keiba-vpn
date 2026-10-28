"""監視ポータル（cloud_app）の画面・受信・認証。GCS は MemoryIO、ハートビートはファイル、HTTP 確認は差し替え。外部接続なし。"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from src.monitor import cloud_app as M
from src.monitor import environments as E
from src.ops import heartbeat_schema as S
from src.ops import heartbeat_store as store
from src.ops import pipeline_health
from src.ops.gcs_io import MemoryIO
from tests.monitor.test_traffic import _bundle
from tests.ops.test_pipeline_health import all_ok_src

NOW_DT = datetime(2026, 10, 28, 12, 0, tzinfo=M.JST)
NOW = NOW_DT.timestamp()
KEY = {"X-Keiba-Proxy-Key": "k1"}
SECRET = "sec"


def hb_record(age=120, checks=None, units=None):
    return {"received_at": NOW - age, "history": [], "latest": {
        "mem": {"available_mb": 1500, "used_mb": 480}, "swap": {"used_mb": 0}, "disk": {"root_used_pct": 40.0},
        "units": units or {"nginx": {"active": "active"}, "keiba-ui": {"active": "active"}},
        "checks": checks or {"nextjs_http": 200, "nginx_http": 200}}}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("KEIBA_PROXY_KEY_REQUIRED", "1")
    monkeypatch.setenv("KEIBA_PROXY_KEY", "k1")
    monkeypatch.setenv("KEIBA_HEARTBEAT_SECRET", SECRET)
    monkeypatch.setattr(pipeline_health, "GcsSources", lambda io: all_ok_src(days=("20261024", "20261025")))
    E._cache.clear()
    io = MemoryIO()
    io.put_bytes("ops/heartbeat/latest.json", json.dumps(hb_record()).encode())
    io.put_bytes("ops/traffic/bundle.json", json.dumps(_bundle(received_at=NOW - 60)).encode())
    io.put_bytes("ops/seo/audit.json", json.dumps({"status": "ok", "score": 85, "counts": {"ok": 11, "warn": 1, "ng": 1}, "placeholders": 7, "site_url": "https://x",
                                                   "generated_at": "2026-10-28T09:00:00+0900", "groups": [{"title": "技術", "items": [{"level": "ng", "label": "robots", "detail": "d", "fix": "f"}]}],
                                                   "pages": [{"url": "/", "title": "t", "chars": 900, "noindex": False, "canonical": True, "issues": ["説明文が短い"]}]}).encode())
    calls: list[str] = []

    def fake_get(url, timeout):
        calls.append(url)
        return (200, 123) if "stg" not in url else (503, 80)

    backend = store.FileBackend(tmp_path / "hb.json")
    env = {"KEIBA_API_URL_PROD": "https://api.example", "KEIBA_API_URL_STG": "https://rstg---api.example"}
    sent: list = []
    from tests.ops.test_api_cost_guard import hours as cost_hours
    app = M.create_app(io, now_fn=lambda: NOW, job_ids=[("predict-race", True)], hb_backend=backend, http_get=fake_get, env=env,
                       cost_fetch=lambda *a: cost_hours(lambda h: 3600),
                       notify=lambda title, lines, severity="warning": sent.append((title, severity)))
    c = app.test_client()
    c.calls, c.io, c.backend, c.sent = calls, io, backend, sent
    return c


def test_every_page_renders_with_key_and_is_blocked_without(client):
    for path in ("/", "/pipeline", "/traffic", "/seo", "/ops"):
        assert client.get(path).status_code == 401, path
        r = client.get(path, headers=KEY)
        assert r.status_code == 200 and r.headers["Cache-Control"] == "no-store", path
    for path in ("/api/home", "/api/pipeline", "/api/traffic", "/api/seo", "/api/overview"):
        assert client.get(path, headers=KEY).status_code == 200, path
    assert client.get("/api/health").status_code == 200


def test_home_shows_four_environment_cells_and_probes_api_once(client):
    html = client.get("/", headers=KEY).get_data(as_text=True)
    for word in ("本番（prod）", "ステージング（stg）", "VPS（画面・取得役）", "GCP（API・Jobs）", "VPS（stg の画面）", "GCP（stg タグの API）"):
        assert word in html
    j = client.get("/api/home", headers=KEY).get_json()
    assert j["env"]["cells"]["prod"]["gcp"]["level"] in ("green", "unknown")
    assert j["env"]["cells"]["stg"]["gcp"]["level"] == "red"         # stg タグが 503
    assert j["env"]["cells"]["stg"]["vps"]["headline"] == "未導入"     # stg の画面を入れていない
    assert len(client.calls) == 2                                      # 2 回目以降は使い回し（API を叩きすぎない）
    assert client.get("/api/home?probe=0", headers=KEY).get_json()["env"]["probed"] is False


def test_home_and_check_include_api_cost_guard(client, monkeypatch):
    monkeypatch.setenv("GCS_PROJECT_ID", "p")
    j = client.get("/api/cost", headers=KEY).get_json()
    assert j["status"] == "ok" and j["level"] == "red" and j["avg_instances_24h"] == pytest.approx(1.0)
    html = client.get("/", headers=KEY).get_data(as_text=True)
    assert "API の課金ガード" in html and "夜間" in html
    assert client.get("/api/home", headers=KEY).get_json()["overall"] == "red"       # 課金の異常はホーム全体の信号に反映
    r = client.post("/ops/heartbeat/check", headers=KEY).get_json()
    assert r["cost"]["ok"] and "api_alive_overnight" in r["cost"]["sent"]
    assert any("夜間" in t or "月額" in t for t, _ in client.sent)
    assert json.loads(client.io.get_bytes("ops/cost_guard/latest.json"))["status"] == "ok"


def test_home_pipeline_and_traffic_and_seo_summaries(client):
    j = client.get("/api/home", headers=KEY).get_json()
    assert j["pipe"]["status"] == "ok" and j["pipe"]["date"] == "20261025"
    assert j["traf"]["status"] == "ok" and j["traf"]["d7_pv"] == 6
    assert j["seo"]["score"] == 85


def test_traffic_and_seo_pages_content(client):
    html = client.get("/traffic", headers=KEY).get_data(as_text=True)
    assert "<svg" in html and "/race/202604040401" in html and "Googlebot" in html
    seo = client.get("/seo", headers=KEY).get_data(as_text=True)
    assert "85" in seo and "対処: f" in seo and "説明文が短い" in seo


def test_pages_degrade_when_nothing_collected(monkeypatch):
    monkeypatch.setenv("KEIBA_PROXY_KEY_REQUIRED", "0")
    E._cache.clear()
    monkeypatch.setattr(pipeline_health, "GcsSources", lambda io: (_ for _ in ()).throw(RuntimeError("gcs down")))
    app = M.create_app(MemoryIO(), now_fn=lambda: NOW, job_ids=[], env={}, http_get=lambda u, t: (0, 0), cost_fetch=lambda *a: [])
    c = app.test_client()
    for path in ("/", "/pipeline", "/traffic", "/seo"):
        assert c.get(path).status_code == 200, path
    assert "読み取れませんでした" in c.get("/pipeline").get_data(as_text=True)
    assert "未設定" in c.get("/").get_data(as_text=True)


# ── 受信 ──
def sign_headers(body: bytes, secret=SECRET, ts=None):
    ts = int(NOW if ts is None else ts)
    return {S.HEADER_TS: str(ts), S.HEADER_SIG: S.sign(secret, ts, body)}


def test_signed_posts_need_no_proxy_key_but_a_valid_signature(client):
    body = json.dumps(_bundle()).encode()
    body = json.dumps({"v": 1, "sent_at": int(NOW), "readable": True, "windows": {}, "days": {}}).encode()
    assert client.post("/ops/traffic", data=body).status_code == 401
    assert client.post("/ops/traffic", data=body, headers=sign_headers(body, secret="bad")).status_code == 401
    r = client.post("/ops/traffic", data=body, headers=sign_headers(body))
    assert r.status_code == 200 and r.get_json()["ok"]
    assert json.loads(client.io.get_bytes("ops/traffic/bundle.json"))["received_at"] == NOW


def test_heartbeat_post_then_check_notifies_on_stale(client):
    from tests.ops.test_heartbeat import payload

    body = json.dumps(payload(sent_at=NOW)).encode()
    r = client.post("/ops/heartbeat", data=body, headers=sign_headers(body))
    assert r.status_code == 200
    assert client.post("/ops/heartbeat/check").status_code == 401                 # 点検は共有鍵が要る
    assert client.post("/ops/heartbeat/check", headers=KEY).get_json()["sent"] == []
    rec, _ = client.backend.read()
    assert rec["received_at"] == NOW


def test_oversized_post_is_rejected(client):
    big = b"x" * 400_000
    assert client.post("/ops/traffic", data=big, headers=sign_headers(big)).status_code == 413


# ── 環境の 4 マス ──
def test_vps_cells_prod_and_stg():
    ok = M.heartbeat_section(_io_with(hb_record(checks={"nextjs_http": 200, "nginx_http": 200, "stg_ui_http": 200}, units={"nginx": {"active": "active"}, "keiba-ui-stg": {"active": "active"}})), NOW)
    prod, stg = E.vps_cells(ok)
    assert prod["level"] == "green" and stg["level"] == "green"
    down = M.heartbeat_section(_io_with(hb_record(checks={"nextjs_http": 502, "nginx_http": 200, "stg_ui_http": 0})), NOW)
    prod, stg = E.vps_cells(down)
    assert prod["level"] == "red" and stg["level"] == "red"
    stale = M.heartbeat_section(_io_with(hb_record(age=3600)), NOW)
    prod, stg = E.vps_cells(stale)
    assert prod["level"] == "red" and stg["level"] == "unknown"
    none = E.vps_cells(M.heartbeat_section(MemoryIO(), NOW))
    assert none[0]["level"] == "red"


def _io_with(rec):
    io = MemoryIO()
    io.put_bytes("ops/heartbeat/latest.json", json.dumps(rec).encode())
    return io


def test_probe_is_cached_and_unconfigured_is_unknown():
    E._cache.clear()
    n = []
    get = lambda u, t: (n.append(u) or (200, 5))  # noqa: E731
    env = {"KEIBA_API_URL_PROD": "https://p"}
    hb = {"age_sec": 60, "level": "green", "alerts": [], "checks": {}, "units": {}, "mem_available_mb": 1000, "disk_used_pct": 10}
    a = E.build(hb, {"rows": [], "level": "green"}, env=env, get=get, now=1000)
    E.build(hb, {"rows": [], "level": "green"}, env=env, get=get, now=1000 + E.PROBE_TTL_S - 1)
    assert len(n) == 1 and a["cells"]["stg"]["gcp"]["headline"] == "接続先が未設定"
    E.build(hb, {"rows": [], "level": "green"}, env=env, get=get, now=1000 + E.PROBE_TTL_S + 1)
    assert len(n) == 2
