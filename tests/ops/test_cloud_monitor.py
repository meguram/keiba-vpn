"""GCP の監視ポータル（src/monitor/cloud_app.py）。MemoryIO だけで完結する。"""

import json
import time
from datetime import datetime

from src.monitor import cloud_app as M
from src.ops.gcs_io import MemoryIO

NOW = datetime(2026, 10, 26, 10, 0, tzinfo=M.JST).timestamp()


def put(io, key, obj):
    io.put_bytes(key, json.dumps(obj).encode())


def seeded():
    io = MemoryIO()
    put(io, "ops/heartbeat/latest.json", {"received_at": NOW - 120, "history": [], "latest": {
        "mem": {"available_mb": 1500, "used_mb": 480}, "swap": {"used_mb": 0}, "disk": {"root_used_pct": 40.0}, "units": {"nginx": {"active": "active"}}, "checks": {}}})
    put(io, "ops/job_runs/predict-race/latest.json", {"status": "ok", "started_at": "2026-10-26T09:00:00+09:00", "duration_sec": 41.2, "memory_peak_mib": 2900, "memory_limit_mib": 8192, "parallel": False})
    put(io, "ops/job_runs/eve-precompute/latest.json", {"status": "failed", "reason": "step rc=1", "started_at": "2026-10-26T08:00:00+09:00"})
    put(io, "ops/alerts/20261026/a.json", {"title": "ジョブが失敗: eve-precompute", "lines": ["step rc=1"], "severity": "critical", "job": "eve-precompute", "at": "2026-10-26T08:01:00+09:00"})
    put(io, "ops/releases/ui/latest.json", {"release_id": "keiba-ui-b", "commit": "abcdef123456", "created_at": "2026-10-26T01:00:00+09:00"})
    put(io, "ops/releases/ui/applied/vps.json", {"release_id": "keiba-ui-a", "result": "ok"})
    put(io, "ops/parity/20261026/predict-race.json", {"ok": False, "status": "mismatch", "problems": ["行数 3 ≠ 4"]})
    return io


IDS = [("predict-race", True), ("eve-precompute", True), ("lap-forecast-eve", False)]


def test_collect_levels_and_rows():
    d = M.collect(seeded(), NOW, IDS)
    s = d["sections"]
    assert s["heartbeat"]["level"] == "green" and s["heartbeat"]["mem_available_mb"] == 1500
    rows = {r["job"]: r for r in s["jobs"]["rows"]}
    assert rows["predict-race"]["level"] == "green" and rows["eve-precompute"]["level"] == "red" and rows["lap-forecast-eve"]["status"] == "未実行"
    assert s["jobs"]["level"] == "red" and d["overall"] == "red"
    assert s["alerts"]["critical"] == 1 and s["alerts"]["items"][0]["job"] == "eve-precompute"
    ui = s["releases"]["rows"][0]
    assert ui["latest"] == "keiba-ui-b" and ui["applied"] == "keiba-ui-a" and not ui["in_sync"] and ui["commit"] == "abcdef12"
    assert s["parity"]["date"] == "20261026" and s["parity"]["level"] == "yellow"


def test_stale_or_missing_heartbeat_is_red():
    assert M.collect(MemoryIO(), NOW, [])["sections"]["heartbeat"]["level"] == "red"
    io = MemoryIO()
    put(io, "ops/heartbeat/latest.json", {"received_at": NOW - 3600, "latest": {"mem": {"available_mb": 1500}, "swap": {"used_mb": 0}, "disk": {}, "units": {}, "checks": {}}, "history": []})
    assert M.collect(io, NOW, [])["sections"]["heartbeat"]["level"] == "red"


def test_app_pages_and_proxy_key(monkeypatch):
    app = M.create_app(seeded(), now_fn=lambda: NOW, job_ids=IDS)
    c = app.test_client()
    monkeypatch.setenv("KEIBA_PROXY_KEY_REQUIRED", "1")
    monkeypatch.setenv("KEIBA_PROXY_KEY", "k1")
    assert c.get("/healthz").status_code == 200                       # ヘルスチェックは鍵なし
    assert c.get("/").status_code == 401 and c.get("/api/overview", headers={"X-Keiba-Proxy-Key": "bad"}).status_code == 401
    r = c.get("/", headers={"X-Keiba-Proxy-Key": "k1"})
    assert r.status_code == 200 and "運用監視" in r.get_data(as_text=True) and "predict-race" in r.get_data(as_text=True)
    j = c.get("/api/overview", headers={"X-Keiba-Proxy-Key": "k1"}).get_json()
    assert j["overall"] == "red"
    monkeypatch.setenv("KEIBA_PROXY_KEY_REQUIRED", "0")
    assert c.get("/api/overview").status_code == 200                  # 既定オフ（ローカル確認用）
