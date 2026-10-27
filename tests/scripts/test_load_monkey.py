"""load_monkey: 判定・集計・ブラウズ計画と、ダミーサーバーに対する一連の流れ（段階→崩壊検知→回復）。"""

import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from src.scripts.ops import load_monkey as L

POOLS = {"items": [{"date": "20261017", "hot": True, "race_id": f"20260504010{i}", "horse_ids": [], "jockey_ids": [], "trainer_ids": []} for i in range(1, 6)]}


def test_classify_levels():
    assert L.classify(300, 0.0) == "快適"
    assert L.classify(2000, 0.0) == "許容" and L.classify(500, 0.008) == "許容"
    assert L.classify(5000, 0.0) == "悪化" and L.classify(900, 0.05) == "悪化"
    assert L.classify(20000, 0.0) == "崩壊" and L.classify(100, 0.2) == "崩壊" and L.classify(None, 0.0) == "崩壊"


def test_percentile_and_stats_count_failures_as_slow():
    assert L.percentile([], 95) is None and L.percentile(list(range(1, 101)), 95) == 96
    st = L.Stats()
    for _ in range(9):
        st.add("/a", "200", 10.0, True)
    st.add("/a", "ReadTimeout", 30000.0, False)
    s = st.summary(10)
    assert s["error_rate"] == 0.1 and s["p99_ms"] == 30000.0 and s["verdict"] == "崩壊" and s["rps"] == 1.0


def test_browse_plan_mixes_pages_and_apis_for_one_date():
    plan = L.browse_plan(POOLS, random.Random(3))
    paths = [p for p, _ in plan]
    assert paths[0] == "/weekly-predictions" and any("race-values" in p for p in paths) and any(p.endswith("/predictions") for p in paths)
    assert all(t > 0 for _, t in plan)


class _H(BaseHTTPRequestHandler):
    delay = 0.0

    def do_GET(self):
        if "/slow" in self.path:
            time.sleep(self.delay)
        body = json.dumps({"ok": True}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_end_to_end_against_dummy_server(tmp_path):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        out = tmp_path / "o"
        rc = L.main(["single", "--path", "/fast", "--all-base", base, "--steps", "2,4", "--hold", "2", "--cooldown", "0", "--recovery-max", "10", "--max-load-per-core", "1000", "--min-mem-gb", "0", "--out", str(out)])
        assert rc == 0
        s = json.loads((out / "summary.json").read_text())
        assert [r["level"] for r in s["steps"]] == [1, 2, 4] and s["good_max"] == 4 and s["recovery"]["recovered"] is True
        assert all(r["error_rate"] == 0 for r in s["steps"]) and s["steps"][-1]["rps"] > 5
        pools = tmp_path / "p.json"
        pools.write_text(json.dumps(POOLS))
        out2 = tmp_path / "o2"
        assert L.main(["browse", "--pools", str(pools), "--all-base", base, "--steps", "3", "--hold", "3", "--cooldown", "0", "--recovery-max", "10", "--max-load-per-core", "1000", "--min-mem-gb", "0", "--out", str(out2)]) == 0
        assert json.loads((out2 / "summary.json").read_text())["steps"][-1]["requests"] > 0
    finally:
        srv.shutdown()


def test_safety_abort_stops_ramp(tmp_path, monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    monkeypatch.setattr(L, "mem_available_gb", lambda: 1.0)       # 空きメモリが足りない状況
    try:
        out = tmp_path / "o"
        L.main(["single", "--path", "/fast", "--all-base", base, "--steps", "2,4,8", "--hold", "5", "--cooldown", "0", "--recovery-max", "6", "--max-load-per-core", "1000", "--out", str(out)])
        s = json.loads((out / "summary.json").read_text())
        assert s["aborted"] and "空きメモリ" in s["aborted"][0] and len(s["steps"]) < 4
    finally:
        srv.shutdown()


def test_multi_process_steps_aggregate_all_workers(tmp_path):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        out = tmp_path / "o"
        assert L.main(["single", "--path", "/fast", "--all-base", base, "--steps", "16", "--procs", "2", "--hold", "3", "--cooldown", "0", "--recovery-max", "10", "--max-load-per-core", "1000", "--min-mem-gb", "0", "--out", str(out)]) == 0
        s = json.loads((out / "summary.json").read_text())
        last = s["steps"][-1]
        assert last["level"] == 16 and last["procs"] == 2 and last["requests"] > 50 and last["error_rate"] == 0
    finally:
        srv.shutdown()
