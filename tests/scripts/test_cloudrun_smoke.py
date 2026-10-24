import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from src.scripts.ops import cloudrun_smoke as M


def serve(table, key):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            ok_key = self.headers.get("X-Keiba-Proxy-Key") == key
            if self.path in ("/api/health",) or (self.path == "/api/v1/health" and ok_key):
                self._send(200, {"status": "ok"})
            elif not ok_key:
                self._send(401, {"error": "unauthorized"})
            elif self.path.startswith("/ops/heartbeat/status"):
                self._send(200, {"status": "ok"})
            elif "/predictions" in self.path:
                self._send(200, {"predictions": table})
            else:
                self._send(404, {})

        def _send(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

    s = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, f"http://127.0.0.1:{s.server_port}"


T = [{"horse_number": 1, "win_prob": 0.2, "top2_prob": 0.4, "top3_prob": 0.5}, {"horse_number": 2, "win_prob": 0.1, "top2_prob": 0.3, "top3_prob": 0.4}]


def test_smoke_ok_and_compare(monkeypatch, capsys):
    s1, u1 = serve(T, "k")
    s2, u2 = serve(T, "cf")
    monkeypatch.setenv("KEIBA_PROXY_KEY", "k")
    monkeypatch.setenv("CFV", "cf")
    try:
        # 比較先は X-Keiba-Proxy-Key に別の値を要求するサーバ（ヘッダ名を差し替えて渡す）
        rc = M.main(["--base", u1, "--race-id", "1", "--compare-base", u2, "--compare-header", "X-Keiba-Proxy-Key=CFV"])
    finally:
        s1.shutdown(); s2.shutdown()
    out = capsys.readouterr().out
    assert rc == 0 and "確率が一致（2 頭）" in out


def test_smoke_detects_mismatch_and_missing_key(monkeypatch, capsys):
    other = [{**T[0], "win_prob": 0.25}, T[1]]
    s1, u1 = serve(T, "k")
    s2, u2 = serve(other, "k")
    monkeypatch.setenv("KEIBA_PROXY_KEY", "k")
    try:
        rc = M.main(["--base", u1, "--race-id", "1", "--compare-base", u2, "--compare-header", "X-Keiba-Proxy-Key=KEIBA_PROXY_KEY"])
        assert rc == 1 and "馬番1 win_prob" in capsys.readouterr().out
        monkeypatch.delenv("KEIBA_PROXY_KEY")
        assert M.main(["--base", u1, "--race-id", "1"]) == 1
    finally:
        s1.shutdown(); s2.shutdown()


def test_diff_tables():
    a = M.prob_table(json.dumps({"predictions": T}).encode())
    assert M.diff_tables(a, dict(a)) == []
    assert M.diff_tables(a, {1: a[1]}) and M.diff_tables(a, {**a, 2: (0.1, 0.3, None)})
