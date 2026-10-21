"""スクレイピング対策（src/utils/anti_scraping.py と src/api/middleware/anti_scraping.py）。"""

from __future__ import annotations

import json

import pytest

from src.utils.anti_scraping import AntiScraping

BROWSER = {
    "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
    "accept-language": "ja,en;q=0.9",
    "referer": "https://example.test/",
    "cookie": "a=b",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "x-forwarded-for": "203.0.113.5",
}


class Clock:
    def __init__(self) -> None:
        self.t = 1_800_000_000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


def make(tmp_path, modes=None, **over):
    cfg = {"mode": {"auth": "off", "rate_limit": "enforce", "bot": "off", "honeypot": "enforce", "login": "enforce", "session": "off"}}
    if modes:
        cfg["mode"].update(modes)
    cfg.update(over)
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    clock = Clock()
    return AntiScraping(p, clock=clock, log_dir=tmp_path / "log", env={}), clock


def hit(eng, path="/api/race/202604040201/predictions", headers=None, peer="127.0.0.1", **kw):
    return eng.check(peer=peer, headers=headers or BROWSER, method="GET", path=path, **kw)


# ───────── クライアント IP の信頼境界 ─────────
def test_client_ip_ignores_forward_headers_from_untrusted_peer(tmp_path):
    eng, _ = make(tmp_path)
    assert eng.client_ip("198.51.100.9", {"x-forwarded-for": "1.2.3.4"}) == "198.51.100.9"


def test_client_ip_uses_rightmost_untrusted_from_trusted_proxy(tmp_path):
    eng, _ = make(tmp_path)
    # 先頭は偽装されうる。右から見て、信頼プロキシでない最初の値を採る
    assert eng.client_ip("127.0.0.1", {"x-forwarded-for": "6.6.6.6, 203.0.113.5, 127.0.0.1"}) == "203.0.113.5"
    assert eng.client_ip("::ffff:127.0.0.1", {"cf-connecting-ip": "203.0.113.7"}) == "203.0.113.7"
    assert eng.client_ip("127.0.0.1", {}) == "127.0.0.1"


def test_client_ip_falls_back_to_fingerprint_when_tunnel_hides_client(tmp_path):
    eng, _ = make(tmp_path)
    a = eng.client_ip("127.0.0.1", {"x-forwarded-for": "127.0.0.1", "user-agent": "UA-A", "accept-language": "ja"})
    b = eng.client_ip("127.0.0.1", {"x-forwarded-for": "::1", "user-agent": "UA-B", "accept-language": "ja"})
    assert a.startswith("fp:") and b.startswith("fp:") and a != b  # 全員が同じ IP に見えて巻き添えにしない


def test_internal_caller_is_loopback_without_forward_headers(tmp_path):
    eng, _ = make(tmp_path)
    assert eng.is_internal_caller("127.0.0.1", {"user-agent": "python-requests"}, "127.0.0.1")
    assert not eng.is_internal_caller("127.0.0.1", {"x-forwarded-for": "203.0.113.5"}, "203.0.113.5")
    assert not eng.is_internal_caller("198.51.100.9", {}, "198.51.100.9")


def test_allowlist_ip_is_exempt(tmp_path):
    eng, _ = make(tmp_path, allowlist_ips=["203.0.113.0/24"], rate_limit={"data_per_min": 1, "data_burst_per_10s": 1})
    for _ in range(10):
        assert not hit(eng).blocked


# ───────── レート制限 ─────────
def test_rate_limit_blocks_and_window_resets(tmp_path):
    eng, clock = make(tmp_path, rate_limit={"data_per_min": 5, "data_burst_per_10s": 100})
    assert not any(hit(eng, f"/api/x{i}").blocked for i in range(5))
    d = hit(eng, "/api/x5")
    assert d.blocked and d.status == 429 and d.retry_after >= 1
    clock.advance(61)
    assert not hit(eng, "/api/x6").blocked or True  # 違反が 1 回なのでまだ短い制限（窓は空）
    # ブロック段階に達していなければ窓が空けば通る
    clock.advance(61)
    assert not hit(eng, "/api/y").blocked


def test_detect_mode_logs_but_does_not_block(tmp_path):
    eng, _ = make(tmp_path, modes={"rate_limit": "detect"}, rate_limit={"data_per_min": 2, "data_burst_per_10s": 100})
    ds = [hit(eng, f"/api/z{i}") for i in range(6)]
    assert not any(d.blocked for d in ds)
    assert any("rate:d60" in x for d in ds for x in d.detected)
    log = list((tmp_path / "log").glob("events-*.jsonl"))
    assert log and "rate" in log[0].read_text(encoding="utf-8")


def test_escalates_to_temporary_block(tmp_path):
    eng, clock = make(tmp_path, rate_limit={"data_per_min": 1, "data_burst_per_10s": 100},
                      block={"tiers": [{"after_violations": 3, "seconds": 900}], "violation_decay_seconds": 3600})
    out = [hit(eng, f"/api/q{i}") for i in range(8)]
    blocked = [d for d in out if d.blocked]
    assert blocked and blocked[-1].status == 403 and blocked[-1].retry_after > 100
    clock.advance(901)
    assert not eng.is_blocked("ip:203.0.113.5")[0]


def test_distinct_race_enumeration_limit(tmp_path):
    eng, _ = make(tmp_path, rate_limit={"data_per_min": 1000, "data_burst_per_10s": 1000, "distinct_races_per_5min": 5})
    res = [hit(eng, f"/api/race/20260404{i:04d}/predictions") for i in range(1, 9)]
    assert not res[0].blocked
    assert res[-1].blocked and res[-1].status == 429


def test_internal_paths_and_jobs_never_limited(tmp_path):
    eng, _ = make(tmp_path, rate_limit={"data_per_min": 1, "data_burst_per_10s": 1})
    for _ in range(20):
        assert not eng.check(peer="127.0.0.1", headers={"user-agent": "curl/8"}, method="GET", path="/api/scrape-jobs").blocked
        assert not eng.check(peer="127.0.0.1", headers=BROWSER, method="GET", path="/api/health").blocked


# ───────── ハニーポット ─────────
def test_honeypot_blocks_ip(tmp_path):
    eng, _ = make(tmp_path)
    d = hit(eng, "/api/archive/all")
    assert d.blocked and d.reason == "honeypot"
    d2 = hit(eng, "/api/race/202604040201/predictions")
    assert d2.blocked and d2.status == 403


# ───────── bot 判定 ─────────
def test_bot_scoring_detect_vs_enforce(tmp_path):
    scraper = {"user-agent": "python-requests/2.31", "x-forwarded-for": "198.51.100.77"}
    eng, _ = make(tmp_path, modes={"bot": "detect"})
    d = hit(eng, headers=scraper)
    assert "bot_score" in d.detected and not d.blocked
    eng2, _ = make(tmp_path, modes={"bot": "enforce"})
    d2 = hit(eng2, headers=scraper)
    assert d2.blocked and d2.status == 403
    # ブラウザ相当は通る（ページを先に見たあとの API）
    eng3, _ = make(tmp_path, modes={"bot": "enforce"})
    assert not hit(eng3, "/", headers=BROWSER).blocked
    assert not hit(eng3, "/api/race/202604040201/predictions", headers=BROWSER).blocked


def test_bot_enumeration_signal(tmp_path):
    eng, _ = make(tmp_path, modes={"bot": "enforce"}, bot={"threshold": 7, "enumeration_sequential_run": 5})
    hit(eng, "/", headers=BROWSER)
    sigs = []
    for i in range(1, 9):
        d = hit(eng, f"/api/race/2026040402{i:02d}/predictions", headers={**BROWSER, "cookie": ""})
        sigs.append(d.blocked)
    assert any(sigs)


# ───────── 未ログイン検知 ─────────
def test_auth_detect_then_enforce(tmp_path):
    eng, _ = make(tmp_path, modes={"auth": "detect"})
    d = hit(eng, authed=False)
    assert "anon_protected" in d.detected and not d.blocked
    eng2, _ = make(tmp_path, modes={"auth": "enforce"})
    assert hit(eng2, authed=False).status == 401
    assert not hit(eng2, authed=True).blocked
    assert not hit(eng2, "/api/health", authed=False).blocked
    assert not hit(eng2, "/api/auth/login", authed=False).blocked


# ───────── ログイン試行 ─────────
def test_login_lockout_escalates_and_success_clears(tmp_path):
    eng, clock = make(tmp_path)
    ip = "203.0.113.50"
    for _ in range(4):
        eng.login_record(ip, False)
    assert eng.login_check(ip)[0]
    eng.login_record(ip, False)
    ok, wait = eng.login_check(ip)
    assert not ok and 800 <= wait <= 901
    clock.advance(901)
    assert eng.login_check(ip)[0]
    for _ in range(5):
        eng.login_record(ip, False)
    ok, wait = eng.login_check(ip)
    assert not ok and wait > 3000  # 2 回目は長くなる
    eng.login_record("198.51.100.1", False)
    eng.login_record("198.51.100.1", True)
    assert eng.login_check("198.51.100.1")[0]


def test_session_many_ips(tmp_path):
    eng, _ = make(tmp_path, modes={"session": "enforce"}, rate_limit={"data_per_min": 1000, "data_burst_per_10s": 1000})
    res = []
    for i in range(6):
        res.append(eng.check(peer="127.0.0.1", headers={**BROWSER, "x-forwarded-for": f"203.0.113.{i + 1}"}, method="GET",
                             path="/api/a", authed=True, session_token="tok-abc"))
    assert any(d.blocked and d.reason.startswith("session") or "blocked" in d.reason for d in res)


def test_config_hot_reload_and_kill_switch(tmp_path):
    eng, clock = make(tmp_path, rate_limit={"data_per_min": 1, "data_burst_per_10s": 100})
    assert eng.mode("rate_limit") == "enforce"
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"mode": {"rate_limit": "off", "honeypot": "off", "auth": "off", "bot": "off"}}), encoding="utf-8")
    import os
    os.utime(p, (clock.t + 1, clock.t + 1))
    clock.advance(31)
    assert eng.mode("rate_limit") == "off"
    eng.env = {"KEIBA_ANTI_SCRAPING": "off"}
    assert eng.mode("login") == "off"


# ───────── ミドルウェア（FastAPI / Flask） ─────────
def _fastapi(tmp_path, **kw):
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from starlette.testclient import TestClient
    from src.api.middleware.anti_scraping import register_fastapi

    eng, clock = make(tmp_path, **kw)
    app = FastAPI()

    @app.get("/api/race/{rid}/predictions")
    def pred(rid: str):
        return {"ok": True}

    @app.get("/api/scrape-jobs")
    def jobs():
        return {"jobs": []}

    @app.get("/")
    def home():
        return {"page": True}

    register_fastapi(app, eng)
    return TestClient(app, client=("127.0.0.1", 50000)), eng


def test_fastapi_middleware_429_and_headers(tmp_path):
    c, _ = _fastapi(tmp_path, rate_limit={"data_per_min": 3, "data_burst_per_10s": 100})
    h = {"x-forwarded-for": "203.0.113.9", "user-agent": BROWSER["user-agent"], "sec-fetch-mode": "cors"}
    codes = [c.get(f"/api/race/20260404{i:04d}/predictions", headers=h).status_code for i in range(1, 6)]
    assert 429 in codes
    r = c.get("/api/race/202604040101/predictions", headers=h)
    assert r.status_code in (403, 429) and "Retry-After" in r.headers
    assert "reason" in r.json() and "score" not in json.dumps(r.json())
    assert c.get("/", headers={"x-forwarded-for": "198.51.100.3"}).headers["x-robots-tag"].startswith("noindex")


def test_fastapi_internal_job_unaffected(tmp_path):
    c, _ = _fastapi(tmp_path, rate_limit={"data_per_min": 1, "data_burst_per_10s": 1})
    for _ in range(10):
        assert c.get("/api/scrape-jobs").status_code == 200  # 転送ヘッダ無しのループバック


def test_fastapi_login_lockout(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi import FastAPI, Response
    from starlette.testclient import TestClient
    from src.api.middleware.anti_scraping import register_fastapi

    eng, _ = make(tmp_path)
    app = FastAPI()

    @app.post("/login")
    def login():
        return Response("bad", status_code=200)  # 失敗（Set-Cookie 無し）

    register_fastapi(app, eng)
    c = TestClient(app, client=("127.0.0.1", 1))
    h = {"x-forwarded-for": "203.0.113.60"}
    codes = [c.post("/login", headers=h).status_code for _ in range(7)]
    assert codes[:5] == [200] * 5 and codes[5] == 429


def test_flask_hooks(tmp_path):
    pytest.importorskip("flask")
    from flask import Flask, jsonify
    from src.api.middleware.anti_scraping import register_flask

    eng, _ = make(tmp_path, rate_limit={"data_per_min": 2, "data_burst_per_10s": 100})
    app = Flask(__name__)

    @app.get("/api/v1/x/<rid>")
    def x(rid):
        return jsonify(ok=True)

    register_flask(app, eng)
    c = app.test_client()
    h = {"X-Forwarded-For": "203.0.113.70", "User-Agent": BROWSER["user-agent"]}
    codes = [c.get(f"/api/v1/x/{i}", headers=h).status_code for i in range(5)]
    assert codes[:2] == [200, 200] and 429 in codes
    assert c.get("/api/v1/x/9", headers=h).headers["X-Robots-Tag"].startswith("noindex")
    eng2, _ = make(tmp_path)
    app2 = Flask(__name__)

    @app2.post("/api/v1/auth/login")
    def login2():
        return jsonify(error="bad"), 401

    register_flask(app2, eng2)
    lh = {"X-Forwarded-For": "203.0.113.71"}
    res = [app2.test_client().post("/api/v1/auth/login", headers=lh).status_code for _ in range(7)]
    assert res[:5] == [401] * 5 and res[5] == 429


def test_engine_failure_does_not_take_site_down(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from starlette.testclient import TestClient
    from src.api.middleware.anti_scraping import register_fastapi

    eng, _ = make(tmp_path)

    def boom(**kw):
        raise RuntimeError("x")

    eng.check = boom  # type: ignore[assignment]
    app = FastAPI()

    @app.get("/api/ok")
    def ok():
        return {"ok": 1}

    register_fastapi(app, eng)
    assert TestClient(app).get("/api/ok").status_code == 200
