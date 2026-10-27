"""API の GET 応答への短いキャッシュ指示（許可リスト方式・private・Vary: Cookie）。"""
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from src.api import http_cache_headers as H

CFG = {"enabled": True, "rules": [{"prefix": "/api/race/", "max_age": 30}, {"prefix": "/api/race-calendar", "max_age": 300}],
       "never": ["/api/health", "/api/v1/auth"]}


def test_max_age_rules():
    f = lambda p, m="GET", st=200, cc=False: H.max_age_for(p, m, st, cc, CFG)  # noqa: E731
    assert f("/api/race/202604040102/predictions") == 30
    assert f("/api/race-calendar") == 300
    assert f("/api/v1/auth/status") is None and f("/api/health") is None
    assert f("/api/other") is None
    assert f("/api/race/x", m="POST") is None and f("/api/race/x", st=404) is None and f("/api/race/x", cc=True) is None
    assert H.max_age_for("/api/race/x", "GET", 200, False, {**CFG, "enabled": False}) is None


def test_fastapi_headers(monkeypatch):
    monkeypatch.setattr(H, "config", lambda: CFG)
    app = FastAPI()

    @app.get("/api/race/{rid}")
    def race(rid: str):
        return {"id": rid}

    @app.get("/api/v1/auth/status")
    def auth():
        return {"ok": True}

    @app.get("/api/race-calendar")
    def cal():
        return JSONResponse({"a": 1}, headers={"Cache-Control": "no-store"})

    H.install_fastapi(app)
    c = TestClient(app)
    r = c.get("/api/race/1")
    assert r.headers["cache-control"] == "private, max-age=30" and "cookie" in r.headers["vary"].lower()
    assert "cache-control" not in c.get("/api/v1/auth/status").headers
    assert c.get("/api/race-calendar").headers["cache-control"] == "no-store"      # すでにある指示は上書きしない


def test_shared_cache_only_for_anonymous(monkeypatch):
    cfg = {**CFG, "rules": [{"prefix": "/api/race/", "max_age": 30, "s_maxage": 45}, {"prefix": "/api/race-calendar", "max_age": 300}]}
    monkeypatch.setattr(H, "config", lambda: cfg)
    app = FastAPI()

    @app.get("/api/race/{rid}")
    def race(rid: str):
        return {"id": rid}

    @app.get("/api/race-calendar")
    def cal():
        return {"a": 1}

    H.install_fastapi(app)
    c = TestClient(app)
    anon = c.get("/api/race/1")
    assert anon.headers["cache-control"] == "public, max-age=30, s-maxage=45" and "cookie" in anon.headers["vary"].lower()
    assert c.get("/api/race/1", headers={"Cookie": "keiba_dev_session=x"}).headers["cache-control"] == "private, max-age=30"
    assert c.get("/api/race/1", headers={"Authorization": "Bearer x"}).headers["cache-control"] == "private, max-age=30"
    assert c.get("/api/race-calendar").headers["cache-control"] == "private, max-age=300"     # s_maxage の無いルールは常に private


def test_dev_auto_login_is_not_anonymous(monkeypatch):
    monkeypatch.setenv("KEIBA_ENV", "dev")
    assert H.is_anonymous_request("", "", "127.0.0.1", {"host": "localhost:23110"}) is False
    assert H.is_anonymous_request("", "", "203.0.113.5", {"host": "stg.example.com"}) is True
    assert H.is_anonymous_request("a=b", "", "203.0.113.5", {}) is False
