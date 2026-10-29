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


# ---- 確定済みのレースは共有キャッシュに長く置く ----
from datetime import datetime, timedelta, timezone  # noqa: E402

JST = timezone(timedelta(hours=9))
NOW = datetime(2026, 10, 29, 12, 0, tzinfo=JST)
SCFG = {
    "enabled": True,
    "rules": [{"prefix": "/api/race/", "max_age": 30, "s_maxage": 30}, {"prefix": "/api/race-list/", "max_age": 60, "s_maxage": 60}],
    "never": ["/api/health", "/api/admin"],
    "settled": {"s_maxage": 604800, "grace_days": 3, "prefixes": ["/api/race-list/", "/api/race/"]},
}


def test_settled_by_date_token():
    f = lambda p: H.settled_s_maxage_for(p, SCFG, NOW)  # noqa: E731
    assert f("/api/race-list/20261026") == 604800            # ちょうど3日前 = 確定
    assert f("/api/race-list/20261027") is None              # 2日前はまだ動く
    assert f("/api/race-list/20261029") is None
    assert f("/api/race-list/20261301") is None              # 不正な日付は短いまま
    assert f("/api/other/20200101") is None                  # 許可リスト外
    assert f("/api/admin/20200101") is None


def test_settled_by_race_id_year_rule_and_peek(monkeypatch):
    f = lambda p: H.settled_s_maxage_for(p, SCFG, NOW)  # noqa: E731
    monkeypatch.setattr(H, "_race_date_peek", None)
    assert f("/api/race/202506050811/predictions") == 604800     # 前の年 → 確定扱い
    assert f("/api/race/202606050811/predictions") is None       # 今年で日付不明 → 短いまま
    monkeypatch.setattr(H, "_race_date_peek", lambda rid: {"202606050811": "20260906", "202606050812": "20261028"}.get(rid, ""))
    assert f("/api/race/202606050811") == 604800                 # 索引で9月と分かる
    assert f("/api/race/202606050812") is None                   # 昨日のレースは動く
    assert f("/api/race/202606050899") is None                   # 索引に無い
    monkeypatch.setattr(H, "_race_date_peek", lambda rid: (_ for _ in ()).throw(RuntimeError("boom")))
    assert f("/api/race/202606050811") is None                   # 例外は短いまま（落ちない）


def test_settled_disabled_or_zero():
    assert H.settled_s_maxage_for("/api/race-list/20200101", {**SCFG, "enabled": False}, NOW) is None
    assert H.settled_s_maxage_for("/api/race-list/20200101", {**SCFG, "settled": {}}, NOW) is None


def test_cache_control_value_settled_only_for_anonymous():
    assert H.cache_control_value(30, 30, True, 604800) == "public, max-age=30, s-maxage=604800"
    assert H.cache_control_value(30, 30, True) == "public, max-age=30, s-maxage=30"
    assert H.cache_control_value(30, 30, False, 604800) == "private, max-age=30"
    assert H.cache_control_value(30, None, True, 604800) == "private, max-age=30"     # s_maxage の無いルールは長くしない


def test_fastapi_settled_header(monkeypatch):
    monkeypatch.setattr(H, "config", lambda: SCFG)
    monkeypatch.setattr(H, "_race_date_peek", None)
    app = FastAPI()

    @app.get("/api/race/{rid}")
    def race(rid: str):
        return {"id": rid}

    H.install_fastapi(app)
    c = TestClient(app)
    y = datetime.now(JST).year
    assert c.get(f"/api/race/{y - 1}06050811").headers["cache-control"] == "public, max-age=30, s-maxage=604800"
    assert c.get(f"/api/race/{y - 1}06050811", headers={"Cookie": "k=v"}).headers["cache-control"] == "private, max-age=30"
    assert c.get(f"/api/race/{y}06050811").headers["cache-control"] == "public, max-age=30, s-maxage=30"
