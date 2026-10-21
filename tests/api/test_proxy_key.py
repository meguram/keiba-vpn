"""共有ヘッダ認証（X-Keiba-Proxy-Key）。外部接続なし。"""

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from src.api.middleware.proxy_key import ProxyKeyMiddleware, configured_keys, is_required, key_matches


async def _echo(request: Request):
    return JSONResponse({"client": request.client.host if request.client else None})


def _client(env, **kw):
    inner = Starlette(routes=[Route("/api/health", _echo), Route("/api/x", _echo), Route("/ops/heartbeat", _echo)])
    return TestClient(ProxyKeyMiddleware(inner, env=env, **kw), client=("203.0.113.9", 5555))


def test_default_off_passes_everything():
    c = _client({})
    assert not is_required({})
    assert c.get("/api/x").status_code == 200


def test_required_rejects_missing_and_wrong():
    env = {"KEIBA_PROXY_KEY_REQUIRED": "1", "KEIBA_PROXY_KEY": "secret-A"}
    c = _client(env)
    assert c.get("/api/x").status_code == 401
    assert c.get("/api/x", headers={"X-Keiba-Proxy-Key": "nope"}).status_code == 401
    r = c.get("/api/x", headers={"X-Keiba-Proxy-Key": "secret-A"})
    assert r.status_code == 200
    assert "secret-A" not in c.get("/api/x").text   # 拒否応答に鍵が出ない


def test_rotation_accepts_old_and_next():
    env = {"KEIBA_PROXY_KEY_REQUIRED": "1", "KEIBA_PROXY_KEY": "old", "KEIBA_PROXY_KEY_NEXT": "new"}
    c = _client(env)
    assert c.get("/api/x", headers={"X-Keiba-Proxy-Key": "old"}).status_code == 200
    assert c.get("/api/x", headers={"X-Keiba-Proxy-Key": "new"}).status_code == 200
    assert c.get("/api/x", headers={"X-Keiba-Proxy-Key": "older"}).status_code == 401


def test_required_without_any_key_denies_all():
    c = _client({"KEIBA_PROXY_KEY_REQUIRED": "1"})
    assert c.get("/api/x", headers={"X-Keiba-Proxy-Key": ""}).status_code == 401
    assert c.get("/api/x", headers={"X-Keiba-Proxy-Key": "anything"}).status_code == 401
    assert c.get("/api/health").status_code == 200   # 除外パスは通す


def test_exempt_paths():
    c = _client({"KEIBA_PROXY_KEY_REQUIRED": "1", "KEIBA_PROXY_KEY": "k"}, exempt_paths=("/api/health", "/ops/heartbeat"))
    assert c.get("/api/health").status_code == 200
    assert c.get("/ops/heartbeat").status_code == 200
    assert c.get("/api/x").status_code == 401


def test_client_ip_adopted_only_after_key_verified():
    env = {"KEIBA_PROXY_KEY_REQUIRED": "1", "KEIBA_PROXY_KEY": "k"}
    c = _client(env)
    ok = c.get("/api/x", headers={"X-Keiba-Proxy-Key": "k", "X-Keiba-Client-IP": "198.51.100.7"})
    assert ok.json()["client"] == "198.51.100.7"
    bad = c.get("/api/x", headers={"X-Keiba-Proxy-Key": "k", "X-Keiba-Client-IP": "not-an-ip"})
    assert bad.json()["client"] == "203.0.113.9"
    # 鍵が通らなければ 401（ヘッダの IP は使われない）
    assert c.get("/api/x", headers={"X-Keiba-Client-IP": "198.51.100.7"}).status_code == 401


def test_helpers():
    assert key_matches("a", [b"a", b"b"]) and not key_matches("c", [b"a"]) and not key_matches(None, [b"a"])
    assert configured_keys({"KEIBA_PROXY_KEY": " x ", "KEIBA_PROXY_KEY_NEXT": ""}) == [b"x"]
