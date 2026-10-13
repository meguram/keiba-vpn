"""dev 環境の自動ログイン: この PC の直アクセスだけ。公開トンネル・他ホスト・dev 以外では効かない。"""

from __future__ import annotations

import pytest

from src.api.auth import dev_auto_login_applies

LOCAL = {"host": "127.0.0.1:23100"}


@pytest.fixture()
def dev(monkeypatch):
    monkeypatch.setenv("KEIBA_ENV", "dev")
    monkeypatch.delenv("KEIBA_DEV_AUTO_LOGIN", raising=False)


def test_applies_for_local_browser_in_dev(dev):
    assert dev_auto_login_applies("127.0.0.1", LOCAL)
    assert dev_auto_login_applies("::1", {"host": "localhost:23100"})
    assert dev_auto_login_applies("127.0.0.1", {"host": "127.0.0.1:23110", "x-forwarded-host": "127.0.0.1:23100", "x-forwarded-for": "127.0.0.1"})
    assert dev_auto_login_applies("127.0.0.1", {"host": "[::1]:23100"})
    # Next のリバースプロキシ経由: X-Forwarded-For に IPv6/IPv4 射影のループバックが入る
    nxt = {"host": "127.0.0.1:23110", "x-forwarded-host": "127.0.0.1:23100", "x-forwarded-for": "::ffff:127.0.0.1"}
    assert dev_auto_login_applies("::ffff:127.0.0.1", nxt)
    assert dev_auto_login_applies("127.0.0.1", {**nxt, "x-forwarded-for": "::1"})


@pytest.mark.parametrize("env", ["", "stg", "prod", "production"])
def test_never_outside_explicit_dev(monkeypatch, env):
    monkeypatch.setenv("KEIBA_ENV", env)
    assert not dev_auto_login_applies("127.0.0.1", LOCAL)


def test_disabled_by_switch(dev, monkeypatch):
    monkeypatch.setenv("KEIBA_DEV_AUTO_LOGIN", "0")
    assert not dev_auto_login_applies("127.0.0.1", LOCAL)


def test_not_for_public_tunnel_or_remote_hosts(dev):
    # tcpexposer 経由: 接続元は localhost だが、元の Host は公開ドメイン
    assert not dev_auto_login_applies("127.0.0.1", {"host": "127.0.0.1:23110", "x-forwarded-host": "meguai-dev.tcpexposer.com"})
    assert not dev_auto_login_applies("127.0.0.1", {"host": "meguai-dev.tcpexposer.com"})
    assert not dev_auto_login_applies("127.0.0.1", {"host": "127.0.0.1:23110", "x-forwarded-for": "203.0.113.9"})
    assert not dev_auto_login_applies("127.0.0.1", {"host": "127.0.0.1:23110", "forwarded": "for=203.0.113.9;host=example.com"})
    assert not dev_auto_login_applies("192.168.0.5", LOCAL)  # LAN の他端末が localhost の Host を偽装しても接続元で弾く
    assert not dev_auto_login_applies(None, LOCAL)
    assert not dev_auto_login_applies("127.0.0.1", {"host": "127.0.0.1:23110", "x-forwarded-for": "::ffff:203.0.113.9"})
    assert not dev_auto_login_applies("127.0.0.1", {"host": "127.0.0.1:23110", "x-forwarded-for": "127.0.0.1, 203.0.113.9"})
    assert not dev_auto_login_applies("not-an-ip", LOCAL)


def test_flask_auth_status_and_login_required_endpoint(dev):
    from src.api.flask_app import create_app

    c = create_app().test_client()
    assert c.get("/api/v1/auth/status", headers={"Host": "127.0.0.1:23100"}).get_json()["logged_in"] is True
    tunnel = {"Host": "127.0.0.1:23110", "X-Forwarded-Host": "meguai-dev.tcpexposer.com"}
    assert c.get("/api/v1/auth/status", headers=tunnel).get_json()["logged_in"] is False
    # login 必須 API: 公開経由は 401
    r = c.post("/api/v1/betting/optimize", json={"race_id": "x"}, headers=tunnel)
    assert r.status_code == 401


def test_flask_unaffected_when_not_dev(monkeypatch):
    monkeypatch.setenv("KEIBA_ENV", "stg")
    from src.api.flask_app import create_app

    c = create_app().test_client()
    assert c.get("/api/v1/auth/status", headers={"Host": "127.0.0.1:23100"}).get_json()["logged_in"] is False


def test_fastapi_is_developer_follows_same_policy(dev):
    from types import SimpleNamespace

    from src.api.auth import classify_session, is_developer

    def req(host, headers):
        return SimpleNamespace(client=SimpleNamespace(host=host), headers=headers, cookies={})

    assert is_developer(req("127.0.0.1", LOCAL)) and classify_session(req("127.0.0.1", LOCAL)) == "valid"
    tunnel = {"host": "127.0.0.1:23120", "x-forwarded-host": "meguai-dev.tcpexposer.com"}
    assert not is_developer(req("127.0.0.1", tunnel)) and classify_session(req("127.0.0.1", tunnel)) == "none"
