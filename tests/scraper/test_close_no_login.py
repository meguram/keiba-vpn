"""作業の無い定期ジョブが、閉じるためだけにログイン（と会員ページの確認）をしないこと。"""
from __future__ import annotations

from src.scraper import client as client_mod


def test_close_without_any_request_does_not_login(monkeypatch):
    calls = []
    monkeypatch.setattr(client_mod.NetkeibaClient, "login", lambda self, *a, **k: calls.append("login") or True)
    monkeypatch.setenv("NETKEIBA_LAZY_LOGIN", "1")
    c = client_mod.NetkeibaClient(auto_login=True)
    assert c._login_pending is True
    c.close()
    assert calls == []
    assert c._login_pending is False


def test_first_request_still_logs_in(monkeypatch):
    calls = []
    monkeypatch.setattr(client_mod.NetkeibaClient, "login", lambda self, *a, **k: calls.append("login") or True)
    monkeypatch.setenv("NETKEIBA_LAZY_LOGIN", "1")
    c = client_mod.NetkeibaClient(auto_login=True)
    _ = c._session          # 通信が必要になった時点でログインする（従来どおり）
    assert calls == ["login"]
    c.close()
