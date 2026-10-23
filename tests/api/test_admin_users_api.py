import pytest

from src.api import auth
from src.api.flask_app import create_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "MEMBERS_LOCAL_FILE", str(tmp_path / "auth" / "members.json"))
    monkeypatch.setenv("DEV_SECRET_KEY", "test-secret")
    monkeypatch.delenv("KEIBA_MEMBERS_PEPPER", raising=False)
    auth._members_cache["key"] = None
    app = create_app()
    app.testing = True
    return app.test_client()


def _as(client, role):
    client.set_cookie(auth.COOKIE_NAME, auth._make_token(role=role), domain="localhost")
    return client


def test_only_admin_can_manage_users(client):
    assert client.get("/api/v1/admin/users").status_code == 401                              # 未ログイン
    assert client.post("/api/v1/admin/users", json={"id": "abc", "password": "long-enough-pw"}).status_code == 401
    _as(client, auth.ROLE_MEMBER)
    assert client.get("/api/v1/admin/users").status_code == 401                              # 会員でも不可（管理者＝開発者のセッションだけ）


def test_admin_register_list_disable_delete(client):
    _as(client, auth.ROLE_DEV)
    r = client.post("/api/v1/admin/users", json={"id": "Taro_Yamada", "password": "correct-horse-battery"})
    assert r.status_code == 200 and len(r.get_json()["fp"]) == 8
    fp = r.get_json()["fp"]
    assert auth.verify_member_login("taro_yamada", "correct-horse-battery")
    listing = client.get("/api/v1/admin/users")
    assert listing.headers["Cache-Control"] == "no-store"
    assert "taro" not in listing.get_data(as_text=True).lower() and "correct-horse" not in listing.get_data(as_text=True)      # ID もパスワードも返さない
    assert client.post("/api/v1/admin/users", json={"id": "Taro_Yamada", "password": "another-long-password"}).status_code == 400      # 重複
    bad = client.post("/api/v1/admin/users", json={"id": "newuser", "password": "short"})
    assert bad.status_code == 400 and "10" in bad.get_json()["error"]
    assert client.post(f"/api/v1/admin/users/{fp}/enabled", json={"enabled": False}).status_code == 200
    assert not auth.verify_member_login("taro_yamada", "correct-horse-battery")
    assert client.post(f"/api/v1/admin/users/{fp}/password", json={"password": "brand-new-password-9"}).status_code == 200
    assert client.delete(f"/api/v1/admin/users/{fp}").status_code == 200
    assert client.delete(f"/api/v1/admin/users/{fp}").status_code == 400                     # もう無い
