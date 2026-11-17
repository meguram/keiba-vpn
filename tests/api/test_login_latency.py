"""ログインを速く・安全にする変更（取り込みを待たない・名簿の取り直し・照合の回数・応答の中身）。外部接続なし。"""

import pytest

from src.api import auth
from src.scripts.ops import manage_users as mu


@pytest.fixture()
def members(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "MEMBERS_LOCAL_FILE", str(tmp_path / "auth" / "members.json"))
    monkeypatch.setenv("DEV_SECRET_KEY", "test-secret")
    monkeypatch.delenv("KEIBA_MEMBERS_PEPPER", raising=False)
    monkeypatch.setattr(auth, "members_refresh_hook", None)
    auth._members_cache["key"] = None
    return tmp_path / "auth" / "members.json"


def test_unknown_id_costs_the_same_iterations_as_a_real_member(members, monkeypatch):
    assert auth.MEMBER_ITERATIONS == mu.ITERATIONS          # 存在しない ID も会員と同じ回数（応答時間で ID の有無が分からない）
    seen = []
    real = auth.hashlib.pbkdf2_hmac
    monkeypatch.setattr(auth.hashlib, "pbkdf2_hmac", lambda *a: seen.append(a[3]) or real(a[0], a[1], a[2], 1))
    assert not auth.verify_member_login("nobody", "whatever-password")
    assert seen == [mu.ITERATIONS]


def test_member_login_refreshes_the_roster_before_checking(members):
    calls = []

    def hook():                                              # 定期の取り込みより先に、作ったばかりの会員が名簿に入った状態を再現
        calls.append(1)
        if len(calls) == 1:
            mu.main(["add", "newbie", "--password", "correct-horse-battery"])
        return True

    auth.members_refresh_hook = hook
    assert auth.verify_member_login("newbie", "correct-horse-battery")
    assert not auth.verify_member_login("nobody", "correct-horse-battery")
    assert calls == [1, 1]                                   # ID の有無に関係なく同じ扱い（回数の制限は refresh_members 側）


def test_refresh_failures_and_slow_gcs_do_not_break_or_stall_login(members, monkeypatch):
    import time

    mu.main(["add", "taro", "--password", "correct-horse-battery"])

    def broken():
        raise RuntimeError("gcs down")

    auth.members_refresh_hook = broken
    assert auth.verify_member_login("taro", "correct-horse-battery")      # 取り直しに失敗しても手元の名簿で照合する
    monkeypatch.setattr(auth, "MEMBERS_REFRESH_WAIT_SEC", 0.05)
    auth.members_refresh_hook = lambda: time.sleep(1) or True
    t0 = time.monotonic()
    assert auth.verify_member_login("taro", "correct-horse-battery")
    assert time.monotonic() - t0 < 0.8                       # GCS が止まっていても待つのは上限まで
    called = []
    auth.members_refresh_hook = lambda: called.append(1) or False
    assert not auth.verify_member_login("", "x") and called == []         # ID が空（管理者のパスワード）は名簿と無関係


def test_refresh_members_is_rate_limited(tmp_path, monkeypatch):
    from src.ops import gcs_mirror

    class Be:
        n = 0

        def download(self, key):
            Be.n += 1
            assert key == "ops/mirror/data/local/auth/members.json"
            return b'{"users": []}'

    t = [1000.0]
    monkeypatch.setitem(gcs_mirror._members_refresh, "at", -1e18)
    assert gcs_mirror.refresh_members(tmp_path, Be(), clock=lambda: t[0])
    assert (tmp_path / "data/local/auth/members.json").read_bytes() == b'{"users": []}'
    t[0] += 30
    assert not gcs_mirror.refresh_members(tmp_path, Be(), clock=lambda: t[0])      # 60 秒に 1 回まで（失敗を繰り返されても GCS を読みすぎない）
    t[0] += 31
    assert gcs_mirror.refresh_members(tmp_path, Be(), clock=lambda: t[0])
    assert Be.n == 2


@pytest.fixture()
def flask_client(members, monkeypatch):
    monkeypatch.setenv("KEIBA_ENV", "")
    from src.api.flask_app import create_app

    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_login_response_carries_auth_status_and_timing(flask_client):
    mu.main(["add", "taro", "--password", "correct-horse-battery"])
    r = flask_client.post("/api/v1/auth/login", json={"id": "taro", "password": "correct-horse-battery"})
    assert r.status_code == 200
    j = r.get_json()
    assert j["auth"]["logged_in"] is True and j["auth"]["user_id"] == "taro"       # 画面が状態を取り直さなくてよい
    assert j["auth"] == flask_client.get("/api/v1/auth/status").get_json()        # GET /auth/status と同じ形・同じ中身
    assert "pw;dur=" in r.headers["Server-Timing"] and "app;dur=" in r.headers["Server-Timing"]
    assert r.headers["Cache-Control"] == "no-store"
    bad = flask_client.post("/api/v1/auth/login", json={"id": "taro", "password": "wrong-password-xx"})
    assert bad.status_code == 401 and "Server-Timing" not in bad.headers and "auth" not in bad.get_json()     # 失敗には時間の内訳を出さない
    assert "app;dur=" in flask_client.get("/api/v1/auth/status").headers["Server-Timing"]


def test_auth_paths_skip_the_mirror_gate_and_report_instance_age(monkeypatch):
    monkeypatch.setenv("KEIBA_ENV", "dev")
    monkeypatch.setenv("KEIBA_CLOUDRUN_LAZY", "1")
    monkeypatch.delenv("KEIBA_SERVING_STATE_PULL", raising=False)
    monkeypatch.delenv("GCS_BUCKET", raising=False)
    monkeypatch.setattr(auth, "members_refresh_hook", None)
    from starlette.testclient import TestClient

    from src.api import cloudrun_entry as entry

    for p in ("/api/v1/auth/login", "/api/v1/auth/status", "/api/v1/auth/logout"):
        assert not entry.gate_applies(p), p                  # ログインまわりは取り込みを待たない・きっかけにしない
    assert entry.gate_applies("/api/v1/authx") and entry.gate_applies("/api/v1/races")

    gw_app = entry.build_app()
    assert auth.members_refresh_hook is None                 # バケットが無い環境（テスト・dev）では名簿を取りに行かない
    started = []

    class Fr:
        wait_s = 0

        def should_wait(self):
            return True

        def start_if_due(self):
            started.append(1)
            return None

    gw_app.app.refresher = Fr()
    c = TestClient(gw_app)
    r = c.get("/api/v1/auth/status")
    assert r.status_code == 200 and started == []
    assert 'inst;desc="age=' in r.headers.get("server-timing", "")
    assert "inst;" not in c.get("/api/v1/health").headers.get("server-timing", "")
    assert "inst;" not in c.post("/api/v1/auth/login", json={"id": "x", "password": "y"}).headers.get("server-timing", "")


def test_previous_month_shared_password_is_explained_and_not_counted(flask_client, monkeypatch):
    from src.scripts.ops import shared_login as sl
    from src.utils.anti_scraping import get_engine

    monkeypatch.setattr(sl, "previous_month", lambda now_=None: "2026-11")
    monkeypatch.setattr(sl, "month_of", lambda now_=None: "2026-12")
    mu.main(["add", "shared-id", "--password", sl.password_for("2026-12")])     # 共通ログイン（今月の共通パスワード）
    mu.main(["add", "taro", "--password", "correct-horse-battery"])              # ふつうの会員
    eng = get_engine()
    eng._login_fail.clear()
    eng._login_lock.clear()
    prev = sl.password_for("2026-11")
    for _ in range(12):                                               # 先月のパスワードを何度打っても、ロックの失敗に数えない
        r = flask_client.post("/api/v1/auth/login", json={"id": "shared-id", "password": prev})
        assert r.status_code == 401 and r.get_json()["reason"] == "password_rotated"
        assert "X-Keiba-Login-Not-Counted" not in r.headers           # 内部の合図は外に出さない
    assert not any(eng._login_fail.values()) and not eng._login_lock
    for uid in ("taro", "nobody"):                                    # 共通 ID 以外（ほかの会員・存在しない ID）では、ふつうの失敗として数える
        r = flask_client.post("/api/v1/auth/login", json={"id": uid, "password": prev})
        assert r.status_code == 401 and r.get_json().get("reason") != "password_rotated"
    assert sum(len(q) for q in eng._login_fail.values()) == 2
    assert flask_client.post("/api/v1/auth/login", json={"id": "shared-id", "password": sl.password_for("2026-12")}).status_code == 200


def test_previous_month_is_computed_across_year_end():
    from datetime import datetime

    from src.scripts.ops import shared_login as sl

    assert sl.previous_month(datetime(2027, 1, 1, 0, 10, tzinfo=sl.JST)) == "2026-12"
    assert sl.previous_month(datetime(2026, 12, 31, 23, 0, tzinfo=sl.JST)) == "2026-11"
    assert not sl.is_previous_password("")
    assert sl.is_previous_password(sl.password_for("2026-11"), now=datetime(2026, 12, 1, 0, 1, tzinfo=sl.JST))
    assert not sl.is_previous_password(sl.password_for("2026-12"), now=datetime(2026, 12, 1, 0, 1, tzinfo=sl.JST))


def test_lockout_config_suits_a_shared_account():
    import json
    from pathlib import Path

    cfg = json.loads((Path(__file__).resolve().parents[2] / "config" / "anti_scraping.json").read_text(encoding="utf-8"))["login"]
    assert cfg["max_failures"] >= 10 and max(cfg["lockout_seconds"]) <= 3600   # 共有の出口（携帯の CGNAT）で他の会員を長く巻き込まない
