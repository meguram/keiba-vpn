import time
from datetime import datetime, timedelta, timezone

import pytest

from src.api import auth
from src.scripts.ops import shared_login as sl

JST = timezone(timedelta(hours=9))
ID = "megu-friends"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "MEMBERS_LOCAL_FILE", str(tmp_path / "auth" / "members.json"))
    monkeypatch.setenv("DEV_SECRET_KEY", "test-secret")
    monkeypatch.delenv("KEIBA_MEMBERS_PEPPER", raising=False)
    auth._members_cache["key"] = None
    sent = []
    monkeypatch.setattr("src.utils.notify.notify_slack", lambda m, **k: sent.append(m) or True)
    return sent


def test_password_is_stable_per_month_and_differs_between_months(env):
    assert sl.password_for("2026-11") == sl.password_for("2026-11")
    assert sl.password_for("2026-11") != sl.password_for("2026-12")
    assert len(sl.password_for("2026-11")) == 19


def test_rotation_on_new_month_revokes_old_sessions_and_notifies_once(env):
    oct_ = datetime(2026, 10, 31, 12, tzinfo=JST)
    nov = datetime(2026, 11, 1, 0, 5, tzinfo=JST)
    assert sl.ensure(ID, oct_) == "created"
    assert auth.verify_member_login(ID, sl.password_for("2026-10"))
    old = auth._make_token(role=auth.ROLE_MEMBER, uid=ID)
    assert auth.token_role(old) == auth.ROLE_MEMBER
    assert sl.ensure(ID, oct_) == "ok" and len(env) == 1                       # 同じ月は変更なし・通知も重ねない
    time.sleep(1.1)
    assert sl.ensure(ID, nov) == "rotated"
    assert not auth.verify_member_login(ID, sl.password_for("2026-10"))
    assert auth.verify_member_login(ID, sl.password_for("2026-11"))
    assert auth.token_role(old) is None                                         # 前月のログインは解除
    assert len(env) == 2 and sl.password_for("2026-11") in env[1] and ID in env[1]
    assert sl.ensure(ID, nov) == "ok" and len(env) == 2                         # 冪等


def test_failed_slack_is_retried_and_disabled_account_untouched(env, monkeypatch):
    monkeypatch.setattr("src.utils.notify.notify_slack", lambda m, **k: False)
    now = datetime(2026, 11, 1, 0, 5, tzinfo=JST)
    assert sl.ensure(ID, now) == "created" and sl._state()["notified"] is False
    got = []
    monkeypatch.setattr("src.utils.notify.notify_slack", lambda m, **k: got.append(m) or True)
    assert sl.ensure(ID, now) == "notified" and len(got) == 1                   # 次の実行で再送
    from src.scripts.ops import manage_users as mu
    mu.set_enabled(ID, False)
    assert sl.ensure(ID, datetime(2026, 12, 1, tzinfo=JST)) == "disabled"       # 管理者が止めたアカウントは触らない


def test_dry_run_reports_without_writing_or_notifying(env):
    oct_ = datetime(2026, 10, 31, 12, tzinfo=JST)
    nov = datetime(2026, 11, 1, 0, 5, tzinfo=JST)
    assert sl.ensure(ID, oct_, dry=True) == "would-created"
    assert auth.member_record(ID) is None and sl._state() == {} and env == []          # 名簿・状態・通知に触れない
    assert sl.ensure(ID, oct_) == "created" and len(env) == 1
    assert sl.ensure(ID, oct_, dry=True) == "ok"
    time.sleep(1.1)
    assert sl.ensure(ID, nov, dry=True) == "would-rotated"
    assert auth.verify_member_login(ID, sl.password_for("2026-10")) and len(env) == 1
