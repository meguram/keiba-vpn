import json
import os
import time

import pytest

from src.api import auth
from src.scripts.ops import manage_users as mu


@pytest.fixture()
def members(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "MEMBERS_LOCAL_FILE", str(tmp_path / "auth" / "members.json"))
    monkeypatch.setenv("DEV_SECRET_KEY", "test-secret")
    auth._members_cache["key"] = None
    return tmp_path / "auth" / "members.json"


def test_add_login_and_name(members, capsys):
    import io, sys
    sys.stdin = io.StringIO("correct-horse-battery\n")
    assert mu.main(["add", "taro", "--name", "太郎", "--password-stdin"]) == 0
    sys.stdin = sys.__stdin__
    assert auth.verify_member_login("taro", "correct-horse-battery")
    assert not auth.verify_member_login("taro", "wrong-password-xx")
    assert not auth.verify_member_login("nobody", "correct-horse-battery")
    assert auth.member_display_name("taro") == "太郎"
    raw = members.read_text(encoding="utf-8")
    assert "correct-horse-battery" not in raw and oct(os.stat(members).st_mode & 0o777) == "0o600"      # 平文は残さない・本人だけ読める


def test_generate_password_is_shown_once_and_policy(members, capsys):
    assert mu.main(["add", "hanako", "--generate"]) == 0
    out = capsys.readouterr().out
    pw = out.split("この 1 回だけ表示）: ")[1].strip()
    assert auth.verify_member_login("hanako", pw) and len(pw) >= 12
    with pytest.raises(SystemExit):
        mu.check_password_rules("abc", "short")
    with pytest.raises(SystemExit):
        mu.main(["add", "x"])                      # ID が短すぎる（先に検査される前に入力待ちにならない）


def test_disable_remove_and_session_revocation(members):
    mu.main(["add", "jiro", "--generate"])
    tok = auth._make_token(role=auth.ROLE_MEMBER, uid="jiro")
    assert auth.token_role(tok) == auth.ROLE_MEMBER and auth.token_user(tok) == "jiro"
    mu.main(["disable", "jiro"])
    assert auth.token_role(tok) is None and auth.token_user(tok) == ""      # 無効化すると、ログイン中のセッションも使えない
    mu.main(["enable", "jiro"])
    assert auth.token_role(tok) == auth.ROLE_MEMBER
    mu.main(["remove", "jiro"])
    assert auth.token_role(tok) is None and not auth.verify_member_login("jiro", "x")


def test_token_formats_and_tampering(members):
    old = auth._make_token(role=auth.ROLE_DEV)                              # 従来形式（uid なし）
    assert old.count(":") == 2 and auth.token_role(old) == auth.ROLE_DEV
    mu.main(["add", "ken", "--generate"])
    new = auth._make_token(role=auth.ROLE_MEMBER, uid="ken")
    assert new.count(":") == 3 and auth.token_user(new) == "ken"
    forged = new.replace(("ken".encode().hex()), ("admin".encode().hex()))   # uid だけ書き換えると署名が合わない
    assert auth.token_role(forged) is None
    assert auth.token_role(auth._make_token(timestamp=int(time.time()) - auth.COOKIE_MAX_AGE - 10, role=auth.ROLE_MEMBER, uid="ken")) is None


def test_password_option_sets_chosen_initial_password(members):
    assert mu.main(["add", "ume", "--name", "梅", "--password", "my-first-password-1"]) == 0
    assert auth.verify_member_login("ume", "my-first-password-1")
    assert mu.main(["passwd", "ume", "--password", "another-password-22"]) == 0
    assert auth.verify_member_login("ume", "another-password-22") and not auth.verify_member_login("ume", "my-first-password-1")
    with pytest.raises(SystemExit):
        mu.main(["add", "take", "--password", "short"])            # 規則（10 文字以上）は --password でも守る
    assert "my-first-password-1" not in members.read_text(encoding="utf-8")
