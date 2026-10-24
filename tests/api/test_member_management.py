import os
import time

import pytest

from src.api import auth
from src.scripts.ops import manage_users as mu


@pytest.fixture()
def members(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "MEMBERS_LOCAL_FILE", str(tmp_path / "auth" / "members.json"))
    monkeypatch.setenv("DEV_SECRET_KEY", "test-secret")
    monkeypatch.delenv("KEIBA_MEMBERS_PEPPER", raising=False)
    auth._members_cache["key"] = None
    return tmp_path / "auth" / "members.json"


def test_add_login_and_nothing_readable_is_stored(members):
    assert mu.main(["add", "Taro_Yamada", "--password", "correct-horse-battery"]) == 0
    assert auth.verify_member_login("Taro_Yamada", "correct-horse-battery")
    assert auth.verify_member_login("  taro_yamada ", "correct-horse-battery")          # 大文字小文字・前後の空白は区別しない
    assert not auth.verify_member_login("Taro_Yamada", "wrong-password-xx")
    assert not auth.verify_member_login("nobody", "correct-horse-battery")
    raw = members.read_text(encoding="utf-8")
    assert "taro" not in raw.lower() and "yamada" not in raw.lower() and "correct-horse" not in raw       # ID もパスワードも平文では残らない
    assert "name" not in raw and oct(os.stat(members).st_mode & 0o777) == "0o600"
    assert auth.member_display_name("Taro_Yamada") == "Taro_Yamada"                       # 表示は本人が入力した ID


def test_leaked_file_alone_is_useless_without_the_pepper(members, monkeypatch):
    mu.main(["add", "hanako", "--password", "correct-horse-battery"])
    assert auth.verify_member_login("hanako", "correct-horse-battery")
    monkeypatch.setenv("KEIBA_MEMBERS_PEPPER", "attacker-guess")                          # ペッパー（サーバの秘密）が違えば、ID も パスワードも照合できない
    assert not auth.verify_member_login("hanako", "correct-horse-battery")
    assert auth.member_record("hanako") is None


def test_list_shows_fingerprint_only_and_can_operate_by_it(members, capsys):
    mu.main(["add", "jiro", "--generate"])
    pw = capsys.readouterr().out.split("この 1 回だけ表示）: ")[1].strip()
    rows = [r for r in mu.list_users() if r["source"] == "managed"]
    assert len(rows) == 1 and rows[0]["label"] is None and len(rows[0]["fp"]) == 8 and "jiro" not in str(rows)      # 一覧に ID は出ない
    fp = rows[0]["fp"]
    tok = auth._make_token(role=auth.ROLE_MEMBER, uid="jiro")
    assert auth.verify_member_login("jiro", pw) and auth.token_user(tok) == "jiro"
    mu.set_enabled(fp, False)                                                             # 指紋で無効化 → ログイン中のセッションも使えない
    assert not auth.verify_member_login("jiro", pw) and auth.token_role(tok) is None
    mu.set_enabled("jiro", True)                                                          # ID でも操作できる
    assert auth.token_role(tok) == auth.ROLE_MEMBER
    mu.set_password(fp, "brand-new-password-9")
    assert auth.verify_member_login("jiro", "brand-new-password-9") and not auth.verify_member_login("jiro", pw)
    mu.remove_user("jiro")
    assert auth.token_role(tok) is None
    assert [r for r in mu.list_users() if r["source"] == "managed"] == []


def test_rules_and_duplicates(members):
    with pytest.raises(mu.UserError):
        mu.create_user("ab", "long-enough-password")                                       # ID が短すぎる
    with pytest.raises(mu.UserError):
        mu.create_user("okid", "short")                                                    # パスワードが短い
    with pytest.raises(mu.UserError):
        mu.create_user("sameidsameid", "sameidsameid")                                     # ID と同じ
    mu.create_user("okid", "long-enough-password")
    with pytest.raises(mu.UserError):
        mu.create_user("OKID", "another-long-password")                                    # 重複（大文字小文字は区別しない）
    with pytest.raises(mu.UserError):
        mu.remove_user("ghost-user")


def test_token_formats_and_tampering(members):
    old = auth._make_token(role=auth.ROLE_DEV)                              # 従来形式（uid なし）
    assert old.count(":") == 2 and auth.token_role(old) == auth.ROLE_DEV
    mu.create_user("ken", "long-enough-password")
    new = auth._make_token(role=auth.ROLE_MEMBER, uid="ken")
    assert new.count(":") == 3 and auth.token_user(new) == "ken"
    forged = new.replace(("ken".encode().hex()), ("admin".encode().hex()))   # uid だけ書き換えると署名が合わない
    assert auth.token_role(forged) is None
    assert auth.token_role(auth._make_token(timestamp=int(time.time()) - auth.COOKIE_MAX_AGE - 10, role=auth.ROLE_MEMBER, uid="ken")) is None


def test_password_change_revokes_existing_sessions(members):
    mu.create_user("shared", "correct-horse-battery")
    old = auth._make_token(role=auth.ROLE_MEMBER, uid="shared")
    assert auth.token_role(old) == auth.ROLE_MEMBER
    time.sleep(1.1)
    mu.set_password("shared", "another-long-password-1")
    assert auth.token_role(old) is None
    new = auth._make_token(role=auth.ROLE_MEMBER, uid="shared")
    assert auth.token_role(new) == auth.ROLE_MEMBER


def test_four_tiers_guest_free_paid_dev(members):
    mu.create_user("paiduser", "correct-horse-battery")                     # 区分を指定しない（従来のアカウント）は有料会員
    mu.create_user("freeuser", "correct-horse-battery", plan="free")
    t = lambda uid: auth._make_token(role=auth.ROLE_MEMBER, uid=uid)        # noqa: E731
    assert auth.token_tier("") == "" and auth.token_tier("garbage") == ""            # ゲスト
    assert auth.token_tier(t("freeuser")) == "free"
    assert auth.token_tier(t("paiduser")) == "paid"
    assert auth.token_tier(auth._make_token(role=auth.ROLE_DEV)) == "dev"
    g, f, p, d = (auth.tier_fields(x) for x in ("", t("freeuser"), t("paiduser"), auth._make_token(role=auth.ROLE_DEV)))
    assert (g["logged_in"], g["is_member"], g["tier"]) == (False, False, "public")
    assert (f["logged_in"], f["is_member"], f["tier"]) == (True, False, "free")      # ログインはしているが有料ゾーンは不可
    assert (p["logged_in"], p["is_member"], p["tier"]) == (True, True, "member")
    assert (d["is_member"], d["is_admin"], d["tier"]) == (True, True, "dev")
    mu.set_plan("freeuser", "paid")                                                  # 区分の変更は発行済みのセッションにもすぐ効く
    assert auth.token_tier(t("freeuser")) == "paid"
    with pytest.raises(mu.UserError):
        mu.set_plan("freeuser", "gold")


def test_free_member_is_not_in_paid_zone_for_requests(members):
    from flask import Flask

    from src.api import flask_auth as fa

    mu.create_user("freeuser", "correct-horse-battery", plan="free")
    mu.create_user("paiduser", "correct-horse-battery")
    app = Flask(__name__)
    for uid, want in (("freeuser", False), ("paiduser", True), ("", False)):
        with app.test_request_context("/", headers={"Cookie": f"{auth.COOKIE_NAME}={auth._make_token(role=auth.ROLE_MEMBER, uid=uid) if uid else ''}"}):
            assert fa.is_member_session() is want
            assert fa.get_auth_status()["logged_in"] is bool(uid)


def test_legacy_plaintext_id_record_is_converted_and_listed(members):
    members.parent.mkdir(parents=True, exist_ok=True)
    import json as _j
    members.write_text(_j.dumps({"users": [{"id": "OldUser", "name": "OldUser", "role": "member", "enabled": True, "algo": "pbkdf2_sha256", "iterations": 1000,
                                             "salt": "00" * 16, "hash": "x", "created_at": "t", "updated_at": "t"}]}), encoding="utf-8")
    rows = [r for r in mu.list_users() if r["fp"] != "-"]
    assert len(rows) == 1 and rows[0]["plan"] == "paid"                      # 一覧が落ちず、区分が無い記録は有料会員
    mu.set_plan(rows[0]["fp"], "paid")                                       # 保存すると平文の ID と name が消え、ID のキーだけになる
    raw = members.read_text(encoding="utf-8")
    assert "OldUser" not in raw and '"idk"' in raw and auth.member_record("OldUser") is not None
