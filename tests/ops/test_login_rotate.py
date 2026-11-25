"""共有ログインの毎月の切り替え（src/scripts/ops/login_rotate.py）。GCS・Secret Manager は偽物で、外部接続なし。"""

import json
from datetime import datetime

import pytest

from src.api import auth
from src.ops.secret_store import MemorySecretStore
from src.scripts.ops import login_rotate as lr
from src.scripts.ops import manage_users as mu
from src.scripts.ops import shared_login as legacy
from tests.api.test_members_gcs_store import KEY, FakeGcs

JST = lr.JST
ID = "megu-friends"


def at(y, m, d, h=6):
    return datetime(y, m, d, h, 0, tzinfo=JST)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "MEMBERS_LOCAL_FILE", str(tmp_path / "auth" / "members.json"))
    monkeypatch.setattr(auth, "members_refresh_hook", None)
    monkeypatch.setenv("DEV_SECRET_KEY", "test-secret")
    monkeypatch.delenv("KEIBA_MEMBERS_PEPPER", raising=False)
    monkeypatch.setenv("KEIBA_MEMBERS_STORE", "gcs")
    monkeypatch.setenv("GCS_BUCKET", "test-bucket")
    monkeypatch.setenv("KEIBA_SHARED_LOGIN_ID", ID)
    gcs = FakeGcs()
    gcs.objs[KEY] = (b'{"users": []}', 1)
    monkeypatch.setattr(mu, "_backend_factory", lambda: gcs)
    auth._members_cache["key"] = None
    sent: list[str] = []
    mu.create_user(ID, legacy.password_for("2026-12"))          # 12 月は旧方式（派生の値）で動いている
    return {"gcs": gcs, "sm": MemorySecretStore(), "sent": sent, "notify": lambda m: sent.append(m) or True}


def rec():
    return lr._read_record(ID)


def payload(sm):
    return lr._latest(sm)[1]


def no_secret_leaks(e, pws, extra=()):
    texts = list(e["sent"]) + list(extra)
    if lr.STATUS_KEY in e["gcs"].objs:
        texts.append(e["gcs"].objs[lr.STATUS_KEY][0].decode())
    banned = list(pws) + [lr.pw_mac(p) for p in pws] + [ID]       # 平文・名簿の HMAC・ID
    for t in texts:
        for w in banned:
            assert w not in t


def test_step_by_date():
    assert lr.step_for(at(2026, 11, 26)) == "prepare"
    assert lr.step_for(at(2026, 11, 27)) == "draft"          # 末日（30）の 3 日前
    assert lr.step_for(at(2026, 11, 30)) == "draft"
    assert lr.step_for(at(2026, 12, 25)) == "prepare"
    assert lr.step_for(at(2026, 12, 28)) == "draft"
    assert lr.step_for(at(2027, 2, 25)) == "draft"            # 2 月は準備の初日と同じ日
    assert lr.step_for(at(2027, 1, 1)) == "switch"
    assert lr.step_for(at(2026, 12, 10)) is None
    assert lr.add_months("2026-12", 1) == "2027-01" and lr.add_months("2027-01", -1) == "2026-12"


def test_full_month_cycle_from_legacy(env):
    sm, notify = env["sm"], env["notify"]
    old_pw = legacy.password_for("2026-12")
    assert lr.run(at(2026, 12, 25), sm=sm, notify=notify) == "created"
    nx = payload(sm)["next"]
    assert nx["month"] == "2027-01" and len(nx["password"]) == 19 and payload(sm)["current"] is None
    assert lr.run(at(2026, 12, 26), sm=sm, notify=notify) == "ok"          # 毎日やり直しても作り直さない
    assert payload(sm)["next"] == nx and env["sent"] == []
    assert lr.run(at(2026, 12, 28), sm=sm, notify=notify) == "drafted"
    assert lr.run(at(2026, 12, 29), sm=sm, notify=notify) == "ok" and len(env["sent"]) == 1   # 通知は 1 回
    assert auth.verify_member_login(ID, old_pw)                                # 下書きでは切り替わらない
    text = lr.render_draft(payload(sm), rec(), "2026-12")
    assert old_pw in text and nx["password"] in text and "1月1日 6:00" in text

    assert lr.run(at(2027, 1, 1), sm=sm, notify=notify) == "switched"
    assert auth.verify_member_login(ID, nx["password"]) and not auth.verify_member_login(ID, old_pw)
    r = rec()
    assert r["rotation_month"] == "2027-01" and not any(k in r for k in lr.FIELDS_NEXT)
    assert payload(sm) == {"current": {"month": "2027-01", "password": nx["password"]}, "next": None}
    assert len(sm.enabled_versions()) <= lr.KEEP_VERSIONS + 1
    assert lr.run(at(2027, 1, 1), sm=sm, notify=notify) == "ok"              # 二重の切り替えはしない
    assert auth.verify_member_login(ID, nx["password"]) and len(env["sent"]) == 2
    status = json.loads(env["gcs"].objs[lr.STATUS_KEY][0])
    assert status["step"] == "switch" and status["result"] == "ok" and status["month"] == "2027-01"
    no_secret_leaks(env, [old_pw, nx["password"]])

    # 2 回目の月（今の版が乱数の値）
    assert lr.run(at(2027, 1, 25), sm=sm, notify=notify) == "created"
    assert lr.run(at(2027, 1, 28), sm=sm, notify=notify) == "drafted"
    feb = payload(sm)["next"]["password"]
    text = lr.render_draft(payload(sm), rec(), "2027-01")
    assert nx["password"] in text and feb in text
    assert lr.run(at(2027, 2, 1), sm=sm, notify=notify) == "switched"
    assert auth.verify_member_login(ID, feb) and not auth.verify_member_login(ID, nx["password"])
    no_secret_leaks(env, [old_pw, nx["password"], feb])


def test_previous_month_guidance_uses_prev_mac(env):
    from src.api.flask_auth import _rotated_shared_password

    sm, notify = env["sm"], env["notify"]
    old_pw = legacy.password_for("2026-12")
    lr.run(at(2026, 12, 25), sm=sm, notify=notify)
    lr.run(at(2026, 12, 28), sm=sm, notify=notify)
    lr.run(at(2027, 1, 1), sm=sm, notify=notify)
    new_pw = payload(sm)["current"]["password"]
    assert _rotated_shared_password(ID, old_pw)
    assert not _rotated_shared_password(ID, new_pw)
    assert not _rotated_shared_password(ID, "wrong-password-xx")
    assert not _rotated_shared_password("nobody", old_pw)


def test_switch_without_next_does_not_change_password(env):
    sm, notify = env["sm"], env["notify"]
    old_pw = legacy.password_for("2026-12")
    with pytest.raises(lr.RotateError):
        lr.run(at(2027, 1, 1), sm=sm, notify=notify)
    assert auth.verify_member_login(ID, old_pw) and "rotation_month" not in rec()
    assert "失敗" in env["sent"][-1]
    assert json.loads(env["gcs"].objs[lr.STATUS_KEY][0])["result"] == "failed"


def test_next_is_not_regenerated_after_draft(env):
    sm, notify = env["sm"], env["notify"]
    lr.run(at(2026, 12, 25), sm=sm, notify=notify)
    lr.run(at(2026, 12, 28), sm=sm, notify=notify)
    mac = rec()["next_mac"]
    for v in list(sm.enabled_versions()):                       # Secret Manager の版が失われた
        sm.destroy(v)
    with pytest.raises(lr.RotateError):
        lr.run(at(2026, 12, 29), sm=sm, notify=notify)
    assert rec()["next_mac"] == mac and sm.enabled_versions() == []     # 作り直さない
    with pytest.raises(lr.RotateError):
        lr.run(at(2027, 1, 1), sm=sm, notify=notify)
    assert auth.verify_member_login(ID, legacy.password_for("2026-12"))


def test_latest_version_overwritten_is_repaired_from_older_version(env):
    sm, notify = env["sm"], env["notify"]
    lr.run(at(2026, 12, 25), sm=sm, notify=notify)
    good = payload(sm)["next"]
    sm.add(lr._dump_payload(None, {"month": "2027-01", "password": lr.generate_password()}))   # 別の実行が名簿に載らない版を足した
    assert lr.run(at(2026, 12, 26), sm=sm, notify=notify) == "repaired"
    assert payload(sm)["next"] == good


def test_guards_without_gcs_or_record(env, monkeypatch):
    sm, notify = env["sm"], env["notify"]
    monkeypatch.setenv("KEIBA_SHARED_LOGIN_ID", "someone-else")
    with pytest.raises(lr.RotateError):
        lr.run(at(2026, 12, 25), sm=sm, notify=notify)
    assert len(env["gcs"].users()) == 1 and sm.enabled_versions() == []   # 作らない
    monkeypatch.setenv("KEIBA_SHARED_LOGIN_ID", ID)
    monkeypatch.delenv("KEIBA_MEMBERS_STORE")
    with pytest.raises(lr.RotateError):
        lr.run(at(2026, 12, 25), sm=sm, notify=notify)
    assert sm.enabled_versions() == []


def test_rollback_restores_previous_hash(env):
    sm, notify = env["sm"], env["notify"]
    old_pw = legacy.password_for("2026-12")
    lr.run(at(2026, 12, 25), sm=sm, notify=notify)
    lr.run(at(2026, 12, 28), sm=sm, notify=notify)
    lr.run(at(2027, 1, 1), sm=sm, notify=notify)
    new_pw = payload(sm)["current"]["password"]
    assert lr.rollback(at(2027, 1, 1, 8), sm, ID, notify) == "rolled-back"
    assert auth.verify_member_login(ID, old_pw) and not auth.verify_member_login(ID, new_pw)
    assert rec()["rotation_month"] == "2026-12"
    with pytest.raises(lr.RotateError):                          # 次の版は消えているので、Job がもう一度動いても切り替わらない
        lr.run(at(2027, 1, 1, 9), sm=sm, notify=notify)
    assert auth.verify_member_login(ID, old_pw)


def test_legacy_ensure_does_not_touch_rotated_record(env, monkeypatch):
    sm, notify = env["sm"], env["notify"]
    lr.run(at(2026, 12, 25), sm=sm, notify=notify)
    lr.run(at(2026, 12, 28), sm=sm, notify=notify)
    lr.run(at(2027, 1, 1), sm=sm, notify=notify)
    new_pw = payload(sm)["current"]["password"]
    monkeypatch.setattr("src.utils.notify.notify_slack", lambda m, **k: True)
    assert legacy.ensure(ID, at(2027, 1, 1, 9)) == "managed"
    assert auth.verify_member_login(ID, new_pw)


def test_exported_pepper_keeps_same_keys(env, monkeypatch):
    k = auth.id_key(ID)
    pep = lr.export_pepper()
    assert pep.startswith("hex:")
    monkeypatch.setenv("KEIBA_MEMBERS_PEPPER", pep)
    assert auth.id_key(ID) == k


def test_switch_refuses_without_draft(env):
    sm, notify = env["sm"], env["notify"]
    lr.run(at(2026, 12, 25), sm=sm, notify=notify)                  # 準備だけ（下書きが出ていない＝note に値が出ていない）
    with pytest.raises(lr.RotateError):
        lr.run(at(2027, 1, 1), sm=sm, notify=notify)
    assert auth.verify_member_login(ID, legacy.password_for("2026-12"))


def test_draft_is_not_marked_when_slack_fails(env):
    sm, sent = env["sm"], env["sent"]
    lr.run(at(2026, 12, 25), sm=sm, notify=env["notify"])
    with pytest.raises(lr.RotateError):
        lr.run(at(2026, 12, 28), sm=sm, notify=lambda m: False)
    assert "draft_month" not in rec()
    assert lr.run(at(2026, 12, 29), sm=sm, notify=env["notify"]) == "drafted"     # 翌日にやり直す


def test_secret_sync_failure_after_switch_says_switched(env, monkeypatch):
    sm, notify = env["sm"], env["notify"]
    lr.run(at(2026, 12, 25), sm=sm, notify=notify)
    lr.run(at(2026, 12, 28), sm=sm, notify=notify)
    new_pw = payload(sm)["next"]["password"]
    real_add = sm.add
    monkeypatch.setattr(sm, "add", lambda data: (_ for _ in ()).throw(lr_secret_error()))
    with pytest.raises(lr.RotateError, match="切り替え済み"):
        lr.run(at(2027, 1, 1), sm=sm, notify=notify)
    assert auth.verify_member_login(ID, new_pw)
    assert any("切り替えました" in m for m in env["sent"]) and "切り替え済み" in env["sent"][-1]
    monkeypatch.setattr(sm, "add", real_add)
    assert lr.run(at(2027, 1, 1, 7), sm=sm, notify=notify) == "ok"           # もう一度流せば追いつく
    assert payload(sm) == {"current": {"month": "2027-01", "password": new_pw}, "next": None}


def lr_secret_error():
    from src.ops.secret_store import SecretError

    return SecretError("Secret Manager の版の追加に失敗しました（HTTP 503）")


def test_rollback_then_next_month_cycle(env):
    sm, notify = env["sm"], env["notify"]
    old_pw = legacy.password_for("2026-12")
    lr.run(at(2026, 12, 25), sm=sm, notify=notify)
    lr.run(at(2026, 12, 28), sm=sm, notify=notify)
    lr.run(at(2027, 1, 1), sm=sm, notify=notify)
    lr.rollback(at(2027, 1, 1, 8), sm, ID, notify)
    assert lr.run(at(2027, 1, 25), sm=sm, notify=notify) == "created"
    assert lr.run(at(2027, 1, 28), sm=sm, notify=notify) == "drafted"
    feb = payload(sm)["next"]["password"]
    text = lr.render_draft(payload(sm), rec(), "2027-01")
    assert old_pw in text and feb in text                                     # 戻した後も、今の値（先月の値）を表示できる
    assert lr.run(at(2027, 2, 1), sm=sm, notify=notify) == "switched"
    assert auth.verify_member_login(ID, feb) and not auth.verify_member_login(ID, old_pw)


def test_concurrent_prepare_keeps_roster_value(env):
    sm, notify = env["sm"], env["notify"]
    gcs = env["gcs"]

    def other_run_prepares():                                   # 名簿に書く直前に、別の実行が別の次の版を書いた
        other = lr.generate_password()
        sm.add(lr._dump_payload(None, {"month": "2027-01", "password": other}))
        raw, gen = gcs.read_gen(KEY)
        d = json.loads(raw)
        d["users"][0].update({"next_month": "2027-01", "next_mac": lr.pw_mac(other)})
        gcs.upload_if(KEY, json.dumps(d).encode(), gen)
        env["other"] = other

    gcs.before_upload = other_run_prepares
    assert lr.run(at(2026, 12, 25), sm=sm, notify=notify) == "repaired"
    assert payload(sm)["next"]["password"] == env["other"] and rec()["next_mac"] == lr.pw_mac(env["other"])


def test_cli_output_has_no_secrets(env, monkeypatch, capsys):
    sm = env["sm"]
    monkeypatch.setattr(lr, "_store", lambda: sm)
    monkeypatch.setattr(lr, "_notify", env["notify"])
    assert lr.main(["run", "--now", "2026-12-25T06:00"]) == 0
    assert lr.main(["run", "--now", "2026-12-28T06:00"]) == 0
    assert lr.main(["run", "--now", "2027-01-01T06:00"]) == 0
    out = capsys.readouterr()
    no_secret_leaks(env, [legacy.password_for("2026-12"), payload(sm)["current"]["password"]], extra=[out.out, out.err])


class _Resp:
    def __init__(self, code, body=None):
        self.status_code, self._b = code, body
        self.content = b"x" if body is not None else b""

    def json(self):
        return self._b


class _Session:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def request(self, method, url, timeout=None, **kw):
        self.calls.append((method, url, kw))
        for (m, suffix), fn in self.routes.items():
            if m == method and url.endswith(suffix):
                return fn(kw)
        return _Resp(404, {})


def test_gcp_secret_store_rest_calls():
    import base64

    from src.ops.secret_store import GcpSecretStore, SecretError

    st = GcpSecretStore("KEIBA_SHARED_LOGIN_PW", project="p")
    pages = iter([{"versions": [{"name": "projects/p/secrets/S/versions/2"}], "nextPageToken": "t"},
                  {"versions": [{"name": "projects/p/secrets/S/versions/10"}]}])
    st._session = _Session({
        ("GET", "/versions/latest:access"): lambda kw: _Resp(200, {"name": "projects/p/secrets/S/versions/10", "payload": {"data": base64.b64encode(b"{}").decode()}}),
        ("GET", "/versions"): lambda kw: _Resp(200, next(pages)),
        ("POST", ":addVersion"): lambda kw: _Resp(200, {"name": "projects/p/secrets/S/versions/11"}),
        ("POST", "/versions/2:destroy"): lambda kw: _Resp(200, {}),
    })
    assert st.latest() == ("projects/p/secrets/S/versions/10", b"{}")
    assert st.enabled_versions() == ["projects/p/secrets/S/versions/10", "projects/p/secrets/S/versions/2"]   # ページ送り・新しい順
    assert st.add(b"{}").endswith("/11")
    st.destroy("projects/p/secrets/S/versions/2")
    st._session = _Session({("GET", "/versions"): lambda kw: _Resp(200, {})})
    assert st.latest() is None                                          # 版が無い秘密
    st._session = _Session({("GET", "/versions/latest:access"): lambda kw: _Resp(403, {"error": "denied"})})
    with pytest.raises(SecretError) as e:
        st.latest()
    assert "403" in str(e.value)
