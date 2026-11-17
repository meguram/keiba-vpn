"""会員名簿の正本を GCS に置く作り（KEIBA_MEMBERS_STORE=gcs）。世代の条件つきの書き込みで、どこから同時に書いても失わない。外部接続なし。"""

import json
import threading

import pytest

from src.api import auth
from src.ops import gcs_mirror
from src.scripts.ops import manage_users as mu

KEY = "ops/mirror/data/local/auth/members.json"


class FakeGcs:
    """世代つきの GCS の代わり。before_upload で、書く直前に別の書き手が割り込む状況を作れる。"""

    def __init__(self):
        self.objs: dict[str, tuple[bytes, int]] = {}
        self.lock = threading.Lock()
        self.before_upload = None
        self.conflicts = 0

    def read_gen(self, key):
        with self.lock:
            v = self.objs.get(key)
        return (None, 0) if v is None else v

    def upload_if(self, key, data, generation):
        if self.before_upload:
            hook, self.before_upload = self.before_upload, None
            hook()
        with self.lock:
            cur = self.objs.get(key, (None, 0))[1]
            if cur != generation:
                self.conflicts += 1
                return False
            self.objs[key] = (bytes(data), cur + 1)
            return True

    def upload(self, key, data):
        with self.lock:
            cur = self.objs.get(key, (None, 0))[1]
            self.objs[key] = (bytes(data), cur + 1)

    def list(self, prefix):
        import base64
        import hashlib

        self.listed = getattr(self, "listed", 0) + 1
        with self.lock:
            return {k: (base64.b64encode(hashlib.md5(v[0]).digest()).decode(), 0.0) for k, v in self.objs.items() if k.startswith(prefix)}

    def users(self):
        return json.loads(self.objs[KEY][0])["users"]


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "MEMBERS_LOCAL_FILE", str(tmp_path / "auth" / "members.json"))
    monkeypatch.setattr(auth, "members_refresh_hook", None)
    monkeypatch.setenv("DEV_SECRET_KEY", "test-secret")
    monkeypatch.delenv("KEIBA_MEMBERS_PEPPER", raising=False)
    monkeypatch.setenv("KEIBA_MEMBERS_STORE", "gcs")
    monkeypatch.setenv("GCS_BUCKET", "test-bucket")
    fake = FakeGcs()
    monkeypatch.setattr(mu, "_backend_factory", lambda: fake)
    auth._members_cache["key"] = None
    return fake


def test_write_goes_to_gcs_and_local_copy_and_index(store, tmp_path):
    _seed(store)
    mu.create_user("taro", "correct-horse-battery")
    assert len(store.users()) == 1
    assert auth.verify_member_login("taro", "correct-horse-battery")        # 手元の写しもすぐ更新されて照合できる
    assert gcs_mirror.index_key("auth") in store.objs                       # ほかのインスタンスが目録で拾える
    mu.set_enabled("taro", False)
    assert store.users()[0]["enabled"] is False and not auth.verify_member_login("taro", "correct-horse-battery")


def test_concurrent_writer_is_not_lost(store):
    _seed(store)
    mu.create_user("taro", "correct-horse-battery")

    def other_instance_writes():                                            # 書く直前に、別のインスタンス（決済の Webhook など）が先に書く
        raw, gen = store.read_gen(KEY)
        d = json.loads(raw)
        d["users"][0]["plan"] = "free"
        store.upload_if(KEY, json.dumps(d).encode(), gen)

    store.before_upload = other_instance_writes
    mu.create_user("hanako", "correct-horse-battery")
    assert store.conflicts == 1                                             # 衝突を検出して読み直した
    users = store.users()
    assert len(users) == 2 and users[0]["plan"] == "free"                   # 相手の変更も、自分の追加も残る


def test_many_threads_adding_users_all_survive(store):
    _seed(store)
    errs = []

    def add(i):
        try:
            mu.create_user(f"user{i:02d}", "correct-horse-battery")
        except Exception as e:  # noqa: BLE001
            errs.append(e)

    ts = [threading.Thread(target=add, args=(i,)) for i in range(5)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert errs == [] and len(store.users()) == 5


def test_no_master_in_gcs_refuses_to_write(store, monkeypatch):
    monkeypatch.delenv("KEIBA_MEMBERS_STORE")
    mu.create_user("oldtimer", "correct-horse-battery")                     # 移行の前（手元だけ）に作った会員
    monkeypatch.setenv("KEIBA_MEMBERS_STORE", "gcs")
    with pytest.raises(mu.UserError):                                       # 正本を初めて置くのはホストの cron だけ（空の名簿を正本にしない）
        mu.create_user("newbie", "correct-horse-battery")
    with pytest.raises(mu.UserError):
        mu.remove_pending(24)
    assert KEY not in store.objs


def _seed(store):
    store.objs[KEY] = (b'{"users": []}', 1)


def test_broken_gcs_roster_is_not_overwritten(store):
    store.objs[KEY] = (b"{not json", 7)
    with pytest.raises(mu.UserError):
        mu.create_user("taro", "correct-horse-battery")
    assert store.objs[KEY] == (b"{not json", 7)


def test_user_error_writes_nothing(store):
    _seed(store)
    mu.create_user("taro", "correct-horse-battery")
    gen = store.objs[KEY][1]
    with pytest.raises(mu.UserError):
        mu.create_user("taro", "correct-horse-battery")                     # 重複
    with pytest.raises(mu.UserError):
        mu.set_enabled("nobody", False)
    assert store.objs[KEY][1] == gen
    assert mu.remove_pending(24) == 0 and store.objs[KEY][1] == gen         # 変更が無ければ書かない


def test_host_cron_pulls_instead_of_pushing_when_gcs_is_master(store, tmp_path, monkeypatch):
    local = tmp_path / gcs_mirror.MEMBERS_REL
    local.parent.mkdir(parents=True)
    local.write_text('{"users": []}')
    assert "置かない" in gcs_mirror.pull_members_from_store(store, tmp_path)["members"]   # 空の名簿は正本にしない
    local.write_text('{"users": [{"idk": "host"}]}')
    r = gcs_mirror.pull_members_from_store(store, tmp_path)                 # 初回: GCS に無い → 手元を正本として置く
    assert "移行" in r["members"] and json.loads(store.objs[KEY][0])["users"][0]["idk"] == "host"
    store.objs[KEY] = (b'{"users": [{"idk": "cloudrun"}]}', 5)             # Cloud Run 側が書いた
    r = gcs_mirror.pull_members_from_store(store, tmp_path)
    assert r["members"] == "GCS から取り込んだ" and "cloudrun" in local.read_text()   # 上書きせず、取り込む
    reads = []
    real = store.read_gen
    store.read_gen = lambda k: reads.append(k) or real(k)
    assert gcs_mirror.pull_members_from_store(store, tmp_path)["members"] == "変更なし"
    assert reads == []                                                      # 変わっていなければ中身を読まない（GCP の外への転送を増やさない）
    assert oct(local.stat().st_mode & 0o777) == "0o600"
    assert "host" in local.with_name(local.name + ".bak").read_text()      # 上書きの前の控え
    store.objs[KEY] = (b"{broken", 6)
    assert gcs_mirror.pull_members_from_store(store, tmp_path).get("error") and "cloudrun" in local.read_text()   # 壊れた名簿で手元を上書きしない


def test_default_mode_stays_local_only(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "MEMBERS_LOCAL_FILE", str(tmp_path / "auth" / "members.json"))
    monkeypatch.setenv("DEV_SECRET_KEY", "test-secret")
    monkeypatch.delenv("KEIBA_MEMBERS_STORE", raising=False)
    monkeypatch.setattr(mu, "_backend_factory", lambda: (_ for _ in ()).throw(AssertionError("GCS に触れない")))
    auth._members_cache["key"] = None
    mu.create_user("taro", "correct-horse-battery")
    assert auth.verify_member_login("taro", "correct-horse-battery")
