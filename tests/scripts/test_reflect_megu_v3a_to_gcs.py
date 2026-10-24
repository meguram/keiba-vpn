import base64
import hashlib

import pytest

from src.scripts.data import reflect_megu_v3a_to_gcs as G


def _md5(b: bytes) -> str:
    return base64.b64encode(hashlib.md5(b).digest()).decode()


class _Blob:
    def __init__(self, bucket, name):
        self.bucket, self.name, self.chunk_size = bucket, name, None

    @property
    def md5_hash(self):
        data = self.bucket.store.get(self.name)
        return None if data is None else _md5(data)

    def upload_from_filename(self, path, **kw):
        self.bucket.store[self.name] = open(path, "rb").read()

    def reload(self):
        pass


class _Bucket:
    def __init__(self, store):
        self.store = dict(store)

    def get_blob(self, name):
        return _Blob(self, name) if name in self.store else None

    def blob(self, name):
        return _Blob(self, name)

    def copy_blob(self, blob, dest, new_name):
        dest.store[new_name] = self.store[blob.name]


def _tables(tmp_path, years, content=b"v3a"):
    for y in years:
        d = tmp_path / str(y)
        d.mkdir()
        (d / "megu_index_flat.parquet").write_bytes(content + str(y).encode())


def test_plan_year_cases():
    assert G.plan_year("L", "C", None) == {"backup": True, "upload": True, "backup_conflict": False}
    assert G.plan_year("L", "L", "C")["upload"] is False                 # すでに反映済み
    assert G.plan_year("L", None, None) == {"backup": False, "upload": True, "backup_conflict": False}
    assert G.plan_year("L", "C", "OTHER")["backup_conflict"] is True    # 退避先に別の内容


def test_dry_run_writes_nothing(tmp_path):
    _tables(tmp_path, [2020])
    b = _Bucket({G.BLOB_TPL.format(year=2020): b"old"})
    G.run_forward(b, [2020], tmp_path, execute=False)
    assert b.store == {G.BLOB_TPL.format(year=2020): b"old"}


def test_execute_backs_up_then_uploads_and_is_idempotent(tmp_path):
    _tables(tmp_path, [2020, 2021])
    b = _Bucket({G.BLOB_TPL.format(year=2020): b"old20", G.BLOB_TPL.format(year=2021): b"old21"})
    G.run_forward(b, [2020, 2021], tmp_path, execute=True)
    assert b.store[G.BACKUP_TPL.format(year=2020)] == b"old20" and b.store[G.BLOB_TPL.format(year=2021)] == b"v3a2021"
    again = G.run_forward(b, [2020, 2021], tmp_path, execute=True)
    assert all(r["done"] == [] for r in again)                          # 2 回目は何もしない
    assert b.store[G.BACKUP_TPL.format(year=2021)] == b"old21"          # 最初の退避は守られる


def test_rollback_restores_from_backup(tmp_path):
    _tables(tmp_path, [2020])
    b = _Bucket({G.BLOB_TPL.format(year=2020): b"old20"})
    G.run_forward(b, [2020], tmp_path, execute=True)
    assert b.store[G.BLOB_TPL.format(year=2020)] != b"old20"
    G.run_rollback(b, [2020], execute=True)
    assert b.store[G.BLOB_TPL.format(year=2020)] == b"old20"


def test_rollback_without_backup_stops(tmp_path):
    b = _Bucket({G.BLOB_TPL.format(year=2020): b"x"})
    with pytest.raises(SystemExit):
        G.run_rollback(b, [2020], execute=True)


def test_missing_local_file_stops(tmp_path):
    with pytest.raises(SystemExit):
        G.run_forward(_Bucket({}), [2020], tmp_path, execute=False)
