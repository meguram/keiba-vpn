import os
import time

from src.ops import gcs_mirror as gm


class Fake:
    def __init__(self):
        self.objs: dict[str, tuple[bytes, float]] = {}
        self.uploads = 0

    def list(self, prefix):
        return {k: (gm._md5(v), t) for k, (v, t) in self.objs.items() if k.startswith(prefix)}

    def download(self, key):
        return self.objs[key][0]

    def upload(self, key, data):
        self.uploads += 1
        self.objs[key] = (data, time.time())


TREES = ("data/local/inference/predictions_v3", "data/local/meta/entries_state")


def _w(root, rel, data=b"x"):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def test_push_uploads_only_new_or_changed_and_skips_tmp(tmp_path):
    g = Fake()
    _w(tmp_path, "data/local/inference/predictions_v3/20261024/R1.parquet", b"a")
    _w(tmp_path, "data/local/meta/entries_state/R1.json", b"s")
    _w(tmp_path, "data/local/meta/entries_state/R1.json.tmp", b"partial")
    assert gm.push(g, tmp_path, TREES)["uploaded"] == 2
    assert "ops/mirror/data/local/inference/predictions_v3/20261024/R1.parquet" in g.objs
    assert gm.push(g, tmp_path, TREES)["uploaded"] == 0                           # 同じ内容は再アップロードしない
    _w(tmp_path, "data/local/meta/entries_state/R1.json", b"s2")
    assert gm.push(g, tmp_path, TREES)["uploaded"] == 1 and g.uploads == 3
    assert gm.push(g, tmp_path, TREES, dry_run=True)["would_upload"] == 0


def test_push_recent_days_ignores_old_files(tmp_path):
    g = Fake()
    old = _w(tmp_path, "data/local/inference/predictions_v3/20260101/R0.parquet")
    os.utime(old, (time.time() - 10 * 86400,) * 2)
    _w(tmp_path, "data/local/inference/predictions_v3/20261024/R1.parquet")
    assert gm.push(g, tmp_path, TREES, recent_days=2)["uploaded"] == 1


def test_pull_recreates_the_tree_and_is_idempotent(tmp_path):
    g = Fake()
    src = tmp_path / "host"
    _w(src, "data/local/inference/predictions_v3/20261024/R1.parquet", b"a")
    _w(src, "data/local/inference/predictions_backfill/_shadow_models.json", b"{}")        # TREES に無い → 取り込まれない
    _w(src, "data/local/meta/entries_state/R1.json", b"s")
    gm.push(g, src, TREES)
    run = tmp_path / "cloudrun"
    assert gm.pull(g, run, TREES) == {"seen": 2, "downloaded": 2}
    assert (run / "data/local/inference/predictions_v3/20261024/R1.parquet").read_bytes() == b"a"
    assert not (run / "data/local/inference/predictions_backfill").exists()
    assert gm.pull(g, run, TREES)["downloaded"] == 0
    _w(src, "data/local/meta/entries_state/R1.json", b"s-new")                              # 出馬表の段階が進んだ
    gm.push(g, src, TREES)
    assert gm.pull(g, run, TREES, recent_days=1)["downloaded"] == 1
    assert (run / "data/local/meta/entries_state/R1.json").read_bytes() == b"s-new"
    assert not list(run.rglob("*.tmp"))


def test_pull_recent_days_skips_objects_not_updated_recently(tmp_path):
    g = Fake()
    g.objs["ops/mirror/data/local/meta/entries_state/OLD.json"] = (b"o", time.time() - 9 * 86400)
    g.objs["ops/mirror/data/local/meta/entries_state/NEW.json"] = (b"n", time.time())
    assert gm.pull(g, tmp_path, TREES, recent_days=2)["downloaded"] == 1
    assert (tmp_path / "data/local/meta/entries_state/NEW.json").exists() and not (tmp_path / "data/local/meta/entries_state/OLD.json").exists()
