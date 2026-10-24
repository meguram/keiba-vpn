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


def test_refresher_pulls_everything_then_follows_new_files(tmp_path):
    g = Fake()
    g.objs["ops/mirror/data/local/inference/tracking_v2/2026-10-24/R1.json"] = (b"1", time.time())
    g.objs["ops/mirror/data/local/inference/snapshots/20261017/R0.json"] = (b"0", time.time() - 30 * 86400)     # 古い分も全体の取り込みで届く
    gm.start_refresher(g, tmp_path, interval=0.05, recent_days=1)

    def wait(path):
        for _ in range(100):
            if (tmp_path / path).exists():
                return True
            time.sleep(0.05)
        return False

    assert wait("data/local/inference/tracking_v2/2026-10-24/R1.json") and wait("data/local/inference/snapshots/20261017/R0.json")
    g.objs["ops/mirror/data/local/meta/entries_state/R9.json"] = (b"new", time.time())               # 後から増えた分を追いかける
    assert wait("data/local/meta/entries_state/R9.json")


def test_storage_client_uses_env_service_account_then_adc(monkeypatch):
    import sys
    import types

    from src.config import gcp_credentials as gc

    seen = {}
    fake = types.SimpleNamespace(Client=lambda **kw: seen.update(kw) or object())
    monkeypatch.setitem(sys.modules, "google.cloud.storage", fake)
    import google.cloud as gcloud
    monkeypatch.setattr(gcloud, "storage", fake, raising=False)
    monkeypatch.setattr("src.config.gcp_guard.gcp_forbidden", lambda: False)
    monkeypatch.setenv("GCS_PROJECT_ID", "proj")
    monkeypatch.delenv("GCS_PRIVATE_KEY", raising=False)
    gc.storage_client()
    assert seen == {"project": "proj", "credentials": None}          # 鍵が無ければ ADC（Cloud Run の実行サービスアカウント）


def test_tree_with_glob_only_mirrors_matching_files(tmp_path):
    g = Fake()
    trees = ("data/page_reference/tables::*/race_result_flat.parquet",)
    _w(tmp_path, "data/page_reference/tables/2025/race_result_flat.parquet", b"keep")
    _w(tmp_path, "data/page_reference/tables/2025/race_odds_flat.parquet", b"skip")        # 巨大な他の表は対象外
    _w(tmp_path, "data/page_reference/tables/2025/_chunks_race_result/part.parquet", b"skip")
    assert gm.push(g, tmp_path, trees)["uploaded"] == 1
    assert list(g.objs) == ["ops/mirror/data/page_reference/tables/2025/race_result_flat.parquet"]
    run = tmp_path / "cloudrun"
    assert gm.pull(g, run, trees) == {"seen": 1, "downloaded": 1}
    assert gm.split_tree("a/b") == ("a/b", "**/*")


def test_default_trees_cover_the_data_the_api_reads():
    dirs = {gm.split_tree(t)[0] for t in gm.TREES}
    for need in ("data/page_reference/race_lists", "data/page_reference/tables", "data/local/inference/predictions_v3", "data/local/meta/entries_state"):
        assert need in dirs


def test_tracking_difficulty_is_pushed_but_not_bulk_pulled(tmp_path):
    """追走難度は全体で約 1GB。GCS には全部置くが、Cloud Run の起動時（TREES の全体取り込み）には載せない。レース ID のファイルだけを上げる。"""
    d = "data/calculated_data/tracking_difficulty"
    assert any(t.startswith(d) for t in gm.LAZY_TREES) and not any(t.startswith(d) for t in gm.TREES) and set(gm.LAZY_TREES) <= set(gm.PUSH_TREES)
    src = tmp_path / "src"
    _w(src, f"{d}/202608040901.json", b"{}")
    _w(src, f"{d}/_index_meta.json", b"{}")
    g = Fake()
    assert gm.push(g, src, gm.LAZY_TREES)["uploaded"] == 1
    run = tmp_path / "run"
    assert gm.pull(g, run, TREES)["downloaded"] == 0                       # 起動時の全体取り込みには含まれない
    assert gm.fetch_one(f"{d}/202608040901.json", run, g) is True           # 読まれたときに 1 件だけ取る
    assert gm.fetch_one(f"{d}/202608040901.json", run, g, max_age=600.0) is False       # 取って間もないときは取り直さない


def test_push_flushes_in_batches_so_memory_stays_bounded(tmp_path):
    for i in range(10):
        _w(tmp_path, f"data/local/meta/entries_state/R{i}.json", b"x" * 100)
    g = Fake()
    assert gm.push(g, tmp_path, ("data/local/meta/entries_state",), batch_bytes=250)["uploaded"] == 10


def test_fetch_one_gets_a_single_file_and_remembers_misses(tmp_path, monkeypatch):
    g = Fake()
    g.objs["ops/mirror/data/x/R1.json"] = (b"{}", time.time())
    assert gm.fetch_one("data/x/R1.json", tmp_path, g) is True and (tmp_path / "data/x/R1.json").read_bytes() == b"{}"
    gm._MISS.clear()
    assert gm.fetch_one("data/x/NONE.json", tmp_path, g) is False
    g.objs["ops/mirror/data/x/NONE.json"] = (b"{}", time.time())
    assert gm.fetch_one("data/x/NONE.json", tmp_path, g) is False          # 直前に無かったものは、短時間は問い合わせ直さない
    monkeypatch.delenv("KEIBA_MIRROR_PULL", raising=False)
    assert gm.fetch_one("data/x/R2.json", tmp_path) is False               # ミラー無効（ホスト・dev・テスト）では何もしない
