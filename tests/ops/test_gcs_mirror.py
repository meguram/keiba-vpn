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


def test_push_tells_the_cache_warmer_only_about_files_it_uploaded(tmp_path, monkeypatch):
    from src.utils import cache_marker

    got: list[list[str]] = []
    monkeypatch.setattr(cache_marker, "mark_from_paths", lambda rels: got.append(list(rels)) or len(rels))
    g = Fake()
    _w(tmp_path, "data/local/inference/predictions_v3/20261024/R1.parquet", b"a")
    gm.push(g, tmp_path, TREES)
    assert got == [["data/local/inference/predictions_v3/20261024/R1.parquet"]]
    gm.push(g, tmp_path, TREES)                                                   # 変化なし → 目印は出さない
    gm.push(g, tmp_path, TREES, dry_run=True)
    assert len(got) == 1


def test_push_survives_marker_failure(tmp_path, monkeypatch):
    from src.utils import cache_marker

    def boom(rels):
        raise RuntimeError("gcs down")

    monkeypatch.setattr(cache_marker, "mark_from_paths", boom)
    _w(tmp_path, "data/local/inference/predictions_v3/20261024/R1.parquet", b"a")
    assert gm.push(Fake(), tmp_path, TREES)["uploaded"] == 1


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


def test_pull_calls_download_hooks_only_when_something_was_downloaded(tmp_path, monkeypatch):
    monkeypatch.setattr(gm, "_DOWNLOAD_HOOKS", [])
    seen: list[list[str]] = []
    gm.on_download(seen.append)
    gm.on_download(seen.append)                                   # 二重登録しない
    gm.on_download(lambda keys: 1 / 0)                            # 失敗しても取り込みは止めない
    g = Fake()
    src = tmp_path / "host"
    _w(src, "data/local/meta/entries_state/R1.json", b"s")
    gm.push(g, src, TREES)
    run = tmp_path / "cloudrun"
    assert gm.pull(g, run, TREES)["downloaded"] == 1
    assert seen == [[f"{gm.PREFIX}/data/local/meta/entries_state/R1.json"]]
    assert gm.pull(g, run, TREES)["downloaded"] == 0
    assert len(seen) == 1


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


def test_members_file_is_pulled_to_the_path_auth_reads_and_unreadable_dirs_are_skipped(tmp_path, monkeypatch):
    """Cloud Run は会員名簿を、auth.py の MEMBERS_LOCAL_FILE と同じ場所へ取り込む。root だけが読める場所は、通常の実行者の同期では飛ばす（落ちない）。"""
    from src.api import auth

    assert gm.AUTH_TREES[0].split("::")[0] == "data/local/auth" and set(gm.AUTH_TREES) <= set(gm.TREES)
    assert auth.MEMBERS_LOCAL_FILE.replace("\\", "/").endswith("data/local/auth/members.json") or "auth/members.json" in str(auth.MEMBERS_LOCAL_FILE).replace("..", "")
    src = tmp_path / "src"
    _w(src, "data/local/auth/members.json", b'{"users": []}')
    g = Fake()
    assert gm.push(g, src, gm.AUTH_TREES)["uploaded"] == 1
    run = tmp_path / "run"
    assert gm.pull(g, run, gm.AUTH_TREES)["downloaded"] == 1 and (run / "data/local/auth/members.json").read_bytes() == b'{"users": []}'

    from pathlib import Path
    real_glob = Path.glob

    def deny(self, pattern):
        if "auth" in str(self):
            raise PermissionError("root only")
        return real_glob(self, pattern)

    monkeypatch.setattr(Path, "glob", deny)
    assert gm.push(Fake(), src, gm.AUTH_TREES) == {"seen": 0, "uploaded": 0, "would_upload": 0}


def test_jockey_trainer_lookups_are_mirrored_for_the_entry_card(tmp_path):
    """出馬表の騎手・調教師カードの集計（lookup）が Cloud Run に届く（無いと『この騎手の集計がありません』になる）。"""
    names = ["jockey_tbl/lookup_jockey.parquet", "jockey_tbl/lookup_jockey_slices.parquet", "trainer_tbl/lookup_trainer.parquet",
             "trainer_tbl/lookup_trainer_slices.parquet", "jockey_trainer_stats/_manifest.json"]
    for n in names:
        _w(tmp_path, "data/local/features/" + n, b"x")
    _w(tmp_path, "data/local/features/jockey_trainer_stats/_merge_spec.json", b"x")
    got = {str(p.relative_to(tmp_path / "data/local/features")) for t in gm.PUSH_TREES for p in gm._files(tmp_path, t) if "features" in str(p)}
    assert got == set(names)


# ── リクエスト課金（CPU はリクエスト中だけ）向け: 常駐スレッドの代わりに、リクエストをきっかけに取り込む ──
class Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t


def test_start_initial_pulls_recent_then_full_once_and_sets_events(tmp_path):
    gm.initial_recent_pulled.clear(); gm.initial_full_pulled.clear()
    g = Fake()
    g.objs["ops/mirror/data/local/inference/snapshots/20261017/R0.json"] = (b"0", time.time() - 30 * 86400)
    gm.start_initial(g, tmp_path, recent_days=1).join(5)
    assert gm.initial_recent_pulled.is_set() and gm.initial_full_pulled.is_set()
    assert (tmp_path / "data/local/inference/snapshots/20261017/R0.json").exists()          # 全体の取り込みまで済んでいる


def test_start_initial_failure_still_releases_waiters(tmp_path):
    gm.initial_recent_pulled.clear(); gm.initial_full_pulled.clear()
    class Boom(Fake):
        def list(self, prefix): raise RuntimeError("gcs down")
    gm.start_initial(Boom(), tmp_path).join(5)
    assert gm.initial_recent_pulled.is_set() and gm.initial_full_pulled.is_set()          # 起動を止めない


def test_request_refresher_runs_only_when_due_and_not_concurrently(tmp_path):
    clk, g = Clock(), Fake()
    r = gm.RequestRefresher(g, tmp_path, interval=60, wait_gap=300, clock=clk)
    assert r.start_if_due() is None                                    # 起動直後は間隔未満
    clk.t += 61
    g.objs["ops/mirror/data/local/meta/entries_state/R1.json"] = (b"1", time.time())
    t = r.start_if_due(); assert t is not None
    t.join(5); assert (tmp_path / "data/local/meta/entries_state/R1.json").exists()
    assert r.start_if_due() is None                                    # 取り込んだ直後はまた間隔未満
    clk.t += 61
    assert r.start_if_due() is not None


def test_request_refresher_does_not_start_second_pull_while_running(tmp_path):
    import threading

    gate, started = threading.Event(), threading.Event()
    class Slow(Fake):
        def list(self, prefix): started.set(); gate.wait(5); return super().list(prefix)
    clk = Clock(); r = gm.RequestRefresher(Slow(), tmp_path, interval=10, clock=clk)
    clk.t += 11
    t = r.start_if_due(); started.wait(5)
    clk.t += 11
    assert r.start_if_due() is None                                    # 実行中は重ねない
    gate.set(); t.join(5)


def test_request_refresher_waits_only_after_a_long_gap_and_survives_failure(tmp_path):
    clk = Clock(); r = gm.RequestRefresher(Fake(), tmp_path, interval=60, wait_gap=300, clock=clk)
    assert not r.should_wait()
    clk.t += 301
    assert r.should_wait()                                             # 長く止まっていた後の最初のリクエストは、取り込みを待つ
    r.start_if_due().join(5)
    assert not r.should_wait()
    class Boom(Fake):
        def list(self, prefix): raise RuntimeError("down")
    clk.t += 400
    r2 = gm.RequestRefresher(Boom(), tmp_path, interval=60, clock=clk); clk.t += 61
    r2.start_if_due().join(5)                                          # 失敗しても例外を出さず、次回に再試行できる
    clk.t += 61
    assert r2.start_if_due() is not None


def _day(offset):
    from datetime import date, timedelta

    return (date.today() - timedelta(days=offset)).strftime("%Y%m%d")


def test_pull_window_skips_old_days_in_windowed_dirs_only(tmp_path):
    g = Fake()
    src = tmp_path / "host"
    new, old = _day(3), _day(40)
    bf = "data/local/inference/predictions_backfill"
    _w(src, f"{bf}/bundle_v6_2025/{new}/R1.parquet")
    _w(src, f"{bf}/bundle_v6_2025/{old}/R0.parquet")
    _w(src, f"{bf}/_shadow_models.json", b"{}")                                        # 日付の無いファイルは常に降ろす
    _w(src, f"data/local/inference/lap_forecast/{old[:4]}-{old[4:6]}-{old[6:]}/R0.json")   # ハイフン区切りの日付
    _w(src, f"data/local/inference/predictions_v3/{old}/R0.parquet")                 # 窓の対象外の木は古くても降ろす
    trees = (bf, "data/local/inference/lap_forecast", "data/local/inference/predictions_v3")
    gm.push(g, src, trees)
    dst = tmp_path / "cloud"
    gm.pull(g, dst, trees, window=14.0)
    assert (dst / f"{bf}/bundle_v6_2025/{new}/R1.parquet").is_file()
    assert (dst / f"{bf}/_shadow_models.json").is_file()
    assert (dst / f"data/local/inference/predictions_v3/{old}/R0.parquet").is_file()
    assert not (dst / f"{bf}/bundle_v6_2025/{old}/R0.parquet").exists()
    assert not list((dst / "data/local/inference/lap_forecast").glob("*/*"))
    gm.pull(g, dst, trees)                                                             # window なし（Jobs・CLI）は従来どおり全部
    assert (dst / f"{bf}/bundle_v6_2025/{old}/R0.parquet").is_file()


def test_lazy_fetch_pulls_only_outside_the_window(tmp_path, monkeypatch):
    g = Fake()
    rel_old = f"data/local/inference/predictions_backfill/bundle_v6_2025/{_day(40)}/R0.parquet"
    rel_new = f"data/local/inference/predictions_backfill/bundle_v6_2025/{_day(2)}/R1.parquet"
    for r in (rel_old, rel_new):
        g.objs[f"{gm.PREFIX}/{r}"] = (b"p", time.time())
    real = gm.fetch_one
    monkeypatch.setattr(gm, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(gm, "fetch_one", lambda rel, **kw: real(rel, root=tmp_path, backend=g, **kw))
    gm._MISS.clear()
    assert gm.lazy_fetch(tmp_path / rel_old, _day(40)) is True and (tmp_path / rel_old).is_file()
    assert gm.lazy_fetch(tmp_path / rel_new, _day(2)) is False                         # 窓の内側は起動時の取り込みの担当。ここでは取らない
    assert not (tmp_path / rel_new).exists()
    assert gm.lazy_fetch(tmp_path / rel_old, _day(40)) is True                         # 既にあれば取らずに True
    assert gm.lazy_fetch(tmp_path.parent / "outside.json", _day(40)) is False          # リポジトリ外
    assert gm.lazy_fetch(tmp_path / "data/none.json", "unreadable") is False            # 日付が読めない → 窓の内側扱い


def test_lazy_fetch_is_noop_when_mirror_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("KEIBA_MIRROR_PULL", raising=False)
    monkeypatch.setattr(gm, "REPO_ROOT", tmp_path)
    assert gm.lazy_fetch(tmp_path / f"data/local/inference/lap_forecast/2020-01-01/R.json", "2020-01-01") is False


def test_old_day_readers_fall_back_to_lazy_fetch(tmp_path, monkeypatch):
    from src.api import local_predictions as lp
    from src.pipeline.inference import lap_forecast_service as lfs

    calls = []
    monkeypatch.setattr(lp, "lazy_fetch", lambda p, d: calls.append((p.parent.parent.name, d)) or p.is_file())
    inf = tmp_path / "inf"
    inf.mkdir()
    monkeypatch.setenv("KEIBA_PRED_MODEL", "v6e")
    assert lp.find_local_prediction("R9", "20200105", inf) is None
    assert [c[0] for c in calls] == ["bundle_v6_2025", "bundle_v4_2025", "bundle_v5_2025"]
    f = inf / "predictions_backfill/bundle_v6_2025/20200105/R9.parquet"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"x")
    assert lp.find_local_prediction("R9", "20200105", inf)[1:] == ("backfill", "bundle_v6_2025")

    got = []
    import src.ops.gcs_mirror as gmod

    monkeypatch.setattr(gmod, "lazy_fetch", lambda p, d: got.append(d) or False)
    monkeypatch.setattr(lfs, "path_for", lambda rid, d: tmp_path / "nope.json")
    assert lfs.load_forecast("R9", "2020-01-05") is None and got == ["2020-01-05"]


class CountingFake(Fake):
    def __init__(self):
        super().__init__()
        self.lists = 0

    def list(self, prefix):
        self.lists += 1
        return super().list(prefix)


def test_push_with_manifest_lists_gcs_only_once_then_compares_locally(tmp_path):
    """費用の上限: GCP の外から 5 分おきに GCS 全件を一覧すると転送料が月数千円になる。記録があれば一覧しない。"""
    g = CountingFake()
    man = tmp_path / "manifest.json"
    _w(tmp_path, "data/local/meta/entries_state/R1.json", b"s")
    assert gm.push(g, tmp_path, TREES, manifest=man)["uploaded"] == 1
    first_lists = g.lists
    assert first_lists == len(TREES) and man.is_file()
    assert gm.push(g, tmp_path, TREES, manifest=man)["uploaded"] == 0          # 同じ内容は上げない
    _w(tmp_path, "data/local/meta/entries_state/R1.json", b"s2")
    _w(tmp_path, "data/local/meta/entries_state/R2.json", b"n")
    assert gm.push(g, tmp_path, TREES, manifest=man)["uploaded"] == 2          # 変更・新規は上げる
    assert g.lists == first_lists                                              # 2 回目以降は一覧しない
    assert g.objs["ops/mirror/data/local/meta/entries_state/R1.json"][0] == b"s2"


def test_push_manifest_is_seeded_from_gcs_so_existing_files_are_not_reuploaded(tmp_path):
    g = CountingFake()
    _w(tmp_path, "data/local/meta/entries_state/R1.json", b"s")
    gm.push(g, tmp_path, TREES)                                                # 記録なしで上げた後に
    uploads = g.uploads
    man = tmp_path / "manifest.json"
    assert gm.push(g, tmp_path, TREES, manifest=man)["uploaded"] == 0          # 初回は一覧から記録を作る
    assert g.uploads == uploads


def test_push_verify_remote_and_broken_manifest_fall_back_to_listing(tmp_path):
    g = CountingFake()
    man = tmp_path / "manifest.json"
    _w(tmp_path, "data/local/meta/entries_state/R1.json", b"s")
    gm.push(g, tmp_path, TREES, manifest=man)
    del g.objs["ops/mirror/data/local/meta/entries_state/R1.json"]           # GCS 側で消えた
    assert gm.push(g, tmp_path, TREES, manifest=man)["uploaded"] == 0          # 記録だけでは気づかない
    assert gm.push(g, tmp_path, TREES, manifest=man, verify_remote=True)["uploaded"] == 1
    man.write_text("{broken")
    n = g.lists
    gm.push(g, tmp_path, TREES, manifest=man)
    assert g.lists == n + len(TREES)


def test_push_dry_run_does_not_write_manifest(tmp_path):
    g = CountingFake()
    man = tmp_path / "manifest.json"
    _w(tmp_path, "data/local/meta/entries_state/R1.json", b"s")
    assert gm.push(g, tmp_path, TREES, manifest=man, dry_run=True)["would_upload"] == 1
    assert not man.exists()


class GenFake(CountingFake):
    """世代つきの読み書き（Jobs の目録の追記）も扱える偽の GCS。"""

    def __init__(self):
        super().__init__()
        self.gen: dict[str, int] = {}
        self.downloads = 0

    def download(self, key):
        self.downloads += 1
        if key not in self.objs:
            raise FileNotFoundError(key)
        return super().download(key)

    def upload(self, key, data):
        super().upload(key, data)
        self.gen[key] = self.gen.get(key, 0) + 1

    def read_gen(self, key):
        return (self.objs[key][0], self.gen[key]) if key in self.objs else (None, 0)

    def upload_if(self, key, data, generation):
        if self.gen.get(key, 0) != generation:
            return False
        self.upload(key, data)
        return True


def test_host_push_writes_the_index_and_api_refresh_reads_it_without_listing(tmp_path):
    """費用の上限: API は毎分 24 か所を一覧せず、ホストの push が置く目録 1 つだけを読む。"""
    g = GenFake()
    host, api = tmp_path / "host", tmp_path / "api"
    man = tmp_path / "manifest.json"
    _w(host, "data/local/meta/entries_state/R1.json", b"s")
    gm.push(g, host, TREES, manifest=man, index_writer="host")
    assert gm.index_key("host") in g.objs
    lists = g.lists
    r = gm.pull_via_index(g, api, TREES, recent_days=2.0)
    assert r["downloaded"] == 1 and (api / "data/local/meta/entries_state/R1.json").read_bytes() == b"s"
    assert g.lists == lists                                                    # 一覧していない
    _w(host, "data/local/meta/entries_state/R1.json", b"s2")
    gm.push(g, host, TREES, manifest=man, index_writer="host")
    assert gm.pull_via_index(g, api, TREES, recent_days=2.0)["downloaded"] == 1
    assert (api / "data/local/meta/entries_state/R1.json").read_bytes() == b"s2"
    assert gm.pull_via_index(g, api, TREES, recent_days=2.0)["downloaded"] == 0  # 同じ内容は取り直さない


def test_index_is_unusable_when_missing_or_stale_so_refresh_falls_back_to_listing(tmp_path):
    g = GenFake()
    assert gm.pull_via_index(g, tmp_path, TREES) is None                       # 目録が無い
    man = tmp_path / "m.json"
    _w(tmp_path / "h", "data/local/meta/entries_state/R1.json", b"s")
    gm.push(g, tmp_path / "h", TREES, manifest=man, index_writer="host")
    assert gm.pull_via_index(g, tmp_path / "a", TREES, now=time.time() + gm.INDEX_STALE_SEC + 60) is None   # push が止まっている
    t = [0.0]
    ref = gm.Refresh(clock=lambda: t[0])
    n = g.lists
    assert ref(g, tmp_path / "a", recent_days=2.0, window=None)["via"] == "index" and g.lists == n
    t[0] = gm.FULL_LIST_SEC + 1                                                # 一定間隔ごとには一覧で取りこぼしを拾う
    assert ref(g, tmp_path / "a", recent_days=2.0, window=None)["via"] == "list" and g.lists > n


def test_jobs_push_appends_to_its_index_without_losing_concurrent_entries(tmp_path):
    g = GenFake()
    _w(tmp_path / "j1", "data/local/meta/entries_state/R1.json", b"a")
    _w(tmp_path / "j2", "data/local/meta/entries_state/R2.json", b"b")
    gm.push(g, tmp_path / "j1", TREES, index_writer="jobs")
    gm.push(g, tmp_path / "j2", TREES, index_writer="jobs")
    import json as _json

    entries = _json.loads(g.objs[gm.index_key("jobs")][0])["entries"]
    assert set(entries) == {"ops/mirror/data/local/meta/entries_state/R1.json", "ops/mirror/data/local/meta/entries_state/R2.json"}


def test_index_drops_entries_older_than_the_refresh_window():
    now = time.time()
    kept = gm._prune({"a": ["x", now - 10], "b": ["y", now - gm.INDEX_KEEP_SEC - 10]}, now)
    assert set(kept) == {"a"}
