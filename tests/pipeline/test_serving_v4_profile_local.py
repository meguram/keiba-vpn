"""新しい馬の補い（serving_v4.V4Features.profile_for）を手元優先にしたことの確認。

- 手元の profile_dyn（<state_dir> と本体の置き場）にある馬は GCS（storage.load）を読まない。
- warm の後に profile_dyn が書き足されたら（precompute の書き戻し）、その馬も読まない。
- profile_for は書かない。書き戻しは update_serving_state（precompute）だけで、取れた馬だけを足す。
- KEIBA_PROFILE_LOCAL=0・目印のファイルで今までの動き（毎回 GCS）に戻る。
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

from src.pipeline.features import serving_v4 as sv
from src.scripts.data import update_serving_state as uss


class FakeStorage:
    gcs_enabled = True

    def __init__(self, docs: dict[tuple[str, str], dict] | None = None, fail: bool = False):
        self.docs = docs or {}
        self.calls: list[tuple[str, str]] = []
        self.fail = fail

    @property
    def _gcs_healthy(self) -> bool:
        return not self.fail

    def load(self, category: str, key: str):
        self.calls.append((category, key))
        if self.fail:
            return None                                  # HybridStorage は GCS の不調でも None を返す
        return self.docs.get((category, key))


def _features(state_dir: Path) -> sv.V4Features:
    tables = state_dir.parent / "tables"
    (tables / "2025").mkdir(parents=True, exist_ok=True)   # 履歴は読まない（年の一覧だけ要る）
    f = sv.V4Features(tables_dir=tables, state_dir=state_dir)
    f.hist.profile = pd.DataFrame({"horse_id": ["2020100001"], "owner": ["A"], "breeder": ["B"], "birthplace": ["日高"], "birthday": [pd.NaT]})
    f.hist.attrs = pd.DataFrame({"birthplace": ["日高"], "birthday": [pd.Timestamp("2020-03-01")]}, index=pd.Index(["2020100001"], name="horse_id"))
    return f


def _dyn(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    own, main = tmp_path / "own_state", tmp_path / "main_state"
    own.mkdir()
    main.mkdir()
    monkeypatch.setenv("KEIBA_PROFILE_LOCAL_DIR", str(main))
    monkeypatch.delenv("KEIBA_PROFILE_LOCAL", raising=False)
    monkeypatch.setattr(sv, "ROOT", tmp_path)            # 目印のファイル（data/local/meta/profile_local.off）を一時の場所で見る
    return own, main


def test_local_rows_skip_gcs(dirs):
    own, main = dirs
    _dyn(own / "profile_dyn.parquet", [{"horse_id": "2024100001", "owner": "O1", "breeder": "B1", "birthplace": "浦河", "birthday": "2024年2月2日"}])
    _dyn(main / "profile_dyn.parquet", [{"horse_id": "2024100002", "owner": "O2", "breeder": "B2", "birthplace": "新冠", "birthday": "2024年3月3日",
                                         "birthday_x": pd.NaT, "bp_retry": True}])
    f = _features(own)
    f._absorb_profile_dyn()
    st = FakeStorage({("horse_result", "2024100003"): {"owner": "O3", "breeder": "B3", "birthplace": "安平", "birthday": "2024年4月4日"}})
    prof = f.profile_for(["2020100001", "2024100001", "2024100002", "2024100003"], st)
    assert st.calls == [("horse_result", "2024100003")]   # 手元にある 2 頭は読まない
    assert list(prof.columns) == ["horse_id", "owner", "breeder", "birthplace", "birthday"]   # 余計な列（birthday_x・bp_retry）を混ぜない
    p = prof.set_index("horse_id")
    assert p.loc["2024100002", "owner"] == "O2" and p.loc["2024100003", "owner"] == "O3"
    assert f.hist.attrs.loc["2024100002", "birthplace"] == "新冠"
    assert f.hist.attrs.loc["2024100002", "birthday"] == pd.Timestamp("2024-03-03")
    assert f.profile_stats == {"miss": 1, "local": 0, "absent": 0, "gcs": 1, "notfound": 0, "dyn_rows": 2}   # 手元の 2 頭は warm（_absorb_profile_dyn）の時点で表に入る
    assert not (own / ".profile_dyn.lock").exists() and not (main / ".profile_dyn.lock").exists()   # profile_for は書かない
    f.profile_for(["2024100003"], st)
    assert len(st.calls) == 1                             # 同じプロセスでは読み直さない


def test_rereads_when_profile_dyn_updated(dirs):
    own, main = dirs
    f = _features(own)
    f._absorb_profile_dyn()
    _dyn(main / "profile_dyn.parquet", [{"horse_id": "2024100009", "owner": "O9", "breeder": "B9", "birthplace": "日高", "birthday": "2024年1月1日"}])
    st = FakeStorage()
    f.profile_for(["2024100009"], st)
    assert st.calls == []                                  # 後から書き足された（mtime が変わった）ファイルを読み直す


def test_disabled_falls_back_to_gcs(dirs, monkeypatch):
    own, main = dirs
    _dyn(main / "profile_dyn.parquet", [{"horse_id": "2024100002", "owner": "O2", "breeder": "B2", "birthplace": "新冠", "birthday": None}])
    for how in ("env", "flag"):
        if how == "env":
            monkeypatch.setenv("KEIBA_PROFILE_LOCAL", "0")
        else:
            monkeypatch.delenv("KEIBA_PROFILE_LOCAL")
            (sv.ROOT / "data/local/meta").mkdir(parents=True, exist_ok=True)
            (sv.ROOT / "data/local/meta/profile_local.off").touch()
        assert not sv.profile_local_enabled()
        f = _features(own)
        f._absorb_profile_dyn()
        st = FakeStorage({("horse_result", "2024100002"): {"owner": "O2"}})
        f.profile_for(["2024100002"], st)
        assert st.calls == [("horse_result", "2024100002")]
        assert uss.persist_fetched(f) == (0, 0)            # 止めているときは書き戻さない
        assert not (own / "profile_dyn.parquet").exists()


def test_own_state_dyn_still_read_when_disabled(dirs, monkeypatch):
    """手元を読まない設定でも、<state_dir> の profile_dyn は今までどおり warm で読む。"""
    own, _ = dirs
    monkeypatch.setenv("KEIBA_PROFILE_LOCAL", "0")
    _dyn(own / "profile_dyn.parquet", [{"horse_id": "2024100001", "owner": "O1", "breeder": "B1", "birthplace": "浦河", "birthday": None}])
    f = _features(own)
    assert f._absorb_profile_dyn() == 1
    st = FakeStorage()
    f.profile_for(["2024100001"], st)
    assert st.calls == []


def test_persist_fetched_only_found(dirs):
    own, _ = dirs
    _dyn(own / "profile_dyn.parquet", [{"horse_id": "2024100001", "owner": "OLD", "breeder": "B1", "birthplace": "浦河", "birthday": "2024年2月2日",
                                        "birthday_x": pd.Timestamp("2024-02-02"), "birthday_y": pd.NaT, "color": None, "sex": None, "bp_retry": True}])
    f = _features(own)
    f._absorb_profile_dyn()
    st = FakeStorage({("horse_result", "2024100005"): {"owner": "O5", "breeder": "B5", "birthplace": None, "birthday": "2024年5月5日"},
                      ("horse_profile", "2024100006"): {"owner": "O6", "breeder": None, "birthplace": "日高", "birthday": "2024年6月6日"}})
    f.profile_for(["2024100001", "2024100005", "2024100006", "2024100007"], st)   # 7 は見つからない
    assert f.profile_stats["notfound"] == 1
    assert uss.persist_fetched(f) == (2, 1)              # 7 は『文書なし』として記録
    d = pd.read_parquet(own / "profile_dyn.parquet").set_index("horse_id")
    assert list(d.reset_index().columns) == uss.DYN_COLS   # birthday_x・birthday_y・color・sex を落とす
    assert set(d.index) == {"2024100001", "2024100005", "2024100006"}   # 見つからない馬は書かない
    assert d.loc["2024100001", "owner"] == "OLD" and bool(d.loc["2024100001", "bp_retry"])   # 既にある行は変えない
    assert d.loc["2024100005", "birthday"] == "2024年5月5日" and not bool(d.loc["2024100005", "bp_retry"])
    assert uss.persist_fetched(f)[0] == 0                  # 2 回目は足すものが無い（重複しない）
    assert len(pd.read_parquet(own / "profile_dyn.parquet")) == 3
    f2 = _features(own)                                    # 次のプロセス（weight-runner）は読まない
    f2._absorb_profile_dyn()
    st2 = FakeStorage()
    f2.profile_for(["2024100005", "2024100006"], st2)
    assert st2.calls == []


def test_gcs_outage_not_persisted(dirs):
    own, _ = dirs
    f = _features(own)
    f.profile_for(["2024100008"], FakeStorage(fail=True))
    assert uss.persist_fetched(f) == (0, 0)
    assert not (own / "profile_dyn.parquet").exists() and not (own / "profile_absent.parquet").exists()


def test_save_profile_dyn_prefer_new_keeps_others(dirs):
    own, _ = dirs
    p = own / "profile_dyn.parquet"
    uss.save_profile_dyn(p, pd.DataFrame([{"horse_id": "1", "owner": "a"}, {"horse_id": "2", "owner": "b"}]), prefer_new=False)
    uss.save_profile_dyn(p, pd.DataFrame([{"horse_id": "2", "owner": "B", "birthday": pd.Timestamp("2024-01-02")}]), prefer_new=True)
    d = pd.read_parquet(p).set_index("horse_id")
    assert d.loc["1", "owner"] == "a" and d.loc["2", "owner"] == "B" and d.loc["2", "birthday"] == "2024年1月2日"
    assert len(d) == 2


def test_persist_unwritable_dir_does_not_raise(dirs, monkeypatch):
    own, _ = dirs
    f = _features(own)
    f.fetched = [{"horse_id": "2024100005", "owner": "O5", "breeder": None, "birthplace": None, "birthday": None}]

    def boom(*a, **k):
        raise PermissionError("書けない")

    monkeypatch.setattr(uss, "save_profile_dyn", boom)
    assert uss.persist_fetched(f) == (0, 0)


def test_absent_recorded_and_skipped(dirs, monkeypatch):
    """文書が無い馬（新馬）は precompute が記録し、後のプロセスは期限内は読まない。期限を過ぎたら読む。profile（skip_absent=False）は読む。"""
    own, main = dirs
    f = _features(main)                                    # precompute（本体の state）
    st = FakeStorage()
    f.profile_for(["2024100011", "2024100012"], st)
    assert len(st.calls) == 4                              # horse_result → horse_profile の 2 回ずつ
    assert uss.persist_fetched(f) == (0, 2)
    assert set(pd.read_parquet(main / "profile_absent.parquet")["horse_id"]) == {"2024100011", "2024100012"}

    g = _features(own)                                     # 影のランナー（別の state。本体の記録を読む）
    g._absorb_profile_dyn()
    st2 = FakeStorage()
    prof = g.profile_for(["2024100011", "2024100012"], st2)
    assert st2.calls == []
    assert g.profile_stats["absent"] == 2
    assert set(prof["horse_id"]) >= {"2024100011", "2024100012"}   # 値の無い行として足す（GCS で読んだときと同じ）
    g.profile_for(["2024100011"], st2)
    assert st2.calls == [] and g.profile_stats["miss"] == 2       # 同じプロセスでは数え直さない

    monkeypatch.setenv("KEIBA_PROFILE_ABSENT_TTL_SEC", "0")
    h = _features(own)
    st3 = FakeStorage()
    h.profile_for(["2024100011"], st3)
    assert st3.calls == [("horse_result", "2024100011"), ("horse_profile", "2024100011")]   # 期限切れは読む

    monkeypatch.delenv("KEIBA_PROFILE_ABSENT_TTL_SEC")
    k = _features(own)
    k.skip_absent = False                                  # update_serving_state profile
    st4 = FakeStorage()
    k.profile_for(["2024100012"], st4)
    assert len(st4.calls) == 2


def test_absent_not_recorded_when_degraded(dirs, monkeypatch):
    """読みの間に GCS の不調（note_degraded）があった馬は『文書なし』にしない。"""
    own, _ = dirs
    from src.scraper import storage as sto

    class Degrading(FakeStorage):
        def load(self, category, key):
            sto.note_degraded()
            return super().load(category, key)

    f = _features(own)
    f.profile_for(["2024100013"], Degrading())
    assert f.absent == {}
    f2 = _features(own)
    f2.profile_for(["2024100014"], FakeStorage())
    assert set(f2.absent) == {"2024100014"}


def test_save_profile_absent_merges_and_expires(dirs):
    own, _ = dirs
    p = own / "profile_absent.parquet"
    now = time.time()
    uss.save_profile_absent(p, {"a": now - 8 * 86400, "b": now - 100})
    uss.save_profile_absent(p, {"b": now, "c": now})
    d = pd.read_parquet(p).set_index("horse_id")
    assert set(d.index) == {"b", "c"}                      # 7 日より古い a は落とす
    assert d.loc["b", "checked_at"] == pytest.approx(now)


def test_empty_dyn_rows_are_refetched_and_replaced(dirs):
    """profile_dyn の値がすべて空の行は手元にあるとみなさず GCS で読み直し、precompute が取れた値で置き換える。"""
    own, _ = dirs
    _dyn(own / "profile_dyn.parquet", [{"horse_id": "2024100021", "owner": None, "breeder": None, "birthplace": None, "birthday": None, "bp_retry": True},
                                        {"horse_id": "2024100022", "owner": "O22", "breeder": None, "birthplace": None, "birthday": None, "bp_retry": False}])
    f = _features(own)
    f._absorb_profile_dyn()
    st = FakeStorage({("horse_result", "2024100021"): {"owner": "O21", "breeder": "B21", "birthplace": "浦河", "birthday": "2024年1月2日"}})
    f.profile_for(["2024100021", "2024100022"], st)
    assert st.calls == [("horse_result", "2024100021")]
    assert uss.persist_fetched(f) == (1, 0)
    d = pd.read_parquet(own / "profile_dyn.parquet").set_index("horse_id")
    assert len(d) == 2 and d.loc["2024100021", "owner"] == "O21" and d.loc["2024100022", "owner"] == "O22"


def test_reread_when_existing_file_replaced_and_state_first(dirs):
    """既にある profile_dyn が置き換わったら読み直す。<state_dir> と本体の両方にある馬は <state_dir> の値。"""
    own, main = dirs
    _dyn(own / "profile_dyn.parquet", [{"horse_id": "2024100031", "owner": "OWN", "breeder": None, "birthplace": "浦河", "birthday": None}])
    _dyn(main / "profile_dyn.parquet", [{"horse_id": "2024100031", "owner": "MAIN", "breeder": None, "birthplace": "浦河", "birthday": None}])
    f = _features(own)
    f._absorb_profile_dyn()
    assert f.hist.profile.set_index("horse_id").loc["2024100031", "owner"] == "OWN"
    uss.save_profile_dyn(main / "profile_dyn.parquet", pd.DataFrame([{"horse_id": "2024100032", "owner": "O32"}]), prefer_new=False)
    st = FakeStorage()
    f.profile_for(["2024100032"], st)
    assert st.calls == [] and f.profile_stats["local"] == 1


def test_absent_read_from_state_dir_when_local_dir_differs(dirs):
    """precompute が KEIBA_PROFILE_LOCAL_DIR と別の state に書いても、同じ state で動くプロセスは記録を読む。"""
    own, main = dirs
    f = _features(own)                                     # state=own、本体の置き場=main（ずれている）
    f.profile_for(["2024100041"], FakeStorage())
    assert uss.persist_fetched(f) == (0, 1)
    assert (own / "profile_absent.parquet").exists() and not (main / "profile_absent.parquet").exists()
    g = _features(own)
    st = FakeStorage()
    g.profile_for(["2024100041"], st)
    assert st.calls == []


def test_broken_profile_dyn_does_not_stop_precompute(dirs):
    own, _ = dirs
    (own / "profile_dyn.parquet").write_bytes(b"not parquet")
    f = _features(own)
    f._absorb_profile_dyn()                                # 読めなければ警告だけ
    f.profile_for(["2024100051"], FakeStorage({("horse_result", "2024100051"): {"owner": "O51"}}))
    assert uss.persist_fetched(f) == (0, 0)                # 壊れたファイルに書き戻せなくても落ちない
    assert not list(own.glob(".profile_dyn.parquet.*.tmp"))
