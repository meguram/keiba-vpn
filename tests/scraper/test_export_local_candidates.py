"""一覧なしの日次の追記（export_tables.export_year_local / local_candidates）。GCS の一覧を呼ばないこと・窓・試行の間隔を確かめる。"""
from datetime import date

import pandas as pd
import pytest

from src.scraper import date_coverage as dc
from src.scraper import export_tables as et


class _NoListStorage:
    """batch_list_blobs が呼ばれたら失敗する偽の storage。load は docs にあるものだけ返す。"""

    def __init__(self, docs, healthy=True):
        self.docs = docs
        self.loaded = []
        self._gcs_healthy = healthy

    def batch_list_blobs(self, category, year):
        raise AssertionError(f"一覧を取ってはいけない: {category}/{year}")

    def load(self, category, key):
        self.loaded.append((category, key))
        return self.docs.get((category, key))


def _setup(tmp_path, monkeypatch, races):
    monkeypatch.setattr(et, "OUTPUT_DIR", tmp_path / "tables")
    monkeypatch.setattr(dc, "NOT_AVAILABLE_DIR", tmp_path / "meta" / "not_available")
    d = tmp_path / "tables" / "2026"
    d.mkdir(parents=True)
    pd.DataFrame([{"race_id": r, "date": dt, "venue": "東京", "distance": 1600, "surface": "芝", "horse_id": "h1",
                   "horse_number": 1, "finish_position": 1} for r, dt in races]).to_parquet(d / "race_result_flat.parquet")
    return d


def _oikiri(rid):
    return {"race_id": rid, "entries": [{"horse_number": 1, "horse_id": "h1", "evaluation": "A"}]}


def _seed_flat(d, cat, rid):
    pd.DataFrame([{"race_id": rid, "horse_id": "h0", "horse_number": 1}]).to_parquet(d / f"{cat}_flat.parquet")


def test_candidates_window_and_existing(tmp_path, monkeypatch):
    d = _setup(tmp_path, monkeypatch, [("202605010101", "2026-10-01"), ("202605010201", "2026-11-07"),
                                       ("202605010301", "2026-11-12"), ("202605010401", "2026-11-13")])
    _seed_flat(d, "race_oikiri", "202605010201")
    got = et.local_candidates("2026", "race_oikiri", date(2026, 11, 13), attempts={})
    # 10-01 は窓（21 日）の外、11-07 は flat にある、11-13 は当日なので翌日から
    assert got == ["202605010301"]


def test_barometer_waits_for_publish_and_skips_not_available(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, [("202605010101", "2026-11-07"), ("202605010102", "2026-11-01"), ("202605010103", "2026-11-01")])
    dc.record_not_available("race_barometer", "202605010103", reason="steeplechase")
    # 11-07（土）の公開は 11-13（金）18 時 → 11-14 から。11-01（日）は 11-06 公開 → 11-07 から
    assert et.local_candidates("2026", "race_barometer", date(2026, 11, 13), attempts={}) == ["202605010102"]
    assert et.local_candidates("2026", "race_barometer", date(2026, 11, 14), attempts={}) == ["202605010101", "202605010102"]


def test_attempt_backoff_and_cap():
    t = date(2026, 11, 13)
    assert et._attempt_due(None, t)
    assert et._attempt_due([1, "2026-11-12"], t)            # 1 回目の後は 1 日
    assert not et._attempt_due([2, "2026-11-12"], t)        # 2 回目の後は 2 日
    assert et._attempt_due([2, "2026-11-11"], t)
    assert not et._attempt_due([et.LOCAL_MAX_ATTEMPTS, "2026-01-01"], t)


def test_export_local_appends_without_listing_and_records_attempts(tmp_path, monkeypatch):
    d = _setup(tmp_path, monkeypatch, [("202605010101", "2026-11-07"), ("202605010102", "2026-11-07"), ("202605010103", "2026-10-31")])
    _seed_flat(d, "race_oikiri", "202605010103")
    st = _NoListStorage({("race_oikiri", "202605010101"): _oikiri("202605010101")})
    rep = et.export_year_local("2026", ["race_oikiri"], storage=st, today=date(2026, 11, 9))
    s = rep["categories"][0]
    assert (s["candidates"], s["races"], s["not_found"]) == (2, 1, 1)
    flat = pd.read_parquet(d / "race_oikiri_flat.parquet")
    assert set(flat["race_id"].astype(str)) == {"202605010101", "202605010103"}     # 既存の行は残り、読めたものが足される
    att = et._load_attempts("race_oikiri", "2026")
    assert att == {"202605010102": [1, "2026-11-09"]}
    # 翌日: 読めたものは選ばれず、無かったものは間隔 1 日で再び選ばれる
    st2 = _NoListStorage({})
    et.export_year_local("2026", ["race_oikiri"], storage=st2, today=date(2026, 11, 10))
    assert st2.loaded == [("race_oikiri", "202605010102")]
    assert et._load_attempts("race_oikiri", "2026") == {"202605010102": [2, "2026-11-10"]}


def test_unhealthy_gcs_does_not_count_attempts(tmp_path, monkeypatch):
    d = _setup(tmp_path, monkeypatch, [("202605010101", "2026-11-07")])
    _seed_flat(d, "race_oikiri", "202605010199")
    et.export_year_local("2026", ["race_oikiri"], storage=_NoListStorage({}, healthy=False), today=date(2026, 11, 9))
    assert et._load_attempts("race_oikiri", "2026") == {}


def test_empty_doc_is_not_read_again(tmp_path, monkeypatch):
    d = _setup(tmp_path, monkeypatch, [("202605010101", "2026-11-07")])
    _seed_flat(d, "race_paddock", "202605010199")
    st = _NoListStorage({("race_paddock", "202605010101"): {"race_id": "202605010101", "entries": []}})
    et.export_year_local("2026", ["race_paddock"], storage=st, today=date(2026, 11, 9))
    # 中身の空の文書も race 表には入るので、次の日は選ばない
    assert et.local_candidates("2026", "race_paddock", date(2026, 11, 10), attempts={}) == []


def test_missing_flat_is_built_from_results_without_listing_and_other_categories_rejected(tmp_path, monkeypatch):
    """年が替わった最初の日など、flat が無いカテゴリは、結果のあるレース（今日より前）を名前で読んで作る（一覧は取らない）。"""
    d = _setup(tmp_path, monkeypatch, [("202605010101", "2026-11-07"), ("202605010102", "2026-11-09")])
    st = _NoListStorage({("race_oikiri", "202605010101"): _oikiri("202605010101"),
                         ("race_oikiri", "202605010102"): _oikiri("202605010102")})
    rep = et.export_year_local("2026", ["race_oikiri"], storage=st, today=date(2026, 11, 9))
    assert rep["categories"][0].get("initial")
    assert set(pd.read_parquet(d / "race_oikiri_flat.parquet")["race_id"].astype(str)) == {"202605010101"}
    with pytest.raises(ValueError):
        et.export_year_local("2026", ["race_result"], storage=_NoListStorage({}), today=date(2026, 11, 9))


def test_no_results_for_new_year_skips_without_listing(tmp_path, monkeypatch):
    monkeypatch.setattr(et, "OUTPUT_DIR", tmp_path / "tables")
    rep = et.export_year_local("2027", ["race_oikiri"], storage=_NoListStorage({}), today=date(2027, 1, 1))
    assert rep["categories"][0]["skipped"] and rep["errors"] == 0


def test_empty_flat_is_not_overwritten_and_other_categories_continue(tmp_path, monkeypatch):
    """flat が空（読めない）のとき、候補だけで上書きしない。失敗はそのカテゴリだけで、残りは続ける。"""
    d = _setup(tmp_path, monkeypatch, [("202605010101", "2026-11-07")])
    pd.DataFrame({"race_id": pd.Series([], dtype=str)}).to_parquet(d / "race_oikiri_flat.parquet")
    _seed_flat(d, "race_paddock", "202605010199")
    st = _NoListStorage({("race_oikiri", "202605010101"): _oikiri("202605010101"),
                         ("race_paddock", "202605010101"): _oikiri("202605010101")})
    rep = et.export_year_local("2026", ["race_oikiri", "race_paddock"], storage=st, today=date(2026, 11, 9))
    assert rep["errors"] == 1 and "error" in rep["categories"][0]
    assert len(pd.read_parquet(d / "race_oikiri_flat.parquet")) == 0
    assert rep["categories"][1]["races"] == 1
