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
        et.export_year_local("2026", ["horse_result"], storage=_NoListStorage({}), today=date(2026, 11, 9))


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


# ── 本体の 6 カテゴリ（race_result・出馬表・ラップ・速報・オッズ 2 種）──────────────────────────────

def _race_lists(tmp_path, ymd, ids):
    import json
    d = tmp_path / "race_lists"
    d.mkdir(exist_ok=True)
    (d / f"{ymd}.json").write_text(json.dumps({"date": ymd, "races": [{"race_id": r} for r in ids]}))


def test_result_candidates_come_from_race_lists_and_shutuba_and_fill_missing_rounds(tmp_path, monkeypatch):
    d = _setup(tmp_path, monkeypatch, [("202605010101", "2026-10-31")])
    # race_lists は 11-07 の 1R・2R だけ（途中で切れた）。11-08 は race_lists に無く、出馬表の flat にだけある
    _race_lists(tmp_path, "20261107", ["202605020101", "202605020102"])
    pd.DataFrame([{"race_id": "202605020201", "date": "2026-11-08", "horse_id": "h1", "horse_number": 1}]).to_parquet(d / "race_shutuba_flat.parquet")
    got = set(et.local_candidates("2026", "race_result", date(2026, 11, 9), attempts={}))
    assert {f"2026050201{n:02d}" for n in range(1, 13)} <= got          # 欠けた回（3〜12R）も補う
    assert {f"2026050202{n:02d}" for n in range(1, 13)} <= got          # race_lists に無い日も出馬表から
    assert "202605010101" not in got                                    # 確定結果の flat にあるもの


def test_todays_odds_are_not_candidates_but_shutuba_ahead_are(tmp_path, monkeypatch):
    d = _setup(tmp_path, monkeypatch, [("202605010101", "2026-10-31")])
    _race_lists(tmp_path, "20261114", ["202605030101"])
    _race_lists(tmp_path, "20261120", ["202605040101"])
    _seed_flat(d, "race_odds", "202605010101")
    assert "202605030101" not in et.local_candidates("2026", "race_odds", date(2026, 11, 14), attempts={})   # 当日は途中の版
    assert "202605030101" in et.local_candidates("2026", "race_odds", date(2026, 11, 15), attempts={})
    sh = set(et.local_candidates("2026", "race_shutuba", date(2026, 11, 14), attempts={}))
    assert {"202605030101", "202605040101"} <= sh                         # 出馬表は先の日も


def test_pre_draw_shutuba_is_read_again(tmp_path, monkeypatch):
    d = _setup(tmp_path, monkeypatch, [("202605010101", "2026-10-31")])
    pd.DataFrame([{"race_id": "202605030101", "date": "2026-11-15", "horse_id": "h1", "horse_number": 0}]).to_parquet(d / "race_shutuba_flat.parquet")
    pd.DataFrame([{"race_id": "202605030101"}]).to_parquet(d / "race_shutuba_race.parquet")
    assert "202605030101" in et.local_candidates("2026", "race_shutuba", date(2026, 11, 13), attempts={})


def test_result_is_retried_daily_but_side_categories_back_off(tmp_path):
    t = date(2026, 11, 13)
    assert et._attempt_due([9, "2026-11-12"], t, "daily") and not et._attempt_due([9, "2026-11-13"], t, "daily")
    assert not et._attempt_due([2, "2026-11-12"], t, "backoff")


def test_export_local_main_categories_without_listing_and_bridges(tmp_path, monkeypatch):
    d = _setup(tmp_path, monkeypatch, [("202605010101", "2026-10-31")])
    pd.read_parquet(d / "race_result_flat.parquet").to_parquet(d / "race_result_flat.parquet")
    pd.DataFrame([{"race_id": "202605010101"}]).to_parquet(d / "race_result_race.parquet")
    _race_lists(tmp_path, "20261107", ["202605020101"])
    new = {"race_id": "202605020101", "date": "2026-11-07", "race_name": "x", "venue": "東京", "distance": 1600, "surface": "芝",
           "field_size": 1, "entries": [{"horse_number": 1, "horse_id": "h9", "finish_position": 1}]}
    st = _NoListStorage({("race_result", "202605020101"): new, ("race_shutuba", "202605020101"): new})
    called = []
    monkeypatch.setattr(et, "bridge_on_time_results", lambda y: called.append(y) or {"bridged_races": 0})
    rep = et.export_year_local("2026", ["race_odds", "race_result"], storage=st, today=date(2026, 11, 9))
    assert [s["category"] for s in rep["categories"]][:2] == ["race_result", "race_odds"]   # race_result が先
    assert "202605020101" in set(pd.read_parquet(d / "race_result_flat.parquet")["race_id"].astype(str))
    assert called == ["2026"] and rep["errors"] == 0


def test_new_year_builds_from_race_lists_without_listing(tmp_path, monkeypatch):
    monkeypatch.setattr(et, "OUTPUT_DIR", tmp_path / "tables")
    monkeypatch.setattr(dc, "NOT_AVAILABLE_DIR", tmp_path / "meta" / "not_available")
    (tmp_path / "tables" / "2027").mkdir(parents=True)
    _race_lists(tmp_path, "20270105", ["202706010101"])
    doc = {"race_id": "202706010101", "date": "2027-01-05", "race_name": "x", "venue": "中山", "distance": 2000, "surface": "芝",
           "field_size": 1, "entries": [{"horse_number": 1, "horse_id": "h1", "finish_position": 1}]}
    st = _NoListStorage({("race_result", "202706010101"): doc, ("race_shutuba", "202706010101"): doc})
    rep = et.export_year_local("2027", ["race_result"], storage=st, today=date(2027, 1, 6))
    assert rep["categories"][0].get("initial")
    assert "202706010101" in set(pd.read_parquet(tmp_path / "tables" / "2027" / "race_result_flat.parquet")["race_id"].astype(str))


def test_whole_missing_day_is_filled_from_the_same_meeting(tmp_path, monkeypatch):
    """race_lists が丸ごと無い日（2026-09-22 型）も、同じ開催（場・回）の前の日から次の日を補う。"""
    _setup(tmp_path, monkeypatch, [("202606040401", "2026-11-07")])
    got = set(et.local_candidates("2026", "race_result", date(2026, 11, 10), attempts={}))
    assert {f"2026060405{n:02d}" for n in range(1, 13)} <= got
    assert {f"2026060406{n:02d}" for n in range(1, 13)} <= got
    # 推定の日付は確定結果だけに使う（オッズなどはレース前の版を読まないよう、本当の日付が分かってから）
    assert not {f"2026060405{n:02d}" for n in range(1, 13)} & set(et.local_candidates("2026", "race_odds", date(2026, 11, 10), attempts={}))


def test_empty_shutuba_doc_in_race_table_is_read_again(tmp_path, monkeypatch):
    """本体のカテゴリは flat だけで判定する（空の出馬表は race 表にだけ入るが、翌日以降に読み直す）。"""
    d = _setup(tmp_path, monkeypatch, [("202605010101", "2026-10-31")])
    _race_lists(tmp_path, "20261115", ["202605030101"])
    _seed_flat(d, "race_shutuba", "202605010101")
    pd.DataFrame([{"race_id": "202605030101"}]).to_parquet(d / "race_shutuba_race.parquet")
    assert "202605030101" in et.local_candidates("2026", "race_shutuba", date(2026, 11, 13), attempts={})
