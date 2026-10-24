"""出馬表の馬名・騎手・調教師をタップしたときの情報（src/api/entry_card.py）。合成データで、先読みの境界・少数サンプル・データなし・乗り替わりを確かめる。"""

from __future__ import annotations

import pytest

from src.api import entry_card as E



@pytest.fixture(autouse=True)
def _legacy_growth_axis(monkeypatch):
    # 既定の軸は megu_robust。旧列(megu_index)だけの検証データを使うテストは従来の軸を指定する
    monkeypatch.setenv("KEIBA_GROWTH_AXIS", "megu")

@pytest.fixture(autouse=True)
def _fresh_cache():
    E.clear_cache()
    yield
    E.clear_cache()


class FakeStorage:
    def __init__(self, data):
        self.data = data

    def load(self, cat, key):
        return self.data.get((cat, key))


RID = "202605040611"            # 東京(05) 2026-10-17
PAST_RAW = "2026.10.03 京都 5 2歳未勝利 ダ1400 1:26.4 良 15頭 8番 12人 森田誠也 53.0 14-13 (37.3) 486(-4) ペアスターロード (1.4)"


def make_storage(over=None):
    data = {
        ("race_shutuba", RID): {
            "date": "2026-10-17", "venue": "東京", "surface": "ダート", "distance": 1600, "grade": "1勝", "race_class": "3歳以上1勝クラス", "race_name": "テスト特別",
            "entries": [{"horse_number": 3, "horse_name": "テストホース", "horse_id": "H1", "jockey_id": "J1", "jockey_name": "騎手A", "trainer_id": "T1", "trainer_name": "調教師A",
                         "sex_age": "牡3", "jockey_weight": 56.0, "weight": 0, "weight_change": 0, "sire": "", "dam_sire": ""}]},
        ("race_shutuba_past", RID): {"entries": [{"horse_id": "H1", "past_races": [
            {"date": "2026.10.03", "surface": "ダ", "distance": 1400, "finish_time": "1:26.4", "raw": PAST_RAW, "finish_position": 5, "race_id": "202608040101"},
            {"date": "2026.09.20", "surface": "ダ", "distance": 1200, "finish_time": "1:15.7", "raw": "2026.09.20 阪神 7 2歳新馬 ダ1200 1:15.7 良 15頭 4番 15人 森田誠也 52.0 12-12 (37.2) 490(0) フィリップヴィル (1.9)", "finish_position": 7, "race_id": "202609040604"},
            {"date": "2026.10.17", "surface": "ダ", "distance": 1600, "finish_time": "1:40.0", "raw": "2026.10.17 東京 1 テスト特別 ダ1600 1:40.0 良 10頭 3番 1人 騎手A 56.0 1-1 (35.0) 500(+2) X (0.0)", "finish_position": 1, "race_id": RID},   # 当日の行は除く
        ]}]},
        ("race_oikiri", RID): {"entries": [{"horse_id": "H1", "training_date": "2026/10/14(水)", "course": "", "condition": "良", "rider": "助手", "lap_times": "-69.1(16.9)52.2(13.1)", "impression": "仕掛", "evaluation": "C", "comment": "やや良化"}]},
        ("horse_result", "H1"): {"horse_name": "テストホース", "sire": "ジャスタウェイ", "dam": "母", "dam_sire": "スピルバーグ", "career": "3戦1勝 [1-0-1-1]", "race_history": [
            {"date": "2026/10/03", "venue": "4京都1", "race_name": "2歳未勝利", "surface": "ダ", "distance": 1400, "finish_position": 5, "race_id": "202608040101", "jockey_name": "森田誠也"},
            {"date": "2026/09/20", "venue": "5阪神1", "race_name": "2歳新馬", "surface": "ダ", "distance": 1200, "finish_position": 7, "race_id": "202609040604"},
            {"date": "2026/08/01", "venue": "2東京3", "race_name": "新馬", "surface": "芝", "distance": 1600, "finish_position": 1, "race_id": "202605020103"},
            {"date": "2026/10/17", "venue": "2東京8", "race_name": "当日の結果", "surface": "ダ", "distance": 1600, "finish_position": 1, "race_id": RID},                # 当日は除く
        ]},
        ("race_shutuba", "202608040101"): {"entries": [{"horse_id": "H1", "jockey_id": "J9", "jockey_name": "森田誠也"}]},
        ("race_shutuba", "202609040604"): {"entries": [{"horse_id": "H1", "jockey_id": "J9", "jockey_name": "森田誠也"}]},
    }
    data.update(over or {})
    return FakeStorage(data)


@pytest.fixture
def features(tmp_path, monkeypatch):
    """騎手・調教師の lookup を合成して差し替える。"""
    pl = pytest.importorskip("polars")
    for kind, ident in (("jockey", "J1"), ("trainer", "T1")):
        pre = "jk" if kind == "jockey" else "tr"
        d = tmp_path / f"{kind}_tbl"
        d.mkdir()
        pl.DataFrame({f"{kind}_id": [ident], f"{pre}_cal90_starts": [30.0], f"{pre}_cal90_wins": [3.0], f"{pre}_cal90_top3": [9.0],
                      f"{pre}_prior_all_starts": [1000.0], f"{pre}_prior_all_wins": [100.0], f"{pre}_prior_all_top3": [300.0], f"{pre}_prior_all_avg_finish": [6.5]}).write_parquet(d / f"lookup_{kind}.parquet")
        pl.DataFrame({f"{kind}_id": [ident] * 3, "slice_kind": ["venue_code", "surface_coarse", "distance_band"], "slice_key": ["05", "dirt_ob", "dirt_ob_M_1600_1999"],
                      "starts": [200, 5, 400], "wins": [20.0, 1.0, 40.0], "top3": [60.0, 2.0, 120.0], "avg_finish": [6.0, 5.0, 6.2]}).write_parquet(d / f"lookup_{kind}_slices.parquet")
    (tmp_path / "jockey_trainer_stats").mkdir()
    (tmp_path / "jockey_trainer_stats" / "_manifest.json").write_text('{"generated_at": "2026-10-15T20:30:41+00:00"}')
    monkeypatch.setenv("KEIBA_FEATURES_DIR", str(tmp_path))
    E._lookup_cache.clear()
    return tmp_path


def test_horse_recent_runs_exclude_the_race_day_and_parse_the_horse_line():
    c = E.build_entry_card(RID, "H1", make_storage())
    runs = c["horse"]["recent"]
    assert [r["date"] for r in runs] == ["20261003", "20260920"], "レース当日（10/17）の行は先読みなので含めない"
    r = runs[0]
    assert (r["venue"], r["race_name"], r["going"], r["field"], r["post"], r["popularity"]) == ("京都", "2歳未勝利", "良", 15, 8, 12)
    assert (r["jockey"], r["jockey_weight"], r["passing"], r["last3f"], r["weight"], r["weight_change"], r["margin"]) == ("森田誠也", 53.0, "14-13", 37.3, 486, -4, 1.4)


def test_same_conditions_use_only_runs_before_the_race_day():
    c = E.build_entry_card(RID, "H1", make_storage())
    by = {x["label"].split("（")[0]: x for x in c["horse"]["same_conditions"]}
    assert by["この馬場"]["starts"] == 2 and by["この馬場"]["wins"] == 0          # ダ 2 走（芝の1勝・当日のダ1勝は含まない）
    assert by["この馬場"]["win_rate"] == 0.0 and by["この競馬場"]["win_rate"] == 100.0
    assert by["この競馬場"]["starts"] == 1 and by["この競馬場"]["wins"] == 1        # 東京は 8/1 の芝（1着）のみ。当日の東京は除く
    assert by["この距離"]["starts"] == 0                                        # ダ マイル(1600-1999) の過去走は無い（1400/1200 は短距離）


def test_changes_layoff_distance_and_jockey_change_by_ids():
    c = E.build_entry_card(RID, "H1", make_storage())
    ch = c["horse"]["changes"]
    assert ch["layoff_days"] == 14 and ch["distance_change"] == 200
    assert ch["jockey_changed"] is True and ch["previous_jockey"] == "森田誠也"        # 前走は J9、今回は J1 → 乗り替わり
    s2 = make_storage({("race_shutuba", "202608040101"): {"entries": [{"horse_id": "H1", "jockey_id": "J1", "jockey_name": "騎手A"}]}})
    assert E.build_entry_card(RID, "H1", s2)["horse"]["changes"]["jockey_changed"] is False
    s3 = make_storage({("race_shutuba", "202608040101"): None, ("race_result", "202608040101"): None})
    assert E.build_entry_card(RID, "H1", s3)["horse"]["changes"]["jockey_changed"] is None        # 前走の騎手 id が分からないときは断定しない


def test_training_profile_and_jockey_history_with_this_horse():
    c = E.build_entry_card(RID, "H1", make_storage({("race_shutuba", "202608040101"): {"entries": [{"horse_id": "H1", "jockey_id": "J1"}]}}))
    assert c["horse"]["training"]["evaluation"] == "C" and c["horse"]["training"]["training_date"] == "2026/10/14(水)"
    assert c["horse"]["profile"]["sire"] == "ジャスタウェイ" and c["horse"]["profile"]["dam_sire"] == "スピルバーグ"
    assert c["jockey"]["with_horse"]["rides_in_last5"] == 1 and c["jockey"]["with_horse"]["checked"] == 2
    assert c["horse"]["trainer_comment"] is None                                  # 無いものは None（作らない）


def test_jockey_and_trainer_cards_use_lookup_with_counts_and_snapshot_date(features):
    c = E.build_entry_card(RID, "H1", make_storage())
    j = c["jockey"]
    assert j["status"] == "ok" and j["as_of"] == "2026-10-15" and j["includes_after_race"] is False        # 10/15 の集計は 10/17 のレースより前
    by = {x["label"].split("（")[0]: x for x in j["conditions"]}
    assert (by["この競馬場"]["starts"], by["この競馬場"]["wins"], by["この競馬場"]["win_rate"], by["この競馬場"]["top3_rate"]) == (200, 20, 10.0, 30.0)
    assert by["この馬場"]["starts"] == 5 and by["この馬場"]["win_rate"] == 20.0                                 # 母数が小さい値もそのまま母数つきで返す
    assert by["このクラス"]["has_data"] is False and by["このクラス"]["starts"] is None                     # 区分に無いものは None（0 に見せない）
    assert j["recent"]["cal90"]["starts"] == 30 and j["career"]["win_rate"] == 10.0
    assert c["trainer"]["status"] == "ok" and c["trainer"]["conditions"][0]["starts"] == 200


def test_past_race_marks_that_the_snapshot_includes_later_results(features):
    s = make_storage({("race_shutuba", RID): {**make_storage().data[("race_shutuba", RID)], "date": "2026-10-10"}})
    assert E.build_entry_card(RID, "H1", s)["jockey"]["includes_after_race"] is True


def test_no_data_cases_do_not_break_and_say_no_data(tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_FEATURES_DIR", str(tmp_path / "none"))
    E._lookup_cache.clear()
    st = FakeStorage({("race_shutuba", RID): make_storage().data[("race_shutuba", RID)]})
    c = E.build_entry_card(RID, "H1", st)
    assert c["status"] == "ok"
    assert c["horse"]["status"] == "no_data" and c["horse"]["recent"] == [] and c["horse"]["training"] is None
    assert c["jockey"]["status"] == "no_data" and c["trainer"]["status"] == "no_data"
    assert E.build_entry_card(RID, "NOPE", st)["status"] == "not_found"
    assert E.build_entry_card("../x", "H1", st)["status"] == "bad_request"


def test_falls_back_to_horse_history_when_the_past_card_is_missing():
    s = make_storage({("race_shutuba_past", RID): None})
    runs = E.build_entry_card(RID, "H1", s)["horse"]["recent"]
    assert [r["date"] for r in runs] == ["20261003", "20260920", "20260801"] and runs[0]["venue"] == "京都"


def test_storage_errors_are_not_fatal():
    class Boom(FakeStorage):
        def load(self, cat, key):
            if cat != "race_shutuba":
                raise RuntimeError("gcs down")
            return super().load(cat, key)

    c = E.build_entry_card(RID, "H1", Boom(make_storage().data))
    assert c["status"] == "ok" and c["horse"]["status"] == "no_data"


def test_slice_keys_match_the_pipeline_definitions():
    """集計の区分キーは jockey_trainer_stats.py と一致していること（lookup を正しく引くため）。"""
    pd = pytest.importorskip("pandas")
    from src.pipeline.features import jockey_trainer_stats as J

    for dist, surf in ((1200, "芝"), (1600, "ダート"), (2000, "芝"), (2400, "ダ"), (3000, "障")):
        assert E.distance_band_key(dist, surf) == str(J._distance_band(pd.Series([dist]), pd.Series([surf])).iloc[0])
    for g, rc in (("G1", ""), ("G3", ""), ("L", ""), ("OP", ""), ("1勝", "3歳以上1勝クラス"), ("", "2歳未勝利"), ("", "")):
        assert E.grade_bucket_key(g, rc) == str(J._grade_bucket(pd.Series([g]), pd.Series([rc])).iloc[0]), (g, rc)


def test_endpoint_returns_card_and_error_codes(monkeypatch):
    from fastapi.testclient import TestClient

    from src.api import app as A

    monkeypatch.setattr(A, "_get_storage", lambda: make_storage())
    c = TestClient(A.app)
    r = c.get(f"/api/race/{RID}/entry/H1/card")
    assert r.status_code == 200 and r.json()["horse"]["name"] == "テストホース" and len(r.content) < 20000
    assert c.get(f"/api/race/{RID}/entry/NOPE/card").status_code == 404
    assert c.get(f"/api/race/{RID}/entry/H1;rm/card").status_code in (400, 404)


def test_card_has_growth_curve_with_only_runs_before_the_race_day(tmp_path, monkeypatch):
    import pandas as pd

    from src.pipeline.inference import horse_growth as HG

    for y, ids in ((2026, ["202608040101", "202609040604", "202605020103", RID]),):
        d = tmp_path / str(y)
        d.mkdir()
        pd.DataFrame([{"race_id": r, "horse_id": "H1", "megu_index": 40.0 + i * 5, "computation_status": "valid"} for i, r in enumerate(ids)]).to_parquet(d / "megu_index_flat.parquet")
    monkeypatch.setattr(HG, "tables_dir", lambda: tmp_path)
    HG._baseline_cache.clear()
    g = E.build_entry_card(RID, "H1", make_storage())["horse"]["growth"]
    assert g["status"] == "ok" and g["n"] == 3
    assert RID not in [p["race_id"] for p in g["points"]], "レース当日の走は含めない（先読みなし）"
    assert [p["date"] for p in g["points"]] == ["20260801", "20260920", "20261003"]


def test_card_growth_is_no_data_without_megu_files(tmp_path, monkeypatch):
    from src.pipeline.inference import horse_growth as HG

    monkeypatch.setattr(HG, "tables_dir", lambda: tmp_path)
    assert E.build_entry_card(RID, "H1", make_storage())["horse"]["growth"]["status"] == "no_data"
