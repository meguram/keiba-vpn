"""馬柱が無い年（2016〜2018）・一部しか無い年（2019）の位置取り予想を、結果の表の過去走から作る代替（tp53_dataset.past_frame）。"""
import json

import pandas as pd

from src.scripts.data import tp53_dataset as t53


def _write(root, year, rows):
    d = root / str(year)
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(d / "race_result_flat.parquet")


def test_past_runs_are_only_before_the_race_and_shaped_like_the_shutuba_past(tmp_path, monkeypatch):
    monkeypatch.setattr(t53, "TABLES", tmp_path)
    base = {"venue": "東京", "surface": "芝", "distance": 1600, "horse_number": 1}
    _write(tmp_path, 2016, [{"race_id": "201605010101", "date": "2016-05-01", "horse_id": "h1", "field_size": 16, "passing_order": "3-3", **base},
                            {"race_id": "201605020101", "date": "2016-06-01", "horse_id": "h1", "field_size": 12, "passing_order": "1-1", **base}])
    _write(tmp_path, 2017, [{"race_id": "201705010101", "date": "2017-05-01", "horse_id": "h1", "field_size": 10, "passing_order": "5-4", **base}])
    p = t53._past_from_results(2017)
    pr = json.loads(p["past_races"].iloc[0])
    assert [r["date"] for r in pr] == ["20160501", "20160601"]                  # そのレース（2017-05-01）より前だけ
    assert pr[-1]["raw"] == "2016.06.01 12頭 1-1"                               # position_forecast.parse_run が読める形
    from src.pipeline.inference.position_forecast import parse_run

    assert parse_run(pr[-1]["raw"])[1:3] == (1, 12)


# ── 監査 2026-11-16: 馬柱の表が一部しか無い年（2019 年）でも、載っていないレースは結果の表の過去走で補う ──

def _setup_partial(tmp_path, monkeypatch, past_rows):
    monkeypatch.setattr(t53, "TABLES", tmp_path)
    base = {"venue": "東京", "surface": "芝", "distance": 1600}
    _write(tmp_path, 2018, [{"race_id": "201805010101", "date": "2018-05-01", "horse_id": "h1", "horse_number": 1, "field_size": 16,
                             "passing_order": "3-3", **base}])
    _write(tmp_path, 2019, [{"race_id": "201905010101", "date": "2019-05-01", "horse_id": "h1", "horse_number": 1, "field_size": 10,
                             "passing_order": "5-4", **base},
                            {"race_id": "201905010102", "date": "2019-05-01", "horse_id": "h2", "horse_number": 2, "field_size": 10,
                             "passing_order": "2-2", **base}])
    if past_rows is not None:
        pd.DataFrame(past_rows).to_parquet(tmp_path / "2019" / "race_shutuba_past_flat.parquet")


def test_mostly_missing_shutuba_past_year_uses_results_for_the_whole_year(tmp_path, monkeypatch):
    shutuba_run = json.dumps([{"date": "20180401", "raw": "2018.04.01 14頭 7-7"}])
    _setup_partial(tmp_path, monkeypatch, [{"race_id": "201905010101", "date": "2019-05-01", "venue": "東京", "surface": "芝",
                                            "distance": 1600, "horse_number": 1, "horse_id": "h1", "past_races": shutuba_run}])
    monkeypatch.setattr(t53, "PAST_MIN_SHARE", 0.9)                            # 2 レース中 1 レースしか載っていない → 年全体を結果の表から
    p = t53.past_frame(2019).set_index("race_id")
    assert set(p.index) == {"201905010101", "201905010102"}
    assert json.loads(p.loc["201905010101", "past_races"])[0]["raw"] == "2018.05.01 16頭 3-3"


def test_mostly_present_year_fills_only_races_missing_from_the_shutuba_past(tmp_path, monkeypatch):
    shutuba_run = json.dumps([{"date": "20180401", "raw": "2018.04.01 14頭 7-7"}])
    _setup_partial(tmp_path, monkeypatch, [{"race_id": "201905010101", "date": "2019-05-01", "venue": "東京", "surface": "芝",
                                            "distance": 1600, "horse_number": 1, "horse_id": "h1", "past_races": shutuba_run}])
    monkeypatch.setattr(t53, "PAST_MIN_SHARE", 0.5)
    p = t53.past_frame(2019).set_index("race_id")
    assert p.loc["201905010101", "past_races"] == shutuba_run                  # 馬柱のあるレースは馬柱をそのまま
    assert json.loads(p.loc["201905010102", "past_races"]) == []               # 馬柱に無いレースは結果の表から（h2 は過去走なし）


def test_debut_race_with_empty_shutuba_past_is_not_refilled_and_full_years_keep_the_shutuba_past(tmp_path, monkeypatch):
    run = json.dumps([{"date": "20180401", "raw": "2018.04.01 14頭 7-7"}])
    _setup_partial(tmp_path, monkeypatch, [
        {"race_id": "201905010101", "date": "2019-05-01", "venue": "東京", "surface": "芝", "distance": 1600, "horse_number": 1, "horse_id": "h1", "past_races": run},
        {"race_id": "201905010102", "date": "2019-05-01", "venue": "東京", "surface": "芝", "distance": 1600, "horse_number": 2, "horse_id": "h2", "past_races": "[]"}])
    calls = []
    orig = t53._past_from_results
    monkeypatch.setattr(t53, "_past_from_results", lambda y: calls.append(y) or orig(y))
    p = t53.past_frame(2019)
    assert calls == []                                                         # 全レースが馬柱に載っている年は、結果の表を読まない
    assert sorted(p["past_races"].tolist()) == sorted([run, "[]"])            # 新馬（空の馬柱）もそのまま


def test_no_shutuba_past_file_uses_results(tmp_path, monkeypatch):
    _setup_partial(tmp_path, monkeypatch, None)
    assert len(t53.past_frame(2019)) == 2
