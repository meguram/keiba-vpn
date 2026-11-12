"""馬柱が無い年（2016〜2018）の位置取り予想を、結果の表の過去走から作る代替（tp53_dataset._past_from_results）。"""
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
