import json

from src.scraper.horse_training_local import horse_training_json_path
from src.scripts.data import build_training_laps_past as B


def test_build_reads_local_records_and_counts_absent(tmp_path):
    f = horse_training_json_path(tmp_path, "2005105204")
    f.parent.mkdir(parents=True)
    e = {"date": "2016-11-16", "race_info": "2016/11/20 福島記念", "course": "栗坂", "lap_times": [52.0, 38.0, 25.0, 12.5], "track_condition": "良"}
    f.write_text(json.dumps({"horse_id": "2005105204", "entries": [e, {"date": "", "race_info": ""}]}), encoding="utf-8")
    df, absent = B.build(["2005105204", "2005999999"], tmp_path)
    assert absent == 1 and len(df) == 1
    r = df.iloc[0]
    assert r["race_date"] == "2016-11-20" and r["kind"] == "slope" and r["t4"] == 52.0 and r["days_before"] == 4


def test_build_with_no_records_returns_empty(tmp_path):
    df, absent = B.build(["2005999999"], tmp_path)
    assert df.empty and absent == 1
