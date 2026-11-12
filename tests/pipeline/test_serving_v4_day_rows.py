"""field7 の calendar_env に、同じ日の他のレースの行を足す（最終レース・1 レース目との馬場差などの同日集計に要る）。"""

from __future__ import annotations

import pandas as pd

from src.pipeline.features.serving_v4 import V4Groups as SV


def _rows(race_id, rnd, start, tc, n=2):
    return pd.DataFrame({"race_id": race_id, "horse_id": [f"{race_id}{k}" for k in range(n)], "date": "2025-02-02", "venue_code": "05", "surface": "芝",
                         "track_condition": tc, "round": rnd, "start_time": start, "weather": "晴", "horse_number": range(1, n + 1)})


def test_with_day_rows_adds_other_races_and_fills_blank_conditions():
    tdf = _rows("R12", 12, "16:00", "良")
    tdf["date"] = pd.to_datetime(tdf["date"])
    day = pd.concat([_rows("R01", 1, "10:00", ""), _rows("R12", 12, "16:00", "良")], ignore_index=True)
    out = SV._with_day_rows(tdf, day)
    assert set(out["race_id"]) == {"R01", "R12"} and len(out) == 4
    assert out.loc[out["race_id"] == "R01", "track_condition"].eq("良").all()
    assert out["round"].max() == 12 and out["start_time"].min() == "10:00"


def test_with_day_rows_is_noop_without_day_card_or_other_races():
    tdf = _rows("R12", 12, "16:00", "良")
    tdf["date"] = pd.to_datetime(tdf["date"])
    assert SV._with_day_rows(tdf, None) is tdf
    assert len(SV._with_day_rows(tdf, tdf.copy())) == len(tdf)


def test_with_day_rows_does_not_fill_blank_going_from_another_surface():
    tdf = _rows("R11", 11, "15:00", "良")
    tdf["date"] = pd.to_datetime(tdf["date"])
    other = _rows("R01", 1, "10:00", "").assign(surface="ダート")
    out = SV._with_day_rows(tdf, pd.concat([other, tdf], ignore_index=True))
    assert out.loc[out["race_id"] == "R01", "track_condition"].isna().all()
