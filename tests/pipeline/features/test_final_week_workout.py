import pandas as pd

from src.pipeline.features import final_week_workout as W
from src.pipeline.features import tp35_lap_groups as L


def _laps(rows):
    d = pd.DataFrame(rows, columns=["horse_id", "train_date", "race_date", "course", "kind", "center", "t1", "t2", "t3", "t4", "days_before"])
    d["t5"] = d["t6"] = None
    return L.prepare_laps(d)


def test_best_time_in_race_week_weekdays_only():
    # 土曜開催 2024-06-08。週は 6/3(月)〜6/7(金)。前週の日曜 6/2 は含めない。土曜当日(0日前)も含めない。
    tp = _laps([
        ("h", "2024-06-02", "2024-06-08", "栗坂", "slope", "栗", 11.8, 24.0, 37.0, 50.0, 6),   # 前週日曜・最速でも対象外
        ("h", "2024-06-05", "2024-06-08", "栗坂", "slope", "栗", 12.5, 25.0, 38.0, 52.0, 3),
        ("h", "2024-06-06", "2024-06-08", "栗坂", "slope", "栗", 12.9, 26.0, 39.5, 54.0, 2),   # 新しいが遅い
    ])
    g = W.pick_best_weekday(tp)
    assert len(g) == 1 and g.loc[0, "BW_t4"] == 52.0 and g.loc[0, "BW_days"] == 3 and g.loc[0, "BW_n"] == 2


def test_monday_race_belongs_to_previous_week_and_no_leak():
    # 月曜開催 2024-06-10 の週は 6/3〜6/7。6/10 以降の調教は入らない。
    tp = _laps([
        ("h", "2024-06-05", "2024-06-10", "栗坂", "slope", "栗", 12.5, 25.0, 38.0, 52.0, 5),
        ("h", "2024-06-11", "2024-06-10", "栗坂", "slope", "栗", 11.0, 22.0, 33.0, 44.0, -1),
    ])
    g = W.pick_best_weekday(tp)
    assert len(g) == 1 and g.loc[0, "BW_t4"] == 52.0


def test_tie_prefers_newer():
    tp = _laps([
        ("h", "2024-06-04", "2024-06-08", "栗坂", "slope", "栗", 12.5, 25.0, 38.0, 52.0, 4),
        ("h", "2024-06-06", "2024-06-08", "栗坂", "slope", "栗", 12.5, 25.0, 38.0, 52.0, 2),
    ])
    assert W.pick_best_weekday(tp).loc[0, "BW_days"] == 2


def test_wood_band_boundaries_and_slope_uses_lap_class():
    t4 = pd.Series([49.9, 50.0, 51.9, 52.0, 53.9, 54.0, 55.9, 56.0, 57.9, 58.0])
    assert list(W.wood_band(t4)) == ["W~49.9", "W50.0-51.9", "W50.0-51.9", "W52.0-53.9", "W52.0-53.9", "W54.0-55.9", "W54.0-55.9", "W56.0-57.9", "W56.0-57.9", "W58.0~"]
    kind, cls, t = pd.Series(["slope", "wood", "poly"]), pd.Series(["A2", "B2", "A3"]), pd.Series([52.0, 52.0, 55.0])
    assert list(W.pattern_of(kind, cls, t)) == ["A2", "W52.0-53.9", "W54.0-55.9"]


def test_best_pick_carries_pattern():
    tp = _laps([("h", "2024-06-05", "2024-06-08", "CW", "wood", "栗", 11.8, None, 36.0, 50.8, 3)])
    g = W.pick_best_weekday(tp)
    assert g.loc[0, "BW_pat"] == "W50.0-51.9" and g.loc[0, "BW_place"] == "Cウッド(栗東)"


def test_midweek_race_excludes_post_race_workouts():
    # 水曜開催 2024-06-05 の週は 6/3(月)〜。レース日の後(6/6, 6/7)の調教と days_before<=0 は入らない。
    tp = _laps([
        ("h", "2024-06-04", "2024-06-05", "栗坂", "slope", "栗", 12.5, 25.0, 38.0, 52.0, 1),
        ("h", "2024-06-06", "2024-06-05", "栗坂", "slope", "栗", 11.0, 22.0, 33.0, 44.0, -1),
        ("h", "2024-06-05", "2024-06-05", "栗坂", "slope", "栗", 11.5, 23.0, 34.0, 46.0, 0),
    ])
    g = W.pick_best_weekday(tp)
    assert len(g) == 1 and g.loc[0, "BW_t4"] == 52.0 and g.loc[0, "BW_n"] == 1
