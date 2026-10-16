"""SmartRC テン1F 特徴量(smartrc_ten1f)の単体テスト。先読み（当該レース以降の戦績）が混ざらないことを中心に確かめる。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.features import smartrc_ten1f as S


def _fr(hid: str, rcode: str, rdate: str, ten: str, c1: str = "3", entry: str = "10", rng: str = "1600", tk: str = "0") -> dict:
    return {"horse_id": hid, "rcode": rcode, "rdate": rdate, "place": "05", "trackkind": tk, "range": rng, "ten1f": ten, "corner1": c1, "corner2": "3",
            "corner3": "3", "corner4": "3", "entry": entry, "uno": "1", "rank": "1", "stime": "1340", "furlong3": "340", "f3rank": "1", "interval": "14"}


def _entries(rows):
    return pd.DataFrame(rows, columns=["race_id", "horse_id", "date", "distance", "surface"])


def test_parse_ten1f():
    assert S.parse_ten1f("132") == 13.2 and S.parse_ten1f(" 125 ") == 12.5
    assert S.parse_ten1f("") is None and S.parse_ten1f("   ") is None and S.parse_ten1f(None) is None
    assert S.parse_ten1f("000") is None and S.parse_ten1f("999") is None and S.parse_ten1f("abc") is None


def test_features_use_only_strictly_earlier_races():
    career = S.career_table([
        _fr("H1", "R0", "20230101", "140"), _fr("H1", "R1", "20230201", "130"),
        _fr("H1", "RT", "20230301", "100"),          # 当該レース当日 → 特徴量に入れてはいけない
        _fr("H1", "RF", "20230401", "100"),          # 未来 → 入れてはいけない
    ])
    f = S.build_features(_entries([("2023T", "H1", "2023-03-01", 1600, "芝")]), career)
    row = f.iloc[0]
    assert row["sx_ten_n"] == 2
    assert row["sx_ten_best"] == 13.0 and row["sx_ten_last"] == 13.0           # 10.0 秒（当日・未来）が混ざると 10.0 になる
    assert abs(row["sx_ten_mean5"] - 13.5) < 1e-9 and abs(row["sx_ten_mean3"] - 13.5) < 1e-9


def test_target_uses_only_the_race_day_row_and_is_relative_to_race():
    career = S.career_table([_fr("A", "R1", "20230301", "130"), _fr("B", "R1", "20230301", "140"), _fr("C", "R1", "20230301", "120"),
                             _fr("A", "R0", "20230101", "110")])
    ent = _entries([("2023T", h, "2023-03-01", 1600, "芝") for h in "ABC"])
    t = S.race_targets(ent, career).set_index("horse_id")
    assert t.loc["A", "ten_actual"] == 13.0 and t.loc["C", "ten_actual"] == 12.0
    assert abs(t["ten_rel_actual"].sum()) < 1e-9 and abs(t.loc["B", "ten_rel_actual"] - 1.0) < 1e-9


def test_target_is_missing_when_race_coverage_is_low():
    career = S.career_table([_fr("A", "R1", "20230301", "130")])
    ent = _entries([("2023T", h, "2023-03-01", 1600, "芝") for h in "ABCDE"])
    assert S.race_targets(ent, career)["ten_rel_actual"].isna().all()


def test_missing_history_is_nan_not_zero_and_counts_are_zero():
    career = S.career_table([_fr("A", "R0", "20230101", "")])      # テンが無い過去走
    f = S.build_features(_entries([("2023T", "A", "2023-03-01", 1600, "芝"), ("2023T", "NEW", "2023-03-01", 1600, "芝")]), career)
    assert (f["sx_ten_n"] == 0).all() and f["sx_ten_mean3"].isna().all() and f["sx_ten_best"].isna().all()
    assert f.set_index("horse_id").loc["A", "sx_n_prior"] == 1 and f.set_index("horse_id").loc["NEW", "sx_n_prior"] == 0


def test_join_is_by_horse_id_not_by_position_or_name():
    career = S.career_table([_fr("B", "R0", "20230101", "120"), _fr("A", "R0", "20230101", "140")])
    ent = _entries([("2023T", "A", "2023-03-01", 1600, "芝"), ("2023T", "B", "2023-03-01", 1600, "芝")])
    f = S.build_features(ent, career).set_index("horse_id")
    assert f.loc["A", "sx_ten_last"] == 14.0 and f.loc["B", "sx_ten_last"] == 12.0


def test_similar_condition_mean_uses_same_surface_and_distance_band():
    career = S.career_table([_fr("A", "R0", "20230101", "120", rng="1600", tk="0"), _fr("A", "R1", "20230102", "150", rng="1600", tk="1"),
                             _fr("A", "R2", "20230103", "160", rng="2400", tk="0")])
    f = S.build_features(_entries([("2023T", "A", "2023-03-01", 1600, "芝")]), career).iloc[0]
    assert f["sx_ten_sim_n"] == 1 and f["sx_ten_sim_mean"] == 12.0


def test_race_relative_features_are_computed_within_the_race_only():
    career = S.career_table([_fr(h, "R0", "20230101", t) for h, t in (("A", "120"), ("B", "125"), ("C", "140"), ("D", "135"))])
    ent = _entries([("2023T", h, "2023-03-01", 1600, "芝") for h in "ABCD"])
    f = S.build_features(ent, career).set_index("horse_id")
    assert f.loc["A", "sx_rel_gap_fast"] == 0 and abs(f.loc["C", "sx_rel_gap_fast"] - 2.0) < 1e-9
    assert f["sx_race_n_fast03"].iloc[0] == 1          # 最速 12.0 から 0.3 秒以内は A だけ（B は 12.5）


def test_first_recorded_corner_is_used_when_corner1_is_absent():
    # 1200m など: corner1・2 が 0、通過順は corner3・4 から
    row = _fr("A", "R0", "20230101", "125", c1="0", entry="10")
    row.update({"corner2": "0", "corner3": "4", "corner4": "2"})
    career = S.career_table([row])
    assert career.loc[0, "c1"] == 4.0 and abs(career.loc[0, "c1_rel"] - 0.4) < 1e-9
    none = _fr("B", "R0", "20230101", "125", c1="0")
    none.update({"corner2": "0", "corner3": "0", "corner4": "0"})
    assert np.isnan(S.career_table([none]).loc[0, "c1"])
