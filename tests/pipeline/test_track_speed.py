"""track_speed ユーティリティのユニットテスト。"""

import unittest

import numpy as np
import pandas as pd

from src.pipeline.megu_index.track_speed import (
    DEFAULT_K_SHRINKAGE,
    assign_class_group,
    attach_pace_filter,
    attach_par_2nd_baseline,
    attach_track_speed_to_horses,
    build_pace_filter_params,
    build_race_table,
    needs_track_speed_correction,
)


def _horse_row(race_id, finish_pos, adj, **kw):
    base = {
        "race_id": race_id,
        "date": "2024-06-01",
        "venue": "東京",
        "surface": "芝",
        "direction": "左",
        "distance": 1600,
        "grade": "1勝",
        "race_class": "1勝クラス",
        "finish_pos": finish_pos,
        "adjusted_time_sec": adj,
        "year": 2024,
    }
    base.update(kw)
    return base


def _base_rows(n=30, year=2023, date="2023-05-01", t2=96.0):
    rows = []
    for i in range(n):
        rows.extend([
            _horse_row(f"base_{i}", 1, t2 - 1.0, year=year, date=date),
            _horse_row(f"base_{i}", 2, t2,        year=year, date=date),
        ])
    return rows


class TestClassGroup(unittest.TestCase):
    def test_six_groups(self):
        self.assertEqual(assign_class_group("未勝利", ""), "未勝利")
        self.assertEqual(assign_class_group("1勝",   ""), "1勝")
        self.assertEqual(assign_class_group("2勝",   ""), "2勝")
        self.assertEqual(assign_class_group("3勝",   ""), "3勝")
        self.assertEqual(assign_class_group("G3",    ""), "G3orOP")
        self.assertEqual(assign_class_group("OP",    ""), "G3orOP")
        self.assertEqual(assign_class_group("G2",    ""), "G1orG2")
        self.assertEqual(assign_class_group("G1",    ""), "G1orG2")

    def test_age_restricted_returns_none(self):
        self.assertIsNone(assign_class_group("未勝利", "サラ系２歳 未勝利"))
        self.assertIsNone(assign_class_group("未勝利", "3歳 未勝利"))
        self.assertIsNone(assign_class_group("新馬",   "新馬"))
        # 3歳以上 は世代限定でないため None にならない
        self.assertEqual(assign_class_group("1勝", "3歳以上1勝クラス"), "1勝")


class TestBuildRaceTable(unittest.TestCase):
    def test_uses_second_place(self):
        rows = [
            _horse_row("r1", 1, 95.0),
            _horse_row("r1", 2, 96.0),
            _horse_row("r1", 3, 97.0),
        ]
        races = build_race_table(pd.DataFrame(rows))
        self.assertEqual(len(races), 1)
        self.assertAlmostEqual(races.iloc[0]["t2nd_adj_sec"], 96.0)

    def test_obstacle_races_excluded(self):
        rows = []
        for i in range(35):
            rows.extend([
                _horse_row(f"t_{i}", 1, 94.0, year=2023, date="2023-06-01"),
                _horse_row(f"t_{i}", 2, 96.0, year=2023, date="2023-06-01"),
            ])
        rows.extend([
            _horse_row("jump_1", 1, 120.0, year=2023, date="2023-06-02", surface="障"),
            _horse_row("jump_1", 2, 130.0, year=2023, date="2023-06-02", surface="障"),
        ])
        races = build_race_table(pd.DataFrame(rows))
        self.assertFalse((races["race_id"] == "jump_1").any())
        self.assertEqual(len(races), 35)

    def test_front_split_sec_passed_through(self):
        rows = [
            _horse_row("r1", 1, 95.0),
            _horse_row("r1", 2, 96.0),
        ]
        df = pd.DataFrame(rows)
        df["front_split_sec"] = 48.5
        races = build_race_table(df)
        self.assertIn("front_split_sec", races.columns)
        self.assertAlmostEqual(races.iloc[0]["front_split_sec"], 48.5)


class TestParBaseline(unittest.TestCase):
    def test_from_train_only(self):
        rows = _base_rows(n=35, year=2023, date="2023-06-01")
        rows.extend([
            _horse_row("test_1", 1, 94.0, year=2025),
            _horse_row("test_1", 2, 98.0, year=2025),
        ])
        df = pd.DataFrame(rows)
        races = build_race_table(df)
        year_by_race = df.drop_duplicates("race_id").set_index("race_id")["year"]
        races["year"] = races["race_id"].map(year_by_race)
        races = attach_par_2nd_baseline(races, races["year"].isin([2023]), min_samples=30)
        races["race_track_dev_sec"] = races["t2nd_adj_sec"] - races["par_2nd_adj_sec"]
        test = races[races["race_id"] == "test_1"].iloc[0]
        self.assertAlmostEqual(test["par_2nd_adj_sec"], 96.0)
        self.assertAlmostEqual(test["race_track_dev_sec"], 2.0)


class TestPaceFilter(unittest.TestCase):
    def test_filter_removes_pace_outlier(self):
        # 学習データ: front_split_sec が 48.0 ± 1.0 程度（σ≒1）
        rng = np.random.default_rng(0)
        rows = []
        for i in range(20):
            rows.append({
                "race_id": f"tr_{i}",
                "surface": "芝",
                "distance": 1600,
                "front_split_sec": 48.0 + float(rng.normal(0, 1.0)),
            })
        df_train = pd.DataFrame(rows)
        params = build_pace_filter_params(df_train)
        self.assertFalse(params.empty)

        # レース: 正常ペースと異常ペース
        races = pd.DataFrame({
            "race_id": ["ok", "out"],
            "surface": ["芝", "芝"],
            "distance": [1600, 1600],
            "front_split_sec": [48.0, 60.0],  # 60秒はσ=2σを超える
        })
        result = attach_pace_filter(races, params, n_sigma=2.0)
        self.assertIn("is_pace_valid", result.columns)
        self.assertTrue(result.loc[result["race_id"] == "ok", "is_pace_valid"].iloc[0])
        self.assertFalse(result.loc[result["race_id"] == "out", "is_pace_valid"].iloc[0])

    def test_no_front_split_all_valid(self):
        races = pd.DataFrame({
            "race_id": ["r1", "r2"],
            "surface": ["芝", "芝"],
            "distance": [1600, 1600],
        })
        params = pd.DataFrame(columns=["distance", "surface", "front_split_median", "front_split_sigma"])
        result = attach_pace_filter(races, params)
        self.assertTrue(result["is_pace_valid"].all())
        self.assertTrue(result["front_split_dev"].isna().all())


class TestDayAggregate(unittest.TestCase):
    def test_shrinkage_applied_with_small_n(self):
        """n=2レースのとき収縮で prior 方向に引き寄せられる (0 < result < observed)。"""
        rows = []
        for i, dev in enumerate([96.0, 96.5]):
            rows.extend([
                _horse_row(f"r{i}", 1, dev - 1.0, date="2024-06-01"),
                _horse_row(f"r{i}", 2, dev,        date="2024-06-01"),
            ])
        rows.extend(_base_rows(n=30))
        df = pd.DataFrame(rows)
        out, races, day = attach_track_speed_to_horses(df, train_years=[2023], min_samples=5)

        self.assertIn("tsi_raw",       out.columns)
        self.assertIn("track_dev_sec", out.columns)
        self.assertIn("n_valid_races", out.columns)

        day_eval = day[day["date_str"] == "2024-06-01"]
        self.assertEqual(len(day_eval), 1)

        observed_median = 0.25  # median([0.0, 0.5])
        k   = DEFAULT_K_SHRINKAGE
        w   = 2.0 / (2.0 + k)
        # prior が 0 に近い（学習期 track_dev = 0 のはず）ので shrunk < observed
        result = day_eval.iloc[0]["track_dev_sec"]
        self.assertAlmostEqual(result, w * observed_median, places=4)
        self.assertAlmostEqual(day_eval.iloc[0]["tsi_raw"], -result, places=6)
        self.assertEqual(int(day_eval.iloc[0]["n_valid_races"]), 2)

    def test_aggregate_ignores_direction(self):
        rows = []
        for rid, direction, t2 in [("r0", "右", 96.0), ("r1", "", 98.0)]:
            rows.extend([
                _horse_row(rid, 1, t2 - 1.0, date="2024-06-01", direction=direction),
                _horse_row(rid, 2, t2,        date="2024-06-01", direction=direction),
            ])
        rows.extend(_base_rows(n=30))
        df = pd.DataFrame(rows)
        _, _, day = attach_track_speed_to_horses(df, train_years=[2023], min_samples=5)
        day_eval = day[day["date_str"] == "2024-06-01"]
        self.assertEqual(len(day_eval), 1)
        self.assertEqual(int(day_eval.iloc[0]["n_races_track"]), 2)

    def test_pace_invalid_race_excluded_from_median(self):
        """ペース異常レースが除外されると n_valid_races < n_races_track になる。"""
        rows = []
        # 正常: dev=0.0
        rows.extend([
            _horse_row("r_ok", 1, 95.0, date="2024-06-01"),
            _horse_row("r_ok", 2, 96.0, date="2024-06-01"),
        ])
        # 異常ペース用: front_split_sec を後から splits_df で渡す
        rows.extend([
            _horse_row("r_out", 1, 99.0, date="2024-06-01"),
            _horse_row("r_out", 2, 100.0, date="2024-06-01"),
        ])
        rows.extend(_base_rows(n=30))
        df = pd.DataFrame(rows)

        # splits_df: r_ok は正常ペース(48.0)、r_out は極端な外れ値(80.0)
        rng = np.random.default_rng(0)
        train_splits = pd.DataFrame([
            {"race_id": f"base_{i}", "front_split_sec": 48.0 + float(rng.normal(0, 0.5))}
            for i in range(30)
        ])
        test_splits = pd.DataFrame([
            {"race_id": "r_ok",  "front_split_sec": 48.0},
            {"race_id": "r_out", "front_split_sec": 80.0},  # 外れ値
        ])
        splits = pd.concat([train_splits, test_splits], ignore_index=True)

        _, _, day = attach_track_speed_to_horses(df, train_years=[2023], min_samples=5, splits_df=splits)
        day_eval = day[day["date_str"] == "2024-06-01"].iloc[0]
        self.assertEqual(int(day_eval["n_races_track"]), 2)
        self.assertLessEqual(int(day_eval["n_valid_races"]), 2)


class TestObstacle(unittest.TestCase):
    def test_obstacle_rows_get_no_track_values(self):
        rows = [
            _horse_row("jump_1", 1, 120.0, surface="障"),
            _horse_row("jump_1", 2, 130.0, surface="障"),
            _horse_row("turf_1", 1, 94.0, year=2023, date="2023-06-01"),
            _horse_row("turf_1", 2, 96.0, year=2023, date="2023-06-01"),
        ]
        rows.extend(_base_rows(n=30))
        out, _, _ = attach_track_speed_to_horses(pd.DataFrame(rows), train_years=[2023], min_samples=5)
        jump = out[out["race_id"] == "jump_1"].iloc[0]
        self.assertTrue(pd.isna(jump["tsi_raw"]))
        self.assertTrue(pd.isna(jump["track_dev_sec"]))
        self.assertFalse(needs_track_speed_correction("障"))


class TestIdempotent(unittest.TestCase):
    def test_rerun_is_idempotent(self):
        rows = _base_rows(n=30)
        rows.extend([_horse_row("r0", 1, 95.0), _horse_row("r0", 2, 96.0)])
        df = pd.DataFrame(rows)
        out1, _, _ = attach_track_speed_to_horses(df, train_years=[2023], min_samples=5)
        out2, _, _ = attach_track_speed_to_horses(out1, train_years=[2023], min_samples=5)
        self.assertIn("race_track_dev_sec", out2.columns)
        self.assertNotIn("race_track_dev_sec_x", out2.columns)


if __name__ == "__main__":
    unittest.main()
