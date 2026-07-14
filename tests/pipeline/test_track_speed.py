"""track_speed ユーティリティのユニットテスト。"""

import unittest

import pandas as pd

from src.pipeline.megu_index.track_speed import (
    assign_class_group,
    attach_par_2nd_baseline,
    attach_track_speed_to_horses,
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


class TestTrackSpeed(unittest.TestCase):
    def test_assign_class_group(self):
        self.assertEqual(assign_class_group("未勝利", ""), "未勝利")
        self.assertEqual(assign_class_group("2勝", ""), "1-3勝")
        self.assertEqual(assign_class_group("G1", ""), "OP")

    def test_build_race_table_uses_second_place(self):
        rows = [
            _horse_row("r1", 1, 95.0),
            _horse_row("r1", 2, 96.0),
            _horse_row("r1", 3, 97.0),
        ]
        races = build_race_table(pd.DataFrame(rows))
        self.assertEqual(len(races), 1)
        self.assertAlmostEqual(races.iloc[0]["t2nd_adj_sec"], 96.0)

    def test_par_baseline_from_train_only(self):
        rows = []
        for i in range(35):
            rid = f"train_{i}"
            rows.extend(
                [
                    _horse_row(rid, 1, 94.0, year=2023, date="2023-06-01"),
                    _horse_row(rid, 2, 96.0, year=2023, date="2023-06-01"),
                ]
            )
        rows.extend(
            [
                _horse_row("test_1", 1, 94.0, year=2025),
                _horse_row("test_1", 2, 98.0, year=2025),
            ]
        )
        df = pd.DataFrame(rows)
        races = build_race_table(df)
        year_by_race = df.drop_duplicates("race_id").set_index("race_id")["year"]
        races["year"] = races["race_id"].map(year_by_race)
        races = attach_par_2nd_baseline(races, races["year"].isin([2023]), min_samples=30)
        races["race_track_dev_sec"] = races["t2nd_adj_sec"] - races["par_2nd_adj_sec"]
        test = races[races["race_id"] == "test_1"].iloc[0]
        self.assertAlmostEqual(test["par_2nd_adj_sec"], 96.0)
        self.assertAlmostEqual(test["race_track_dev_sec"], 2.0)

    def test_attach_track_speed_day_aggregate(self):
        rows = []
        for i, dev in enumerate([96.0, 96.5]):
            rid = f"r{i}"
            rows.extend(
                [
                    _horse_row(rid, 1, dev - 1.0, date="2024-06-01"),
                    _horse_row(rid, 2, dev, date="2024-06-01"),
                ]
            )
        for i in range(30):
            rid = f"base_{i}"
            rows.extend(
                [
                    _horse_row(rid, 1, 94.0, year=2023, date="2023-05-01"),
                    _horse_row(rid, 2, 96.0, year=2023, date="2023-05-01"),
                ]
            )
        df = pd.DataFrame(rows)
        out, races, day = attach_track_speed_to_horses(df, train_years=[2023], min_samples=5)
        self.assertIn("tsi_raw", out.columns)
        self.assertIn("track_dev_sec", out.columns)
        day_eval = day[day["date_str"] == "2024-06-01"]
        self.assertEqual(len(day_eval), 1)
        self.assertAlmostEqual(day_eval.iloc[0]["track_dev_sec"], 0.25, places=2)
        self.assertAlmostEqual(out["tsi_raw"].iloc[0], -0.25, places=2)

    def test_day_aggregate_ignores_direction(self):
        rows = []
        for rid, direction, t2 in [("r0", "右", 96.0), ("r1", "", 98.0)]:
            rows.extend(
                [
                    _horse_row(rid, 1, t2 - 1.0, date="2024-06-01", direction=direction),
                    _horse_row(rid, 2, t2, date="2024-06-01", direction=direction),
                ]
            )
        for i in range(30):
            rid = f"base_{i}"
            rows.extend(
                [
                    _horse_row(rid, 1, 94.0, year=2023, date="2023-05-01"),
                    _horse_row(rid, 2, 96.0, year=2023, date="2023-05-01"),
                ]
            )
        df = pd.DataFrame(rows)
        _, _, day = attach_track_speed_to_horses(df, train_years=[2023], min_samples=5)
        day_eval = day[day["date_str"] == "2024-06-01"]
        self.assertEqual(len(day_eval), 1)
        self.assertEqual(int(day_eval.iloc[0]["n_races_track"]), 2)

    def test_obstacle_races_excluded_from_baseline(self):
        rows = []
        for i in range(35):
            rid = f"train_{i}"
            rows.extend(
                [
                    _horse_row(rid, 1, 94.0, year=2023, date="2023-06-01"),
                    _horse_row(rid, 2, 96.0, year=2023, date="2023-06-01"),
                ]
            )
        rows.extend(
            [
                _horse_row("jump_1", 1, 120.0, year=2023, date="2023-06-02", surface="障"),
                _horse_row("jump_1", 2, 130.0, year=2023, date="2023-06-02", surface="障"),
            ]
        )
        races = build_race_table(pd.DataFrame(rows))
        self.assertFalse((races["race_id"] == "jump_1").any())
        self.assertEqual(len(races), 35)

    def test_obstacle_rows_get_no_track_correction_values(self):
        rows = [
            _horse_row("jump_1", 1, 120.0, surface="障"),
            _horse_row("jump_1", 2, 130.0, surface="障"),
            _horse_row("turf_1", 1, 94.0, year=2023, date="2023-06-01"),
            _horse_row("turf_1", 2, 96.0, year=2023, date="2023-06-01"),
        ]
        for i in range(30):
            rid = f"base_{i}"
            rows.extend(
                [
                    _horse_row(rid, 1, 94.0, year=2023, date="2023-05-01"),
                    _horse_row(rid, 2, 96.0, year=2023, date="2023-05-01"),
                ]
            )
        out, _, _ = attach_track_speed_to_horses(pd.DataFrame(rows), train_years=[2023], min_samples=5)
        jump = out[out["race_id"] == "jump_1"].iloc[0]
        self.assertTrue(pd.isna(jump["tsi_raw"]))
        self.assertTrue(pd.isna(jump["track_dev_sec"]))
        self.assertFalse(needs_track_speed_correction("障"))

    def test_rerun_is_idempotent(self):
        rows = []
        for i in range(30):
            rid = f"base_{i}"
            rows.extend(
                [
                    _horse_row(rid, 1, 94.0, year=2023, date="2023-05-01"),
                    _horse_row(rid, 2, 96.0, year=2023, date="2023-05-01"),
                ]
            )
        rows.extend([_horse_row("r0", 1, 95.0), _horse_row("r0", 2, 96.0)])
        df = pd.DataFrame(rows)
        out1, _, _ = attach_track_speed_to_horses(df, train_years=[2023], min_samples=5)
        out2, _, _ = attach_track_speed_to_horses(out1, train_years=[2023], min_samples=5)
        self.assertIn("race_track_dev_sec", out2.columns)
        self.assertNotIn("race_track_dev_sec_x", out2.columns)


if __name__ == "__main__":
    unittest.main()
