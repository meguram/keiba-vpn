"""compute_for_dataframe 品質ガードのユニットテスト。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pipeline.megu_index.common import adjusted_time_to_megu
from src.pipeline.megu_index.compute import compute_for_dataframe
from src.pipeline.megu_index.quality_check import passes_quality_gates, summarize_megu_quality


def _base_row(**kw):
    row = {
        "race_id": "202607181001",
        "horse_id": "2021100001",
        "date": "2026-07-18",
        "venue": "函館",
        "venue_code": "01",
        "surface": "芝",
        "distance": 1200,
        "direction": "右",
        "track_condition": "良",
        "grade": "1勝",
        "race_class": "3歳以上1勝クラス",
        "race_name": "3歳以上1勝クラス",
        "time_sec": 70.0,
        "jockey_weight": 55.0,
        "sex_age": "牡3",
        "lap_times": "35.0-34.5",
        "finish_position": 1,
    }
    row.update(kw)
    return row


class TestComputeQualityGuards(unittest.TestCase):
    def setUp(self):
        self.params = {
            "beta_pace": 0.79,
            "beta_track": 0.19,
            "beta_weight": 0.61,
            "beta_level": -0.93,
            "tsi_mean": 3.3,
        }
        self.df_par = pd.DataFrame([{
            "distance": 1200,
            "course": "右",
            "surface": "芝",
            "track_condition": "良",
            "class_bucket": "1勝",
            "par_time_sec": 70.5,
            "par_front_split_sec": 35.0,
        }])

    def test_obstacle_race_excluded(self):
        df = pd.DataFrame([
            _base_row(race_id="202607181099", race_name="障害3歳以上オープン", distance=3350, time_sec=221.0),
            _base_row(race_id="202607181002", horse_id="2021100002"),
        ])
        out = compute_for_dataframe(df, self.params, self.df_par)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["race_id"], "202607181002")

    def test_megu_uses_corrected_without_level(self):
        df = pd.DataFrame([_base_row(finish_position=2, time_sec=70.0)])
        out = compute_for_dataframe(df, self.params, self.df_par)
        row = out.iloc[0]
        expected = adjusted_time_to_megu(row["corrected_time_sec"], row["par_time_final"])
        self.assertAlmostEqual(float(row["megu_index"]), expected, places=4)
        self.assertIn("corrected_time_sec", out.columns)

    def test_bad_par_marked_no_par(self):
        df = pd.DataFrame([
            _base_row(time_sec=165.0, distance=2500, finish_position=1),
        ])
        bad_par = self.df_par.copy()
        bad_par["distance"] = 2500
        bad_par["par_time_sec"] = 56.0
        out = compute_for_dataframe(df, self.params, bad_par)
        self.assertEqual(out.iloc[0]["computation_status"], "no_par")
        self.assertTrue(pd.isna(out.iloc[0]["megu_index"]))

    def test_delta_pace_clipped(self):
        df = pd.DataFrame([_base_row(time_sec=165.0, lap_times="120.0-45.0")])
        par = pd.DataFrame([{
            "distance": 1200,
            "course": "右",
            "surface": "芝",
            "track_condition": "良",
            "class_bucket": "1勝",
            "par_time_sec": 70.5,
            "par_front_split_sec": 35.0,
        }])
        out = compute_for_dataframe(df, self.params, par)
        self.assertLessEqual(abs(float(out.iloc[0]["delta_pace_sec"])), 5.0 + 1e-9)


class TestQualityCheckHelpers(unittest.TestCase):
    def test_passes_gates_clean_sample(self):
        par = 70.5
        corrs = [70.0, 71.0, 69.5]
        megu_vals = [adjusted_time_to_megu(c, par) for c in corrs]
        df = pd.DataFrame({
            "computation_status": ["valid"] * 3,
            "megu_index": megu_vals,
            "corrected_time_sec": corrs,
            "par_time_final": [par] * 3,
            "race_id": ["R1", "R1", "R1"],
            "grade": ["1勝", "1勝", "1勝"],
            "race_class": ["1勝", "1勝", "1勝"],
            "race_name": ["1勝", "1勝", "1勝"],
            "finish_position": [1, 2, 3],
        })
        summary = summarize_megu_quality(df, label="test")
        ok, fails = passes_quality_gates(summary)
        self.assertTrue(ok, fails)


if __name__ == "__main__":
    unittest.main()
