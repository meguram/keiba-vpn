"""par_class_bucket / par_time_resolve のユニットテスト。"""

import unittest

import pandas as pd

from src.pipeline.megu_index.class_bucket import par_class_bucket
from src.pipeline.megu_index.par_time_resolve import attach_par_time_with_fallback


class TestParClassBucket(unittest.TestCase):
    def test_maiden(self):
        self.assertEqual(par_class_bucket("未勝利", "3歳未勝利", ""), "未勝利")

    def test_g1(self):
        self.assertEqual(par_class_bucket("G1", "有馬記念(G1)", ""), "G1")


class TestParTimeResolve(unittest.TestCase):
    def test_class_specific_par_preferred(self):
        df = pd.DataFrame([{
            "distance": 1000, "surface": "ダート", "direction": "右",
            "track_cat": "良", "class_bucket": "未勝利",
        }])
        df_par = pd.DataFrame([
            {"distance": 1000, "course": "右", "surface": "ダート", "track_condition": "良",
             "class_bucket": "未勝利", "par_time_sec": 58.0, "par_front_split_sec": 35.0},
            {"distance": 1000, "course": "右", "surface": "ダート", "track_condition": "良",
             "class_bucket": "", "par_time_sec": 57.0, "par_front_split_sec": 34.0},
        ])
        out = attach_par_time_with_fallback(df, df_par)
        self.assertAlmostEqual(float(out.iloc[0]["par_time_final"]), 58.0)

    def test_track_condition_fallback(self):
        df = pd.DataFrame([{
            "distance": 1500, "surface": "芝", "direction": "右",
            "track_cat": "重・不良", "class_bucket": "未勝利",
        }])
        df_par = pd.DataFrame([
            {"distance": 1500, "course": "右", "surface": "芝", "track_condition": "良",
             "class_bucket": "未勝利", "par_time_sec": 90.0, "par_front_split_sec": 35.0},
        ])
        out = attach_par_time_with_fallback(df, df_par)
        self.assertAlmostEqual(float(out.iloc[0]["par_time_final"]), 90.0)
        self.assertTrue(str(out.iloc[0]["par_match_level"]).startswith("L5_"))


if __name__ == "__main__":
    unittest.main()
