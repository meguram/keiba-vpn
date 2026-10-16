"""追い切り表の整形テスト。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pipeline.features import training_laps as tl


class ParseTest(unittest.TestCase):
    def test_course(self):
        self.assertEqual(tl.normalize_course("美Ｗ")[1:], ("wood", "美"))
        self.assertEqual(tl.normalize_course("ＣＷ")[1:], ("wood", "栗"))
        self.assertEqual(tl.normalize_course("南Ｗ")[1:], ("wood", "美"))
        self.assertEqual(tl.normalize_course("栗坂")[1:], ("slope", "栗"))
        self.assertEqual(tl.normalize_course("美芝")[1], "other")

    def test_laps_slope_and_wood(self):
        self.assertEqual(tl.assign_laps([52.7, 38.4, 25.2, 12.3], "slope"), {"t4": 52.7, "t3": 38.4, "t2": 25.2, "t1": 12.3})
        w = tl.assign_laps([84.6, 67.5, 52.4, 37.8, 11.3], "wood")
        self.assertEqual((w["t6"], w["t5"], w["t4"], w["t3"], w["t1"]), (84.6, 67.5, 52.4, 37.8, 11.3))
        self.assertNotIn("t2", w)
        w2 = tl.assign_laps([66.2, 51.5, 37.3, 11.4], "wood")          # 6F が無い行
        self.assertEqual((w2["t5"], w2["t4"], w2["t3"], w2["t1"]), (66.2, 51.5, 37.3, 11.4))

    def test_inconsistent_dropped(self):
        self.assertEqual(tl.assign_laps([52.7, 58.0, 25.2, 12.3], "slope"), {})

    def test_row_and_features(self):
        e = {"date": "2023-11-19", "course": "美坂", "track_condition": "良", "rider": "助手", "lap_times": [52.7, 38.4, 25.2, 12.3], "position": "", "leg_color": "馬也",
             "rank": "C", "race_info": "2023/11/26  東京12R ジャパンカップ 結果 ： 1着", "evaluation": "順調"}
        r = tl.entry_to_row("H1", e)
        self.assertEqual(r["race_date"], "2023-11-26")
        df = tl.add_lap_features(pd.DataFrame([r]))
        self.assertEqual(int(df["days_before"].iloc[0]), 7)
        self.assertAlmostEqual(float(df["lmid"].iloc[0]), (38.4 - 12.3) / 2)
        self.assertAlmostEqual(float(df["accel"].iloc[0]), 12.3 - 13.05)
        self.assertTrue(np.isnan(r["position"]))


if __name__ == "__main__":
    unittest.main()
