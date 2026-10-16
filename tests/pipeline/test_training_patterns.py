"""追い切りパターン採点のテスト（as-of・調教師キー）。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pipeline.features import training_laps as tl
from src.pipeline.features import training_patterns as tp


def _laps(n_horses=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(n_horses):
        for r, race_date in enumerate(pd.to_datetime(["2022-03-06", "2022-04-03", "2022-05-01"])):
            for days, intensity in ((3, "一杯"), (10, "強め"), (17, "馬也")):
                t1 = 12.4 + rng.normal(0, 0.3)
                t3 = 38.5 + rng.normal(0, 0.5)
                t4 = 52.5 + rng.normal(0, 0.6)
                rows.append({"horse_id": f"h{h}", "train_date": (race_date - pd.Timedelta(days=days)).strftime("%Y-%m-%d"), "race_date": race_date.strftime("%Y-%m-%d"), "course": "栗坂", "kind": "slope",
                             "center": "栗", "cond": "良", "rider": "助手", "position": np.nan, "leg_color": intensity, "rank": "B", "evaluation": "", "t1": t1, "t2": t1 + 12.5, "t3": t4 - 14 + 2 if False else t3, "t4": t4, "t5": np.nan, "t6": np.nan})
    return tl.add_lap_features(pd.DataFrame(rows))


class RaceFeatureTest(unittest.TestCase):
    def test_picks_final_and_prev_week(self):
        t = tp.prepare(_laps())
        t = tp.add_reference_z(t, window_months=12, min_n=1)
        f = tp.race_features(t)
        self.assertEqual(int(f["f_days"].iloc[0]), 3)
        self.assertEqual(int(f["w1_days"].iloc[0]), 10)
        self.assertEqual(int(f["w2_days"].iloc[0]), 17)
        self.assertEqual(float(f["f_int"].iloc[0]), 2.0)
        self.assertEqual(float(f["w1_int"].iloc[0]), 1.0)
        self.assertTrue((f["n_work_14"] == 2).all())

    def test_reference_is_asof(self):
        t = tp.prepare(_laps())
        z = tp.add_reference_z(t, min_n=1)
        first_month = z[z["train_date"] < "2022-03-01"]
        self.assertTrue(first_month["t4z"].isna().all())           # 過去 12 か月に何も無い月は標準化できない


class TrainerScoreTest(unittest.TestCase):
    def test_same_day_not_used(self):
        d = pd.DataFrame({"date": ["2022-01-01"] * 600 + ["2022-01-02"] * 2, "trainer_id": ["T1"] * 602, "pattern": ["p"] * 602, "finish_position": [1] * 600 + [1, 5]})
        s = tp.trainer_pattern_scores(d)
        self.assertTrue(s["tp_top3"].iloc[:600].isna().all())     # 最初の日は履歴が無い
        self.assertAlmostEqual(float(s["tp_n"].iloc[600]), 600.0)
        self.assertAlmostEqual(float(s["tp_n"].iloc[601]), 600.0)  # 同日の別レースの結果は入らない

    def test_trainer_specific_edge(self):
        rng = np.random.default_rng(0)
        rows = []
        for day in range(1, 400):
            dt = (pd.Timestamp("2021-01-01") + pd.Timedelta(days=day)).strftime("%Y-%m-%d")
            for tr in ("A", "B"):
                for pat in ("x", "y"):
                    good = (tr == "A") == (pat == "x")                     # A は x が得意、B は y が得意
                    for _ in range(3):
                        rows.append({"date": dt, "trainer_id": tr, "pattern": pat, "finish_position": int(rng.choice([1, 8], p=[0.6, 0.4] if good else [0.1, 0.9]))})
        d = pd.DataFrame(rows).sort_values("date", kind="stable").reset_index(drop=True)
        s = tp.trainer_pattern_scores(d)
        tail = d.index[-600:]
        a = s.loc[tail][(d.loc[tail, "trainer_id"] == "A") & (d.loc[tail, "pattern"] == "x")]["tp_edge"].mean()
        b = s.loc[tail][(d.loc[tail, "trainer_id"] == "B") & (d.loc[tail, "pattern"] == "x")]["tp_edge"].mean()
        self.assertGreater(a, b + 0.05)


class PatternKeyTest(unittest.TestCase):
    def test_key(self):
        f = pd.DataFrame({"f_accel": [-0.8, 0.0, 0.9, np.nan], "f_kind": [0, 0, 1, 0], "f_int": [2, 1, 0, 1], "w1_int": [1.0, np.nan, 0.0, 1.0], "f_days": [1, 3, 6, 2]})
        k = tp.pattern_key(f)
        self.assertEqual(k.iloc[0], "0|2|A|h|s")
        self.assertEqual(k.iloc[1], "0|1|F|n|m")
        self.assertEqual(k.iloc[2], "1|0|D|e|l")
        self.assertEqual(k.iloc[3], "")


if __name__ == "__main__":
    unittest.main()
