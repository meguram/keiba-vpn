import datetime as dt
import unittest

import numpy as np
import pandas as pd

from src.pipeline.features import tp68_lib as L


class TestCalendar(unittest.TestCase):
    def test_holiday_counts_and_substitutes(self):
        self.assertEqual(len(L.jp_holidays(2024)), 21)
        self.assertIn(dt.date(2026, 5, 6), L.jp_holidays(2026))        # 5/3 が日曜 → 5/6 が振替
        self.assertIn(dt.date(2026, 9, 22), L.jp_holidays(2026))       # 敬老の日と秋分の日の間（国民の休日）
        self.assertIn(dt.date(2020, 7, 23), L.jp_holidays(2020))       # 五輪で海の日が移動
        self.assertNotIn(dt.date(2020, 7, 20), L.jp_holidays(2020))
        self.assertIn(dt.date(2021, 8, 9), L.jp_holidays(2021))        # 山の日の振替

    def test_run_features(self):
        d = pd.Series(pd.to_datetime(["2026-10-10", "2026-10-11", "2026-10-12", "2026-10-13", "2026-10-17"]))
        f = L.calendar_features(d)
        self.assertEqual(list(f["cal_run_len"]), [3, 3, 3, 0, 2])
        self.assertEqual(list(f["cal_day_in_run"]), [1, 2, 3, 0, 1])
        self.assertEqual(list(f["cal_is_holiday"]), [0, 0, 1, 0, 0])
        self.assertEqual(list(f["cal_after_run"]), [0, 0, 0, 1, 0])


class TestSelection(unittest.TestCase):
    def test_permute_within_race_keeps_race_counts(self):
        race = np.repeat(np.arange(50), 10)
        y = np.zeros(500, int)
        y[::10] = 1
        yp = L.permute_within_race(y, race, np.random.default_rng(0))
        self.assertTrue(all(yp[race == r].sum() == 1 for r in range(50)))
        self.assertFalse((yp == y).all())

    def test_rfe_keeps_every_group(self):
        g = pd.Series({"a_1": 9.0, "a_2": 8.0, "a_3": 7.0, "b_1": 0.2, "b_2": 0.1, "c_1": 0.05})
        keep = L.rfe_columns(g, 3, min_per_group=1)
        self.assertEqual(set(keep), {"a_1", "a_2", "a_3", "b_1", "c_1"})
        self.assertEqual(len(L.rfe_columns(g, 6)), 6)

    def test_null_summary(self):
        real = pd.Series({"x": 10.0, "y": 0.1})
        null = np.vstack([np.array([1.0, 0.5])] * 20)
        S = L.null_summary(real, null, ["x", "y"]).set_index("col")
        self.assertEqual(S.loc["x", "frac_null_ge_real"], 0.0)
        self.assertEqual(S.loc["y", "frac_null_ge_real"], 1.0)

    def test_missing_features(self):
        X = pd.DataFrame({"a_1": [1, np.nan], "a_2": [np.nan, np.nan], "b_1": [1, 2]})
        M = L.missing_features(X, {"a": ["a_1", "a_2"]})
        self.assertEqual(list(M["miss_n_all"]), [1, 2])
        self.assertEqual(list(M["miss_frac_a"]), [0.5, 1.0])


class TestSoftLabel(unittest.TestCase):
    def test_weights_sum_to_one_per_unlabeled_row(self):
        Xl = pd.DataFrame({"f": [1.0, 2.0]})
        Xu = pd.DataFrame({"f": [3.0, 4.0, 5.0]})
        yl = {"win": np.array([1, 0]), "top2": np.array([1, 1]), "top3": np.array([1, 1])}
        pu = {"win": np.array([0.1, 0.2, 0.3]), "top2": np.array([0.3, 0.4, 0.5]), "top3": np.array([0.5, 0.6, 0.7])}
        X, y, w = L.soft_label_augment(Xl, yl, Xu, pu)
        self.assertEqual(len(X), 2 + 3 + 3)
        for k in yl:
            self.assertEqual(len(y[k]), len(X))
            self.assertTrue(np.allclose(w[k][2:5] + w[k][5:], 1.0))
            self.assertTrue(np.allclose(w[k][:2], 1.0))
            self.assertEqual(list(y[k][2:5]), [1, 1, 1])
            self.assertEqual(list(y[k][5:]), [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
