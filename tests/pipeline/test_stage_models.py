"""tp-04 前段モデルの部品のテスト（as-of 集計に未来が混ざらないこと、OOF の学習範囲、P2 の集計）。"""

import unittest

import numpy as np
import pandas as pd

from src.pipeline.models import stage_models as sm


def _frame(dates, groups, values):
    df = pd.DataFrame({"date": pd.to_datetime(dates), "g": groups, "v": values})
    return df.reset_index(drop=True)


class AsofGroupMeanTest(unittest.TestCase):
    def test_uses_only_strictly_earlier_days(self):
        df = _frame(["2021-01-01", "2021-01-02", "2021-01-02", "2021-01-03"], ["a", "a", "a", "a"], [1.0, 3.0, 5.0, 0.0])
        out = sm.asof_group_mean(df, "g", "v", k=0.0, name="x")
        self.assertEqual(out["x_n"].tolist(), [0, 1, 1, 3])             # 同日の行は数えない
        self.assertAlmostEqual(out["x_mean"].iloc[1], 1.0)
        self.assertAlmostEqual(out["x_mean"].iloc[3], 3.0)               # (1+3+5)/3

    def test_changing_future_value_does_not_change_past(self):
        a = _frame(["2021-01-01", "2021-02-01", "2021-03-01"], ["a", "a", "a"], [1.0, 2.0, 3.0])
        b = a.copy()
        b.loc[2, "v"] = 999.0
        ra = sm.asof_group_mean(a, "g", "v")["g_v_mean"].to_numpy()
        rb = sm.asof_group_mean(b, "g", "v")["g_v_mean"].to_numpy()
        np.testing.assert_allclose(ra[:3], rb[:3])


class YearSplitsTest(unittest.TestCase):
    def setUp(self):
        d = pd.date_range("2020-01-01", "2025-12-31", freq="7D")
        self.df = pd.DataFrame({"date": d, "year": d.year})

    def test_expanding_excludes_gap_and_future(self):
        (y, tr, va), = sm.year_splits(self.df, [2023], "expanding")
        self.assertEqual(y, 2023)
        self.assertTrue((self.df["date"].iloc[tr] < pd.Timestamp("2022-12-22")).all())
        self.assertTrue((self.df["year"].iloc[va] == 2023).all())

    def test_first_year_has_no_expanding_train(self):
        self.assertEqual(sm.year_splits(self.df, [2020], "expanding"), [])

    def test_loyo_excludes_valid_year_and_margin(self):
        (_, tr, va), = sm.year_splits(self.df, [2022], "loyo")
        dates = self.df["date"].iloc[tr]
        self.assertFalse(((dates >= pd.Timestamp("2021-12-22")) & (dates < pd.Timestamp("2023-01-11"))).any())
        self.assertTrue((self.df["year"].iloc[tr] != 2022).all())


class P2Test(unittest.TestCase):
    def test_front_runner_gap_and_count(self):
        race = np.array(["r1"] * 4 + ["r2"] * 2)
        pred = np.array([0.05, 0.20, 0.60, 0.90, 0.5, 0.4])
        o = sm.p2_aggregates(race, pred)
        self.assertAlmostEqual(o["s_front_gap"].iloc[0], 0.05 - 0.20, places=5)       # 最前の馬は次点との差（負）
        self.assertGreater(o["s_front_gap"].iloc[3], 0)
        self.assertEqual(o["s_nfront_pred"].iloc[0], 1)                                  # 自馬を除く先行馬は 0.20 の 1 頭
        self.assertEqual(o["s_nfront_pred"].iloc[2], 2)
        self.assertGreater(o["s_lone_front_prob"].iloc[0], o["s_lone_front_prob"].iloc[1])
        self.assertEqual(o["s_pos_pred_rank"].iloc[0], 1)

    def test_nan_rows_stay_nan(self):
        o = sm.p2_aggregates(np.array(["r"] * 3), np.array([0.1, np.nan, 0.5]))
        self.assertTrue(np.isnan(o["s_front_gap"].iloc[1]))


class PaceClassTest(unittest.TestCase):
    def test_classes_follow_tp02_thresholds(self):
        lab = pd.DataFrame({"race_id": ["a", "b", "c", "d", "h"], "horse_id": "x", "date": pd.Timestamp("2021-01-01"), "dist_band": [1, 1, 1, 1, 1],
                            "pace_bal": [-3.0, -1.0, 2.0, np.nan, 5.0], "rpci": [46.0, 48.0, 55.0, 48.0, 48.0],
                            "is_hurdle": [False, False, False, False, True]})
        r = sm.pace_class_labels(lab, {"1": [-2.5, 1.0]}, {"1": [47.0, 50.0]}).set_index("race_id")
        self.assertEqual(r["pace_bal_cls"].tolist()[:3], [0, 1, 2])      # 小さい順に 0,1,2（pace_bal 小 = 後傾）
        self.assertEqual(r["rpci_cls"].tolist()[:3], [0, 1, 2])          # rpci 小 = 前傾
        self.assertTrue(np.isnan(r.loc["d", "pace_bal_cls"]))
        self.assertNotIn("h", r.index)                                   # 障害は対象外


class MulticlassTest(unittest.TestCase):
    def test_uniform_is_ln3(self):
        P = np.full((4, 3), 1 / 3)
        self.assertAlmostEqual(sm.multiclass_logloss(P, np.array([0, 1, 2, 0.0])), np.log(3), places=6)


class WithinRaceSpearmanTest(unittest.TestCase):
    def test_perfect_and_reverse(self):
        race = np.array(["a"] * 5 + ["b"] * 5)
        x = np.tile(np.arange(5.0), 2)
        self.assertAlmostEqual(sm.within_race_spearman(race, x, x), 1.0)
        self.assertAlmostEqual(sm.within_race_spearman(race, x, -x), -1.0)


if __name__ == "__main__":
    unittest.main()
