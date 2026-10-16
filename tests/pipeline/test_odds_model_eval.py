"""オッズ予測の評価指標テスト。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pipeline.betting import odds_model_eval as oe


class MetricTest(unittest.TestCase):
    def test_market_prob_sums_to_one(self):
        race = np.array(["a", "a", "a", "b", "b"])
        p = oe.market_prob(race, np.array([2.0, 4.0, 4.0, 1.5, 3.0]))
        self.assertAlmostEqual(p[:3].sum(), 1.0)
        self.assertAlmostEqual(p[0], 0.5)

    def test_perfect_prediction_zero_error(self):
        race = np.array(["a"] * 3)
        p = oe.market_prob(race, np.array([2.0, 4.0, 8.0]))
        self.assertTrue((oe.per_race_sq_log_err(race, p, p) == 0).all())
        self.assertAlmostEqual(float(oe.kl_by_race(race, p, p).iloc[0]), 0.0, places=9)

    def test_kl_positive_for_wrong(self):
        race = np.array(["a"] * 3)
        p = oe.market_prob(race, np.array([2.0, 4.0, 8.0]))
        self.assertGreater(float(oe.kl_by_race(race, p, p[::-1].copy()).iloc[0]), 0.0)

    def test_spearman_and_band(self):
        race = np.array(["a"] * 4 + ["b"] * 4)
        odds = np.array([1.5, 3.0, 8.0, 30.0] * 2)
        self.assertAlmostEqual(oe.spearman_by_race(race, odds, odds * 1.1), 1.0)
        t = oe.band_table(odds, odds * 2.0)
        self.assertTrue(np.allclose(t["予測÷確定の中央値"], 2.0))

    def test_bootstrap_sign(self):
        a = pd.Series(np.full(200, 0.9), index=[f"r{i}" for i in range(200)])
        b = pd.Series(np.full(200, 1.0), index=a.index)
        r = oe.paired_bootstrap_mean_diff(a, b, n_boot=100)
        self.assertLess(r["hi"], 0)


if __name__ == "__main__":
    unittest.main()
