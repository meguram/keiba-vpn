"""市場分析（尤度比検定・指数の最尤推定・較正）の単体テスト。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pipeline.betting import market_analysis as ma
from src.pipeline.inference.exotic_probabilities import RankDistribution


class MarketAnalysisTest(unittest.TestCase):
    def test_finishing_orders(self):
        pay = pd.DataFrame({"race_id": ["a", "a", "b", "b"], "bet_type": ["sanrentan", "umatan", "sanrentan", "sanrentan"], "combo": ["3>1>2", "3>1", "1>2>3", "2>1>3"],
                            "ambiguous": [False, False, False, False]})
        o = ma.finishing_orders(pay)
        self.assertEqual(o, {"a": (3, 1, 2)})                 # b は 2 行（一意でない）ので除く

    def test_global_alpha_recovers_truth(self):
        rng = np.random.default_rng(0)
        true_a2, true_a3 = 0.7, 0.8
        groups = []
        for _ in range(500):
            w = rng.dirichlet(np.ones(10) * 1.5)
            d = RankDistribution(w, true_a2, true_a3)
            p3 = d._p3.ravel()
            k = rng.choice(len(p3), p=p3 / p3.sum())
            i, j, kk = np.unravel_index(k, d._p3.shape)
            groups.append((w, int(i), int(j), int(kk)))
        a2, a3, nll = ma.fit_global_alphas(groups)
        self.assertLess(abs(a2 - true_a2), 0.12)
        self.assertLess(abs(a3 - true_a3), 0.2)
        self.assertLess(nll, ma.order_nll(groups, 1.0, 1.0))

    def test_lr_test_detects_model_information(self):
        rng = np.random.default_rng(1)
        informative, null = [], []
        for _ in range(600):
            true = rng.dirichlet(np.ones(8))
            w = rng.choice(8, p=true)
            lk = np.log(true) + rng.normal(0, 0.5, 8)                  # 市場は雑音
            lm_good = np.log(true) + rng.normal(0, 0.5, 8)             # モデルは独立な雑音で、同じ真値の情報を持つ
            lm_bad = rng.normal(0, 1.0, 8)                             # 情報なし
            informative.append((lm_good - np.log(np.exp(lm_good).sum()), lk - np.log(np.exp(lk).sum()), int(w)))
            null.append((lm_bad - np.log(np.exp(lm_bad).sum()), lk - np.log(np.exp(lk).sum()), int(w)))
        good = ma.lr_test_model_weight(informative, n_boot=20)
        bad = ma.lr_test_model_weight(null, n_boot=20)
        self.assertLess(good["p_value"], 0.01)
        self.assertGreater(bad["p_value"], 0.01)

    def test_ev_slope_calibrated_vs_noise(self):
        rng = np.random.default_rng(2)
        n = 4000
        x = rng.uniform(0.5, 1.5, n)
        df = pd.DataFrame({"race_id": rng.integers(0, 800, n).astype(str), "ev_hat": x})
        df["mult"] = (rng.random(n) < x / 10).astype(float) * 10.0          # 較正済み: E[mult] = x
        cal = ma.ev_slope(df, n_boot=50)
        df["mult"] = (rng.random(n) < 0.1).astype(float) * 10.0            # x と無関係
        noise = ma.ev_slope(df, n_boot=50)
        self.assertGreater(cal["slope"], 0.4)
        self.assertLess(abs(noise["slope"]), 0.7)
        self.assertGreater(cal["slope"], noise["slope"])

    def test_isotonic_ev_monotone(self):
        rng = np.random.default_rng(3)
        n = 3000
        x = rng.uniform(0.2, 2.0, n)
        df = pd.DataFrame({"type": ["tansho"] * n, "mult": (rng.random(n) < np.clip(x, 0, 2) / 10) * 10.0})
        iso = ma.IsotonicEV().fit(df, x)
        grid = np.linspace(0.3, 1.9, 20)
        p = iso.predict(pd.DataFrame({"type": ["tansho"] * 20}), grid)
        self.assertTrue((np.diff(p) >= -1e-9).all())


if __name__ == "__main__":
    unittest.main()
