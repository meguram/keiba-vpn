"""多層スタッキング部品のテスト。"""

from __future__ import annotations

import unittest

import numpy as np

from src.pipeline.models import stacking as st


class StackingTest(unittest.TestCase):
    def _data(self, races=300, n=8, seed=0):
        rng = np.random.default_rng(seed)
        race, years, Zs, fin = [], [], [], []
        for r in range(races):
            s = rng.normal(size=n)
            order = np.argsort(-(s + rng.normal(0, 0.7, n)))
            f = np.empty(n, int); f[order] = np.arange(1, n + 1)
            for i in range(n):
                race.append(f"r{r}"); years.append(2022 + r % 3)
                Zs.append([s[i] + rng.normal(0, 0.3), s[i] + rng.normal(0, 0.8)])
                fin.append(f[i])
        Z, fin = np.array(Zs), np.array(fin)
        Y = {"win": (fin == 1).astype(float), "top2": (fin <= 2).astype(float), "top3": (fin <= 3).astype(float)}
        return np.array(race), np.array(years), Z, Y

    def test_meta_features_shape(self):
        race, years, Z, Y = self._data(50)
        F = st.meta_features(Z, race, ["a", "b"])
        self.assertEqual(F.shape, (len(race), 2 + 6))
        self.assertTrue(np.isfinite(F.to_numpy()).all())
        self.assertEqual(st.meta_features(Z, race, ["a", "b"], with_context=False).shape[1], 2)

    def test_lgb_meta_totals_and_quality(self):
        race, years, Z, Y = self._data()
        F = st.meta_features(Z, race, ["a", "b"])
        P = st.loyo_lgb_meta({t: F for t in st.TOT}, Y, years, race, threads=2, n_rounds=60, params={"min_data_in_leaf": 20})
        for t, tot in st.TOT.items():
            np.testing.assert_allclose(__import__("pandas").Series(P[t]).groupby(race).sum().to_numpy(), tot, atol=1e-5)
        self.assertLess(__import__("src.pipeline.models.win_models", fromlist=["x"]).binary_logloss(P["win"], Y["win"]), 0.40)

    def test_nn_meta_runs(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch がありません")
        race, years, Z, Y = self._data(120)
        F = st.meta_features(Z, race, ["a", "b"])
        P = st.loyo_nn_meta(F, Y, years, race, hidden=(8,), epochs=3, seeds=(0,))
        self.assertEqual(len(P["win"]), len(race))
        self.assertTrue(np.isfinite(P["win"]).all())

    def test_logit_mean(self):
        race = np.repeat(np.arange(4), 6).astype(str)
        p = np.tile(np.array([0.4, 0.2, 0.1, 0.1, 0.1, 0.1]), 4)
        out = st.logit_mean([p, p], race, 1.0)
        np.testing.assert_allclose(__import__("pandas").Series(out).groupby(race).sum().to_numpy(), 1.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
