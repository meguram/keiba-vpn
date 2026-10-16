"""低ランク補完・RankGauss の単体テスト。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pipeline.features import imputation as im


def _lowrank(n=3000, d=20, k=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, k)) @ rng.normal(size=(k, d)) + 0.05 * rng.normal(size=(n, d))


class SoftImputerTest(unittest.TestCase):
    def test_beats_mean_fill_and_keeps_observed(self):
        A = _lowrank()
        rng = np.random.default_rng(1)
        M = rng.random(A.shape) < 0.3
        Xn = np.where(M, np.nan, A)
        imp = im.SoftImputer(rank=3).fit(Xn[:2000])
        B = imp.transform(Xn)
        self.assertFalse(np.isnan(B).any())
        np.testing.assert_allclose(B[~M], A[~M])
        err_soft = np.sqrt(np.mean((B[M] - A[M]) ** 2))
        mean_fill = np.where(M, np.nanmean(Xn, axis=0), A)
        err_mean = np.sqrt(np.mean((mean_fill[M] - A[M]) ** 2))
        self.assertLess(err_soft, 0.5 * err_mean)

    def test_impute_frame_adds_flags(self):
        A = _lowrank(n=500, d=6)
        A[:200, 0] = np.nan
        X = pd.DataFrame(A, columns=list("abcdef"))
        out = im.impute_frame(X, np.ones(500, bool), list("abcdef"), rank=2)
        self.assertFalse(out[list("abcdef")].isna().any().any())
        self.assertIn("miss_a", out.columns)
        self.assertEqual(int(out["miss_a"].sum()), 200)
        self.assertNotIn("miss_b", out.columns)


class RankGaussTest(unittest.TestCase):
    def test_outlier_squashed_and_nan_kept(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=2000)
        x[0] = 1e6
        x[1] = np.nan
        X = pd.DataFrame({"a": x})
        out = im.rank_gauss(X, np.ones(2000, bool), ["a"])
        self.assertLess(abs(float(out["a"].iloc[0])), 6.0)
        self.assertTrue(np.isnan(out["a"].iloc[1]))


if __name__ == "__main__":
    unittest.main()
