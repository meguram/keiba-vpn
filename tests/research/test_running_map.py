"""出走馬の走法マップ（2 次元化・グループ分け・入力の整形）のテスト。データファイルは使わない。"""

from __future__ import annotations

import unittest

import numpy as np

from src.research.race import running_map as rm


def _runners(seed: int = 0, n: int = 14) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, len(rm.AXES)))
    Z[:, 1] = Z[:, 0] + rng.normal(scale=0.05, size=n)  # TS と GC はほぼ同じ
    return Z * rm.WEIGHTS


class ProjectTest(unittest.TestCase):
    def test_orientation_tskick_up_front_left(self):
        for seed in range(5):
            pr = rm.project(_runners(seed))
            up = pr["loadings"][0] + pr["loadings"][1]
            self.assertAlmostEqual(float(up[0]), 0.0, places=9)    # 末脚の矢印は真上
            self.assertGreater(float(up[1]), 0.0)
            self.assertLessEqual(float(pr["loadings"][3][0]), 0.0)  # 先行力の矢印は左

    def test_rotation_keeps_distances_of_pca(self):
        from scipy.spatial.distance import pdist

        Z = _runners(1)
        Zc = Z - Z.mean(0)
        _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
        np.testing.assert_allclose(pdist(rm.project(Z)["coords"]), pdist(Zc @ Vt[:2].T), atol=1e-9)

    def test_r2_and_fidelity(self):
        pr = rm.project(_runners(2))
        self.assertTrue(0 < pr["explained"] <= 1)
        self.assertTrue(-1 <= pr["distance_spearman"] <= 1)
        self.assertTrue(all(r is None or -1e-9 <= r <= 1 + 1e-9 for r in pr["r2"]))

    def test_constant_axis_has_no_r2(self):
        Z = _runners(3)
        Z[:, 5] = 0.0
        pr = rm.project(Z)
        self.assertIsNone(pr["r2"][5])
        np.testing.assert_allclose(pr["loadings"][5], 0.0, atol=1e-9)


class GroupTest(unittest.TestCase):
    def test_separated_clusters_are_found(self):
        rng = np.random.default_rng(0)
        a = rng.normal(loc=0, scale=0.1, size=(6, 6))
        b = rng.normal(loc=3, scale=0.1, size=(5, 6))
        lab = rm.group(np.vstack([a, b]))
        self.assertEqual(len(set(lab)), 2)
        self.assertEqual(len(set(lab[:6])), 1)
        self.assertEqual(len(set(lab[6:])), 1)

    def test_too_few_or_identical_is_one_group(self):
        self.assertEqual(set(rm.group(np.zeros((3, 6)))), {0})
        self.assertEqual(set(rm.group(np.ones((8, 6)))), {0})

    def test_describe_group_merges_ts_and_gc(self):
        Zall = np.zeros((10, 6))
        Zall[:3, 0] = Zall[:3, 1] = 2.0  # 3 頭だけ末脚が速い
        traits = rm.describe_group(Zall[:3], Zall)
        self.assertEqual(traits[0]["axis"], "末脚")
        self.assertEqual(traits[0]["text"], "末脚が速い")
        self.assertEqual(len(traits), 1)


class TransformTest(unittest.TestCase):
    def test_percentile_ties_take_middle(self):
        q = np.quantile(np.r_[np.zeros(80), np.linspace(1, 2, 20)], np.linspace(0, 1, 1001))
        p = rm._to_pct(np.array([0.0, 2.0, np.nan]), q)
        self.assertAlmostEqual(float(p[0]), 0.4, delta=0.01)  # 0 が 80% を占めるので中央の 40%
        self.assertAlmostEqual(float(p[1]), 1.0)
        self.assertTrue(np.isnan(p[2]))

    def test_condition_and_surface_normalization(self):
        self.assertEqual([rm._norm_condition(s) for s in ("稍", "稍重", "不", "重", "良", "", None)],
                         ["稍重", "稍重", "不良", "重", "良", None, None])
        self.assertEqual([rm._norm_surface(s) for s in ("芝", "ダ", "ダート", "障", "")], ["芝", "ダート", "ダート", None, None])

    def test_race_conditions_falls_back_to_race_id_venue(self):
        c = rm.race_conditions({"date": "2026/05/04", "distance": "1600", "surface": "芝"}, "202605040301")
        self.assertEqual(c, {"date": "2026-05-04", "venue": "東京", "surface": "芝", "distance": 1600, "track_condition": None})
        self.assertIsNone(rm.race_conditions({"distance": 0}, "202605040301")["distance"])


if __name__ == "__main__":
    unittest.main()
