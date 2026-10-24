"""shrinkage / fit_coeff_pace / fit_par_time_class のユニットテスト。"""

import unittest

import numpy as np
import pandas as pd

from src.pipeline.megu_index.fit_coeff_pace import fit_coeff_pace
from src.pipeline.megu_index.fit_par_time_class import fit_par_time_class, fit_pool_betas
from src.pipeline.megu_index.shrinkage import shrink_scalar


class TestShrinkage(unittest.TestCase):
    def test_small_n_pulls_toward_prior(self):
        shrunk, w = shrink_scalar(1.0, 0.5, n=10, strength=30)
        self.assertLess(w, 0.5)
        self.assertGreater(shrunk, 0.5)
        self.assertLess(shrunk, 1.0)

    def test_large_n_keeps_estimate(self):
        shrunk, w = shrink_scalar(1.0, 0.5, n=3000, strength=30)
        self.assertGreater(w, 0.99)
        self.assertAlmostEqual(shrunk, 1.0, places=2)


class TestFitCoeffPace(unittest.TestCase):
    def _make_train(self, n_races: int = 40, slope: float = 0.5) -> pd.DataFrame:
        rows = []
        for i in range(n_races):
            rid = f"r{i}"
            dev = (i % 5) * 0.2
            base = 96.0 + dev * slope
            for pos, adj in [(1, base - 0.5), (2, base)]:
                rows.append(
                    {
                        "race_id": rid,
                        "venue": "東京",
                        "surface": "芝",
                        "distance": 1600,
                        "distance_band": "mile",
                        "front_split_sec": 48.0 + dev,
                        "front_split_dev": dev,
                        "adjusted_time_sec": adj,
                        "year": 2023,
                    }
                )
        return pd.DataFrame(rows)

    def test_shrinkage_uses_cell_information(self):
        df = self._make_train(n_races=40)
        out = fit_coeff_pace(df, min_samples=10)
        self.assertEqual(len(out), 1)
        self.assertGreater(out.iloc[0]["coeff_pace"], 0)
        self.assertIn(out.iloc[0]["source"], {"cell_shrink", "pool_shrink"})

    def test_tiny_cell_uses_prior(self):
        df = self._make_train(n_races=5)
        out = fit_coeff_pace(df, min_samples=10, expand_coverage=False)
        self.assertIn(out.iloc[0]["source"], {"pool_surface", "theory", "pool_distband"})

    def test_expand_coverage_adds_missing_distance(self):
        df = self._make_train(n_races=40)
        extra = pd.DataFrame(
            [{
                "race_id": "x1",
                "venue": "小倉",
                "surface": "芝",
                "distance": 2860,
                "distance_band": "long",
                "front_split_sec": np.nan,
                "front_split_dev": 0.0,
                "adjusted_time_sec": 200.0,
                "year": 2023,
            }]
        )
        full = pd.concat([df, extra], ignore_index=True)
        out = fit_coeff_pace(full, min_samples=10, expand_coverage=True)
        row = out[(out["venue"] == "小倉") & (out["distance"] == 2860)]
        self.assertEqual(len(row), 1)
        self.assertTrue(str(row.iloc[0]["source"]).endswith("_imputed"))


class TestFitParTimeClass(unittest.TestCase):
    def test_positive_beta_cell_falls_back(self):
        rows = []
        for i in range(5):
            rows.append(
                {
                    "venue": "小倉",
                    "surface": "芝",
                    "distance": 2600,
                    "distance_band": "long",
                    "class_rank": 2,
                    "adjusted_time_sec": 140.0 + i,
                    "finish_pos": 2,
                    "year": 2023,
                }
            )
        for i in range(35):
            rows.append(
                {
                    "venue": "東京",
                    "surface": "芝",
                    "distance": 2000,
                    "distance_band": "mile",
                    "class_rank": 2 + (i % 3),
                    "adjusted_time_sec": 120.0 - 0.5 * (2 + (i % 3)),
                    "finish_pos": 2,
                    "year": 2023,
                }
            )
        df = pd.DataFrame(rows)
        global_beta, pool = fit_pool_betas(df)
        out = fit_par_time_class(df, global_beta, pool, min_cell_n=10)
        bad = out[(out["distance"] == 2600) & (out["class_rank"] == 1)]
        self.assertLess(bad.iloc[0]["beta"], 0)

    def test_all_betas_non_positive(self):
        rows = []
        for i in range(40):
            cr = 2 + (i % 4)
            rows.append(
                {
                    "venue": "東京",
                    "surface": "芝",
                    "distance": 1600,
                    "distance_band": "mile",
                    "class_rank": cr,
                    "adjusted_time_sec": 98.0 - 0.4 * cr,
                    "finish_pos": 2,
                    "year": 2023,
                }
            )
        df = pd.DataFrame(rows)
        global_beta, pool = fit_pool_betas(df)
        out = fit_par_time_class(df, global_beta, pool, min_cell_n=10)
        cell_betas = out.drop_duplicates(["venue", "surface", "distance"])["beta"]
        self.assertTrue((cell_betas <= 0).all())

    def test_small_sample_uses_shrink_not_hard_fallback(self):
        rows = []
        for i in range(15):
            cr = 2 + (i % 3)
            rows.append(
                {
                    "venue": "福島",
                    "surface": "芝",
                    "distance": 2000,
                    "distance_band": "mile",
                    "class_rank": cr,
                    "adjusted_time_sec": 121.0 - 0.3 * cr,
                    "finish_pos": 2,
                    "year": 2023,
                }
            )
        df = pd.DataFrame(rows)
        global_beta, pool = fit_pool_betas(df)
        out = fit_par_time_class(df, global_beta, pool, min_cell_n=10)
        cell = out.drop_duplicates(["venue", "surface", "distance"]).iloc[0]
        self.assertIn(cell["source"], {"cell_shrink", "pool_shrink"})
        self.assertLess(cell["beta"], 0)

    def test_n_below_shrink_threshold_uses_pool_only(self):
        rows = []
        for i in range(8):
            rows.append(
                {
                    "venue": "東京",
                    "surface": "芝",
                    "distance": 2300,
                    "distance_band": "middle",
                    "class_rank": 2 + (i % 2),
                    "adjusted_time_sec": 139.0 - 0.1 * i,
                    "finish_pos": 2,
                    "year": 2023,
                }
            )
        for i in range(20):
            rows.append(
                {
                    "venue": "東京",
                    "surface": "芝",
                    "distance": 2000,
                    "distance_band": "mile",
                    "class_rank": 2 + (i % 3),
                    "adjusted_time_sec": 120.0 - 0.5 * (2 + (i % 3)),
                    "finish_pos": 2,
                    "year": 2023,
                }
            )
        df = pd.DataFrame(rows)
        global_beta, pool = fit_pool_betas(df)
        out = fit_par_time_class(df, global_beta, pool, min_cell_n=10)
        cell = out[(out["distance"] == 2300)].drop_duplicates(["venue", "surface", "distance"]).iloc[0]
        self.assertIn(cell["source"], {"pool_distband", "pool_global"})
        self.assertLess(cell["beta"], 0)
        self.assertGreaterEqual(-5 * cell["beta"], 1.5)

    def test_rank2_rank7_gap_has_floor(self):
        rows = []
        for i in range(40):
            cr = 2 + (i % 4)
            rows.append(
                {
                    "venue": "東京",
                    "surface": "芝",
                    "distance": 1600,
                    "distance_band": "mile",
                    "class_rank": cr,
                    "adjusted_time_sec": 98.0 - 0.4 * cr,
                    "finish_pos": 2,
                    "year": 2023,
                }
            )
        df = pd.DataFrame(rows)
        global_beta, pool = fit_pool_betas(df)
        out = fit_par_time_class(df, global_beta, pool, min_cell_n=10)
        cells = out.drop_duplicates(["venue", "surface", "distance"])
        gaps = -5 * cells["beta"]
        self.assertGreaterEqual(gaps.min(), 1.5)


if __name__ == "__main__":
    unittest.main()
