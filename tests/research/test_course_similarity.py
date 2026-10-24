"""コース類似度と、馬の『似たコースの実績』特徴量のテスト。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.research.race import course_similarity as cs


def _master():
    rows = []
    for vc, surf, dist, st, el, slope, fc, nc in [("05", "芝", 1600, 525.9, 2.7, 2.0, 540, 2), ("09", "芝", 1600, 473.6, 2.4, 1.8, 444, 2), ("06", "芝", 1600, 310.0, 5.3, 2.2, 240, 3),
                                                  ("05", "ダート", 1600, 501.6, 2.4, 2.0, 340, 2), ("09", "ダート", 1400, 352.7, 1.6, 1.5, 330, 2)]:
        rows.append({"venue_code": vc, "surface": surf, "distance": dist, "variant": "", "final_straight_m": st, "lap_length_m": 2000.0, "elevation_range_m": el, "goal_slope_exists": True, "goal_slope_rise": slope,
                     "first_curve_m": fc, "n_corner_records": nc})
    return pd.DataFrame(rows)


class SimilarityTest(unittest.TestCase):
    def test_vectors_and_kernel(self):
        v = cs.course_vectors(_master())
        self.assertEqual(len(v), 5)
        K = cs.similarity_kernel(v, tau=1.5)
        np.testing.assert_allclose(np.diag(K), 1.0)
        np.testing.assert_allclose(K, K.T)
        i_turf, i_dirt = list(v.index).index(("05", "芝", 1600)), list(v.index).index(("05", "ダート", 1600))
        self.assertEqual(K[i_turf, i_dirt], 0.0)                                      # 面が違えば 0
        i_h = list(v.index).index(("09", "芝", 1600)); i_n = list(v.index).index(("06", "芝", 1600))
        self.assertGreater(K[i_turf, i_h], K[i_turf, i_n])                            # 東京と阪神外の方が、東京と中山より似ている

    def test_variant_weighting(self):
        m = pd.DataFrame([{"venue_code": "04", "surface": "芝", "distance": 2000, "variant": "内", "final_straight_m": 358.7, "lap_length_m": 1623.0, "elevation_range_m": 0.8, "goal_slope_exists": False, "goal_slope_rise": None, "first_curve_m": 300.0, "n_corner_records": 4},
                          {"venue_code": "04", "surface": "芝", "distance": 2000, "variant": "外", "final_straight_m": 658.7, "lap_length_m": 2223.0, "elevation_range_m": 1.6, "goal_slope_exists": False, "goal_slope_rise": None, "first_curve_m": 900.0, "n_corner_records": 2}])
        v = cs.course_vectors(m)
        self.assertAlmostEqual(float(v["final_straight_m"].iloc[0]), 0.48 * 358.7 + 0.52 * 658.7, places=3)

    def _races(self):
        # 馬 h1 は東京芝1600 で 3 回好走した後、阪神芝1600（似ている）と中山芝1600（似ていない）に出る
        rows = []
        def add(rid, date, hid, v, s, d, fin, fs=10):
            rows.append({"race_id": rid, "date": date, "horse_id": hid, "venue_code": v, "surface": s, "distance": d, "finish_position": fin, "field_size": fs})
        for k, dt in enumerate(["2024-01-06", "2024-01-20", "2024-02-03"]):
            add(f"t{k}", dt, "h1", "05", "芝", 1600, 1)
        add("hh", "2024-03-02", "h1", "09", "芝", 1600, 5)
        add("nk", "2024-03-02", "h1", "06", "芝", 1600, 5)
        add("same_day_b", "2024-03-02", "h1", "09", "芝", 1600, 10)
        return pd.DataFrame(rows)

    def test_features_use_only_prior_races_and_similarity(self):
        v = cs.course_vectors(_master()); K = cs.similarity_kernel(v, tau=1.5)
        df = self._races()
        f = cs.horse_similarity_features(df, v, K)
        first = f.iloc[0]
        self.assertAlmostEqual(first["sim_n"], 0.0)                                  # 初出走は実績なし
        h = f[df["race_id"] == "hh"].iloc[0]; n = f[df["race_id"] == "nk"].iloc[0]
        self.assertGreater(h["sim_aff"], n["sim_aff"])                               # 似たコース（阪神）の方が、東京の好走を多く借りる
        self.assertGreater(h["sim_n"], n["sim_n"])
        self.assertTrue(np.isfinite(h["shape_pred"]) and h["shape_se"] > 0)
        # 同じ日のレース（same_day_b の着順 10）は、同日の hh の特徴量に影響しない
        df2 = df.copy(); df2.loc[df2["race_id"] == "same_day_b", "finish_position"] = 1
        f2 = cs.horse_similarity_features(df2, v, K)
        pd.testing.assert_series_equal(f.loc[df["race_id"] == "hh", "sim_aff"], f2.loc[df2["race_id"] == "hh", "sim_aff"])

    def test_unknown_course_gives_nan(self):
        v = cs.course_vectors(_master()); K = cs.similarity_kernel(v)
        df = self._races().iloc[:2].copy(); df.loc[df.index[1], "distance"] = 1234
        f = cs.horse_similarity_features(df, v, K)
        self.assertTrue(np.isnan(f.iloc[1]["sim_aff"]))


if __name__ == "__main__":
    unittest.main()
