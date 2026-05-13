"""research.race_quality_model のユニットテスト（ストレージなしで検証可能な部分）。"""

from __future__ import annotations

import unittest

import numpy as np

from research.race_quality_model import (
    compute_pace_shape,
    distance_surface_segment,
    extract_lap_times_from_blob,
    get_race_quality_meta,
    going_archetype_multiplier,
    history_distance_band,
    segment_archetype_prior,
    _column_minmax_nonneg,
    _fit_mixture,
    _hand_segment_archetype_prior,
    _nine_probs,
    _parse_lap_string,
)


class TestRaceQualityModel(unittest.TestCase):
    def test_distance_surface_segment(self):
        self.assertEqual(distance_surface_segment("芝", 1200), "芝_1200-1399")
        self.assertEqual(distance_surface_segment("ダート", 1800), "ダート_1800-1999")
        self.assertIn("障害", distance_surface_segment("障", 3200))

    def test_history_distance_band(self):
        self.assertEqual(history_distance_band(1200), "短距離")
        self.assertEqual(history_distance_band(1600), "マイル")
        self.assertEqual(history_distance_band(2000), "中距離")
        self.assertEqual(history_distance_band(2400), "中長距離")
        self.assertEqual(history_distance_band(3200), "長距離")

    def test_segment_prior_positive(self):
        w = segment_archetype_prior("芝_1200-1399")
        self.assertEqual(w.shape, (8,))
        self.assertTrue(np.all(w > 0))

    def test_hand_prior_matches_segment_without_json(self):
        a = _hand_segment_archetype_prior("ダート_1800-1999")
        b = segment_archetype_prior("ダート_1800-1999")
        np.testing.assert_array_almost_equal(a, b)

    def test_parse_lap_string(self):
        self.assertGreaterEqual(len(_parse_lap_string("12.3-11.4-10.5")), 3)

    def test_extract_lap_blob(self):
        blob = {"entries": [{"lap_times": "11.1-10.2-9.8"}]}
        v = extract_lap_times_from_blob(blob)
        self.assertGreaterEqual(len(v), 3)

    def test_going_multiplier_mud(self):
        m = going_archetype_multiplier("重")
        self.assertGreater(m[7], 1.0)
        m2 = going_archetype_multiplier("良")
        self.assertLess(m2[7], m[7])

    def test_pace_shape_grind(self):
        p = compute_pace_shape({"first_half_3f": 33.0, "second_half_3f": 36.0}, [], 1600)
        self.assertEqual(p["has_half_pace"], 1.0)
        self.assertGreater(p["grind_index"], 0.5)
        self.assertLess(p["burst_index"], 0.5)

    def test_pace_shape_even_laps(self):
        laps = [11.0, 11.1, 11.0, 11.05, 11.0]
        p = compute_pace_shape({}, laps, 2000)
        self.assertGreater(p["lap_evenness"], 0.5)

    def test_nnls_pipeline(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(0.1, 1.0, size=(10, 8))
        Xn = _column_minmax_nonneg(X)
        y = rng.uniform(0.2, 1.0, size=10)
        coef, r2, _ = _fit_mixture(Xn, y)
        self.assertEqual(coef.shape, (8,))
        probs, _ = _nine_probs(coef, r2, 10)
        self.assertEqual(len(probs), 9)
        self.assertLess(abs(sum(probs) - 1.0), 1e-5)

    def test_get_meta(self):
        m = get_race_quality_meta()
        self.assertEqual(m["version"], 1)
        self.assertEqual(len(m["axes"]), 9)
        self.assertIn("api", m)


if __name__ == "__main__":
    unittest.main()
