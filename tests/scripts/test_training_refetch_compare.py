"""調教の取り直し前後の比較（training_refetch_compare）。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.scripts.data import training_refetch_compare as T


class CompareTest(unittest.TestCase):
    def _frames(self):
        keys = {"race_id": ["r1"] * 3 + ["r2"] * 3, "horse_id": list("abcdef")}
        before = pd.DataFrame({**keys, "p_win": [0.3, 0.3, 0.4, 0.3, 0.3, 0.4], "tl_x": [np.nan] * 6, "wc_y": [np.nan] * 6})
        after = pd.DataFrame({**keys, "p_win": [0.6, 0.2, 0.2, 0.2, 0.2, 0.6], "tl_x": [1.0] * 6, "wc_y": [1.0] * 6})
        res = pd.DataFrame({"race_id": ["r1"] * 3 + ["r2"] * 3, "horse_id": list("abcdef"), "horse_number": [1, 2, 3] * 2,
                            "finish_position": [1, 2, 3, 3, 2, 1]})
        mk = pd.DataFrame({"race_id": ["r1"] * 3 + ["r2"] * 3, "horse_number": [1, 2, 3] * 2, "p_mkt": [0.4, 0.3, 0.3] * 2})
        return before, after, res, mk

    def test_compare_shapes_and_direction(self):
        out = T.compare(*self._frames())
        self.assertEqual((out["rows"], out["races"]), (6, 2))
        self.assertEqual(out["before"]["training_cols_missing_rate"], 1.0)
        self.assertEqual(out["after"]["training_cols_missing_rate"], 0.0)
        self.assertLess(out["paired"]["delta_ll"], 0.0)               # 勝ち馬に高い確率を付けた after が良い
        self.assertGreater(out["after"]["logit_sd"], out["before"]["logit_sd"])

    def test_paired_delta_zero_when_equal(self):
        race = np.array(["a", "a", "b", "b"])
        y = np.array([1.0, 0.0, 0.0, 1.0])
        p = np.array([0.6, 0.4, 0.3, 0.7])
        out = T.paired_ll_delta(race, y, p, p)
        self.assertAlmostEqual(out["delta_ll"], 0.0)
        self.assertEqual(out["races"], 2)


if __name__ == "__main__":
    unittest.main()
