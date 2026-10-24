"""特徴量グループの割り当てテスト。"""

from __future__ import annotations

import unittest

from src.pipeline.models import feature_groups as fg


class AssignTest(unittest.TestCase):
    def test_assign_and_relative(self):
        fam = {"F3": ["f3_r"], "A": ["a_style"], "B": ["b_aff_all"], "D": ["d_rest"], "C": ["c_res_ewm2"]}
        cols = ["f3_r", "f3_r_rz", "a_style", "a_style_rrank", "b_aff_all", "d_rest", "c_res_ewm2_rgap", "distance", "jockey_prior_win_rate", "s_time_pred", "oikiri_eval", "ped_x"]
        g = fg.assign_groups(cols, fam, ["s_time_pred"], ["oikiri_eval"], ["ped_x"])
        self.assertEqual(g["RATING"], ["f3_r", "f3_r_rz", "c_res_ewm2_rgap", "s_time_pred"])
        self.assertEqual(g["STYLE"], ["a_style", "a_style_rrank"])
        self.assertEqual(g["APTITUDE"], ["b_aff_all", "ped_x"])
        self.assertEqual(g["CONDITION"], ["d_rest", "oikiri_eval"])
        self.assertEqual(g["CONNECTIONS"], ["jockey_prior_win_rate"])
        flat = [c for v in g.values() for c in v]
        self.assertEqual(len(flat), len(set(flat)))
        self.assertNotIn("distance", flat)              # CONTEXT は各モデルへ別に渡す

    def test_unknown_column_raises(self):
        with self.assertRaises(ValueError):
            fg.assign_groups(["mystery"], {}, [], [], [])


if __name__ == "__main__":
    unittest.main()
