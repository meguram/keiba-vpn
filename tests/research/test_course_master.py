"""コース形状データの突き合わせの単体テスト。"""

from __future__ import annotations

import unittest

from src.research.race import course_master as cm


class CourseMasterTest(unittest.TestCase):
    def test_to_num(self):
        self.assertEqual(cm.to_num("約 525m"), 525.0)
        self.assertEqual(cm.to_num("1,624.1"), 1624.1)
        self.assertIsNone(cm.to_num(None)); self.assertIsNone(cm.to_num("なし")); self.assertIsNone(cm.to_num(True))
        self.assertEqual(cm.to_num(3), 3.0)

    def test_keys(self):
        self.assertEqual(cm.row_key({"venue_code": "5", "surface": "ダ", "distance": "1600m", "variant": "外回り"}), ("05", "ダート", 1600, "外"))
        self.assertEqual(cm.canon_variant("内回り"), "内"); self.assertEqual(cm.canon_variant(None), ""); self.assertEqual(cm.canon_variant("直線"), "直")

    def test_compare_statuses(self):
        a = [{"venue_code": "05", "surface": "芝", "distance": 2000, "final_straight_m": 525, "n_corners": 4, "lap_length_m": 2083}]
        b = [{"venue_code": "05", "surface": "芝", "distance": 2000, "final_straight_m": 525.9, "n_corners": 4, "lap_length_m": 2083}]
        c = [{"venue_code": "05", "surface": "芝", "distance": 2000, "final_straight_m": 480, "n_corners": 3}]
        d = cm.compare({"A": a, "B": b, "C": c})
        g = d.set_index("field")
        self.assertEqual(g.loc["final_straight_m", "status"], "合意(外れ値あり)")
        self.assertEqual(g.loc["final_straight_m", "consensus"], 525.45)
        self.assertEqual(g.loc["n_corners", "status"], "合意(外れ値あり)")
        self.assertEqual(g.loc["lap_length_m", "status"], "合意")
        self.assertEqual(g.loc["elevation_range_m", "status"], "欠損")
        d2 = cm.compare({"A": a, "C": c})
        self.assertEqual(d2.set_index("field").loc["final_straight_m", "status"], "不一致")
        d3 = cm.compare({"A": a})
        self.assertEqual(d3.set_index("field").loc["final_straight_m", "status"], "未確認")

    def test_goal_slope_reading(self):
        self.assertTrue(cm.has_goal_slope("ゴール前に高低差 2.2m の急坂"))
        self.assertFalse(cm.has_goal_slope("なし（平坦）"))
        self.assertIsNone(cm.has_goal_slope(None))
        self.assertTrue(cm.has_goal_slope({"有無": "あり", "位置": "ゴール前 約 340m〜", "勾配": "2.2%"}))


if __name__ == "__main__":
    unittest.main()
