"""3 者の突き合わせからコースマスタを作る処理のテスト。"""

from __future__ import annotations

import unittest

from src.research.race import build_course_master as bcm


def _row(**kw):
    base = {"venue_code": "05", "venue": "東京", "surface": "芝", "distance": 2000, "direction": "左", "variant": "", "lap_length_m": 2083, "final_straight_m": 525.9, "elevation_range_m": 2.7,
            "n_corners": 4, "dist_start_to_first_corner_m": 130, "goal_slope": {"exists": True, "start_before_goal_m": 460, "end_before_goal_m": 300, "rise_m": 2.0}}
    base.update(kw)
    return base


class BuildTest(unittest.TestCase):
    def test_agreement_and_outlier(self):
        a = [_row()]
        b = [_row(final_straight_m=526, elevation_range_m=2.6, goal_slope={"present": True, "start_before_goal_m": 480, "end_before_goal_m": 260, "height_m": 2.0})]
        c = [_row(final_straight_m=480, goal_slope="あり: 高低差約2m", dist_start_to_first_corner_m=126)]
        rows, issues = bcm.build(a, b, c)
        r = rows[0]
        self.assertEqual(r["final_straight_m_status"], "合意(外れ値あり)")
        self.assertAlmostEqual(r["final_straight_m"], 525.95, places=2)
        self.assertEqual(r["lap_length_m_status"], "合意")
        self.assertEqual(r["first_curve_status"], "合意")
        self.assertTrue(r["goal_slope_exists"]); self.assertTrue(r["goal_slope_unanimous"])
        self.assertEqual(r["goal_slope_start_status"], "合意")          # 460 と 480 は許容 45m 以内
        self.assertEqual(r["goal_slope_rise_status"], "合意(外れ値あり)" if False else "合意")

    def test_disagreement_goes_to_issues(self):
        a = [_row(goal_slope={"exists": False})]
        b = [_row(goal_slope={"present": True, "start_before_goal_m": 400, "end_before_goal_m": 200, "height_m": 1.0}, final_straight_m=400)]
        c = [_row(goal_slope="あり: 坂", final_straight_m=300)]
        rows, issues = bcm.build(a, b, c)
        fields = {i["field"] for i in issues}
        self.assertIn("final_straight_m", fields)
        self.assertFalse(rows[0]["goal_slope_unanimous"])
        self.assertTrue(rows[0]["goal_slope_exists"])                      # 3 者中 2 者が「あり」

    def test_first_corner_definition_flag(self):
        a = [_row(dist_start_to_first_corner_m=760)]
        b = [_row(dist_start_to_first_corner_m=160)]
        c = [_row(dist_start_to_first_corner_m=156.6)]
        rows, _ = bcm.build(a, b, c)
        self.assertTrue(rows[0]["first_corner_definition_differs"])
        self.assertEqual(rows[0]["first_curve_status"], "合意")


class CorrectionTest(unittest.TestCase):
    def test_apply_corrections(self):
        rows = [{"venue_code": "09", "surface": "芝", "distance": 3200, "variant": "内", "lap_length_m": 2089.0, "final_straight_m": 473.6},
                {"venue_code": "09", "surface": "芝", "distance": 3000, "variant": "内", "lap_length_m": 1689.0}]
        corr = [{"match": {"venue_code": "09", "surface": "芝", "distance": 3200}, "set": {"lap_length_m": 1689.0, "final_straight_m": 356.5}, "evidence": "e", "round": 2}]
        out = bcm.apply_corrections(rows, corr)
        self.assertEqual(out[0]["final_straight_m"], 356.5)
        self.assertEqual(out[0]["corrections"][0]["round"], 2)
        self.assertEqual(out[1]["lap_length_m"], 1689.0)
        self.assertNotIn("corrections", out[1])

    def test_variant_match(self):
        rows = [{"venue_code": "06", "surface": "芝", "distance": 1600, "variant": "外"}, {"venue_code": "06", "surface": "芝", "distance": 1600, "variant": "内"}]
        out = bcm.apply_corrections(rows, [{"match": {"venue_code": "06", "surface": "芝", "distance": 1600, "variant": "外"}, "set": {"n_corner_records": 3}, "evidence": "e"}])
        self.assertEqual(out[0].get("n_corner_records"), 3)
        self.assertNotIn("n_corner_records", out[1])


if __name__ == "__main__":
    unittest.main()
