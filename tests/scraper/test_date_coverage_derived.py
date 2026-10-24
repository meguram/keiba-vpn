"""date_coverage 派生カテゴリ N/A テスト。"""

from __future__ import annotations

import unittest

from src.scraper.date_coverage import apply_derived_category_na


class TestDerivedCategoryNa(unittest.TestCase):
    def test_derived_na_when_parent_ok(self):
        row = {
            "race_shutuba": True,
            "race_shutuba_meta": False,
            "race_result_on_time": True,
            "race_result_on_time_lap": False,
        }
        out = apply_derived_category_na(row)
        self.assertIsNone(out["race_shutuba_meta"])
        self.assertIsNone(out["race_result_on_time_lap"])

    def test_derived_stays_false_when_parent_missing(self):
        row = {"race_shutuba": False, "race_shutuba_meta": False}
        out = apply_derived_category_na(row)
        self.assertFalse(out["race_shutuba_meta"])


if __name__ == "__main__":
    unittest.main()
