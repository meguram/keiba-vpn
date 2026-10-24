"""weekly-update 対象日（直近10日）のユニットテスト。"""

from __future__ import annotations

import unittest
from datetime import date

from src.scraper import auto_scrape as asc


class TestRecentOpeningDates(unittest.TestCase):
    def _cal(self) -> dict:
        return {
            "race_days": [
                {"date": "2026-07-08"},
                {"date": "2026-07-09"},
                {"date": "2026-07-11"},
                {"date": "2026-07-12"},
                {"date": "2026-07-18"},
                {"date": "2026-07-19"},
                {"date": "2026-07-24"},
            ]
        }

    def test_friday_lookback_10_days_excludes_friday(self):
        fri = date(2026, 7, 24)
        out = asc._recent_opening_dates(
            self._cal(), days=10, anchor=fri, include_anchor=False
        )
        self.assertEqual(out, ["20260718", "20260719"])

    def test_lookback_window_start_boundary(self):
        cal = {
            "race_days": [
                {"date": "2026-07-06"},
                {"date": "2026-07-12"},
            ]
        }
        fri = date(2026, 7, 17)
        recent = asc._recent_opening_dates(cal, days=10, anchor=fri, include_anchor=False)
        self.assertNotIn("20260706", recent)
        self.assertIn("20260712", recent)


if __name__ == "__main__":
    unittest.main()
