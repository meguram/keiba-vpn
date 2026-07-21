"""monitor_future_eligible: 開催日フィルタのユニットテスト。"""

import unittest

from src.scraper.monitor_future_eligible import date_has_at_least_one_jra_race


class TestDateHasAtLeastOneJraRace(unittest.TestCase):
    def test_no_race_scheduled_meta(self):
        self.assertFalse(
            date_has_at_least_one_jra_race("20260101", [], {"note": "no_race_scheduled"})
        )

    def test_empty_races(self):
        self.assertFalse(date_has_at_least_one_jra_race("20260101", [], None))

    def test_nar_only(self):
        # 月 11 → race_id[4:6]="11" は JRA 会場コード外
        races = [{"race_id": "202611010501", "race_name": "地方"}]
        self.assertFalse(date_has_at_least_one_jra_race("20261101", races, None))

    def test_one_jra_race(self):
        races = [{"race_id": "202601010501", "race_name": "3歳未勝利"}]
        self.assertTrue(date_has_at_least_one_jra_race("20260101", races, None))

    def test_weekday_with_jra(self):
        """平日でも JRA レースがあれば開催日として扱う。"""
        races = [{"race_id": "202601070501", "race_name": "4歳以上1勝"}]
        self.assertTrue(date_has_at_least_one_jra_race("20260107", races, None))


if __name__ == "__main__":
    unittest.main()
