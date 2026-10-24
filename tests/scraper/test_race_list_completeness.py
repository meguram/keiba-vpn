"""race_lists 完備判定・上書き可否のテスト。"""

import unittest

from src.scraper.race_list_completeness import (
    is_race_list_complete,
    race_list_stats,
    should_replace_race_list,
)


def _race(venue_code: str, rnd: int) -> dict:
    return {
        "race_id": f"2026{venue_code}0108{rnd:02d}",
        "round": rnd,
        "venue": venue_code,
    }


def _full_day(venue_codes: list[str]) -> dict:
    races = []
    for vc in venue_codes:
        races.extend(_race(vc, r) for r in range(1, 13))
    return {"date": "20260711", "races": races}


class TestRaceListCompleteness(unittest.TestCase):
    def test_empty_not_complete(self):
        self.assertFalse(is_race_list_complete({"date": "20260711", "races": []}))

    def test_partial_6_not_complete(self):
        data = {"date": "20260711", "races": [_race("02", r) for r in range(1, 7)]}
        st = race_list_stats(data)
        self.assertFalse(st.is_complete)
        self.assertEqual(st.jra_count, 6)

    def test_full_36_complete(self):
        data = _full_day(["02", "03", "10"])
        st = race_list_stats(data)
        self.assertTrue(st.is_complete)
        self.assertEqual(st.jra_count, 36)

    def test_full_12_single_venue(self):
        data = _full_day(["05"])
        self.assertTrue(is_race_list_complete(data))

    def test_missing_round_not_complete(self):
        races = [_race("02", r) for r in range(1, 12)]  # 1〜11R のみ
        self.assertFalse(is_race_list_complete({"races": races}))

    def test_no_race_scheduled_complete(self):
        data = {"date": "20260711", "races": [], "_meta": {"note": "no_race_scheduled"}}
        self.assertTrue(is_race_list_complete(data))

    def test_should_not_downgrade_complete(self):
        full = _full_day(["02", "03", "10"])
        partial = {"date": "20260711", "races": [_race("02", r) for r in range(1, 7)]}
        self.assertFalse(should_replace_race_list(full, partial))

    def test_should_upgrade_partial(self):
        old = {"date": "20260711", "races": [_race("02", r) for r in range(1, 7)]}
        new = {"date": "20260711", "races": [_race("02", r) for r in range(1, 10)]}
        self.assertTrue(should_replace_race_list(old, new))

    def test_should_not_overwrite_with_empty(self):
        old = {"date": "20260711", "races": [_race("02", r) for r in range(1, 7)]}
        new = {"date": "20260711", "races": []}
        self.assertFalse(should_replace_race_list(old, new))

    def test_should_replace_incomplete_with_complete(self):
        old = {"date": "20260711", "races": [_race("02", r) for r in range(1, 7)]}
        new = _full_day(["02", "03", "10"])
        self.assertTrue(should_replace_race_list(old, new))


if __name__ == "__main__":
    unittest.main()
