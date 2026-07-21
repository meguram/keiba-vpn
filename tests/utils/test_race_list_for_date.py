"""race_list_for_date ユニットテスト。"""

from __future__ import annotations

import unittest

from src.utils.race_list_for_date import decode_race_id, opening_date_display, opening_date_kind


class TestRaceListForDate(unittest.TestCase):
    def test_decode_race_id_not_calendar_date(self):
        d = decode_race_id("202007010201")
        self.assertEqual(d["year"], "2020")
        self.assertEqual(d["venue"], "中京")
        self.assertEqual(d["kaisai_round"], 1)
        self.assertEqual(d["kaisai_day"], 2)
        self.assertNotEqual(d["kaisai_day"], 7)  # 07 は場コード

    def test_opening_date_kind_no_meeting(self):
        self.assertEqual(opening_date_kind("20220110"), "no_meeting")

    def test_opening_date_kind_meeting(self):
        self.assertEqual(opening_date_kind("20200301"), "meeting")

    def test_opening_date_display_no_meeting(self):
        d = opening_date_display("20220110")
        self.assertEqual(d["kind"], "no_meeting")
        self.assertEqual(d["label"], "非開催（対象外）")
        self.assertFalse(d["quality_applicable"])
        self.assertFalse(d["monitor_data_applicable"])


if __name__ == "__main__":
    unittest.main()
