"""distance_band 共通区分のユニットテスト。"""
from __future__ import annotations

import unittest

from src.utils.distance_band import distance_group_key, distance_group_label_ja


class TestDistanceBand(unittest.TestCase):
    def test_sprint_under_1500(self):
        self.assertEqual(distance_group_key(1200), "sprint")
        self.assertEqual(distance_group_key(1400), "sprint")
        self.assertEqual(distance_group_key(1499), "sprint")

    def test_mile_1500_to_1799(self):
        self.assertEqual(distance_group_key(1500), "mile")
        self.assertEqual(distance_group_key(1600), "mile")
        self.assertEqual(distance_group_key(1799), "mile")

    def test_middle_1800_to_2399(self):
        self.assertEqual(distance_group_key(1800), "middle")
        self.assertEqual(distance_group_key(2200), "middle")
        self.assertEqual(distance_group_key(2399), "middle")

    def test_long_from_2400(self):
        self.assertEqual(distance_group_key(2400), "long")
        self.assertEqual(distance_group_key(3200), "long")
        self.assertEqual(distance_group_label_ja("long"), "長距離")


if __name__ == "__main__":
    unittest.main()
