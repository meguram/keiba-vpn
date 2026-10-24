"""race_shutuba / race_result マージのユニットテスト。"""
from __future__ import annotations

import unittest

from src.utils.race_card_merge import (
    is_plausible_sex_age,
    merge_race_card,
    pick_better_text,
)


class TestRaceCardMerge(unittest.TestCase):
    def test_pick_better_text_prefers_valid_japanese(self):
        self.assertEqual(pick_better_text("3罩恰", "3歳未勝利"), "3歳未勝利")
        self.assertEqual(pick_better_text("3歳未勝利", ""), "3歳未勝利")

    def test_merge_race_card_fills_distance_and_surface(self):
        shutuba = {
            "race_name": "3罩恰",
            "venue": "",
            "surface": "",
            "distance": 0,
            "entries": [{"horse_id": "h1", "horse_name": "壊", "sex_age": "3"}],
        }
        result = {
            "race_name": "3歳未勝利",
            "venue": "中山",
            "surface": "ダート",
            "distance": 1200,
            "entries": [{"horse_id": "h1", "horse_name": "アイスメルティング", "sex_age": "牝3"}],
        }
        merged = merge_race_card(shutuba, result)
        assert merged is not None
        self.assertEqual(merged["race_name"], "3歳未勝利")
        self.assertEqual(merged["venue"], "中山")
        self.assertEqual(merged["surface"], "ダート")
        self.assertEqual(merged["distance"], 1200)
        self.assertEqual(merged["entries"][0]["horse_name"], "アイスメルティング")
        self.assertEqual(merged["entries"][0]["sex_age"], "牝3")

    def test_is_plausible_sex_age(self):
        self.assertTrue(is_plausible_sex_age("牝3"))
        self.assertFalse(is_plausible_sex_age("3"))
        self.assertFalse(is_plausible_sex_age(""))
