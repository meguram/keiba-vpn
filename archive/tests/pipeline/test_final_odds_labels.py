"""final_odds ラベル抽出のユニットテスト。"""
from __future__ import annotations

import unittest

from src.pipeline.models.final_odds_dataset import _odds_labels_for_race


class _FakeStorage:
    def __init__(self, odds: dict | None):
        self._odds = odds

    def load(self, category: str, key: str):
        if category == "race_odds":
            return self._odds
        return None


class TestFinalOddsLabels(unittest.TestCase):
    def test_labels_from_result_and_odds(self):
        result = {
            "entries": [
                {"horse_number": 1, "odds": 4.5},
                {"horse_number": 2, "odds": 12.0},
            ]
        }
        odds = {
            "entries": [
                {"horse_number": 1, "place_odds_min": 1.4, "place_odds_max": 2.1},
                {"horse_number": 2, "place_odds_min": 2.5, "place_odds_max": 4.0},
            ]
        }
        labels = _odds_labels_for_race(_FakeStorage(odds), "202401010101", result)
        self.assertEqual(labels[1]["win_odds"], 4.5)
        self.assertEqual(labels[1]["place_min"], 1.4)
        self.assertEqual(labels[2]["place_max"], 4.0)


if __name__ == "__main__":
    unittest.main()
