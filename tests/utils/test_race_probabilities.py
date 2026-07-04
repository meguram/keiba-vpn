"""race_probabilities のユニットテスト。"""
from __future__ import annotations

import unittest

from src.utils.race_probabilities import (
    assign_mece_marks,
    buy_recommendation_tier,
    derive_race_probabilities,
    qualifies_star_mark,
)


class TestRaceProbabilities(unittest.TestCase):
    def test_win_probs_sum_to_one(self):
        probs = derive_race_probabilities([10.0, 8.0, 6.0, 4.0])
        total = sum(p["win_prob"] for p in probs)
        self.assertAlmostEqual(total, 1.0, places=3)

    def test_top3_ge_top2_ge_win(self):
        probs = derive_race_probabilities([12.0, 9.0, 7.0, 5.0, 3.0])
        for p in probs:
            self.assertGreaterEqual(p["top3_prob"], p["top2_prob"])
            self.assertGreaterEqual(p["top2_prob"], p["win_prob"])

    def test_mece_honmei_and_star_at_most_one(self):
        entries = [
            {"horse_number": 1, "win_prob": 0.28, "top2_prob": 0.45, "top3_prob": 0.6,
             "win_odds": 15.0, "ev_win": 1.2, "ev_place": 1.0},
            {"horse_number": 2, "win_prob": 0.18, "top2_prob": 0.4, "top3_prob": 0.55,
             "win_odds": 8.0, "ev_win": 0.8, "ev_place": 0.9},
            {"horse_number": 3, "win_prob": 0.05, "top2_prob": 0.28, "top3_prob": 0.42},
            {"horse_number": 4, "win_prob": 0.04, "top2_prob": 0.26, "top3_prob": 0.38},
        ]
        assign_mece_marks(entries)
        types = [e["mark_type"] for e in entries]
        self.assertEqual(types.count("honmei"), 1)
        self.assertLessEqual(types.count("star"), 1)

    def test_mece_allows_multiple_pair(self):
        entries = [
            {"horse_number": 1, "win_prob": 0.22, "top2_prob": 0.5, "top3_prob": 0.6},
            {"horse_number": 2, "win_prob": 0.1, "top2_prob": 0.45, "top3_prob": 0.5},
            {"horse_number": 3, "win_prob": 0.08, "top2_prob": 0.4, "top3_prob": 0.48},
            {"horse_number": 4, "win_prob": 0.05, "top2_prob": 0.15, "top3_prob": 0.3},
        ]
        assign_mece_marks(entries)
        pairs = [e for e in entries if e["mark_type"] == "pair"]
        self.assertGreaterEqual(len(pairs), 2)

    def test_qualifies_star_odds_band(self):
        self.assertTrue(
            qualifies_star_mark(
                {"win_odds": 15.0, "ev_win": 1.1, "ev_place": 0.5, "top3_prob": 0.2},
                0.1,
            )
        )
        self.assertFalse(
            qualifies_star_mark(
                {"win_odds": 5.0, "ev_win": 1.2, "ev_place": 0.5, "top3_prob": 0.2},
                0.1,
            )
        )

    def test_assign_star_mark(self):
        entries = [
            {"horse_number": 1, "win_prob": 0.2, "top2_prob": 0.35, "top3_prob": 0.5,
             "win_odds": 15.0, "ev_win": 1.2, "ev_place": 1.0},
            {"horse_number": 2, "win_prob": 0.15, "top2_prob": 0.3, "top3_prob": 0.45,
             "win_odds": 8.0, "ev_win": 0.8, "ev_place": 0.9},
        ]
        assign_mece_marks(entries)
        stars = [e for e in entries if e["mark_type"] == "star"]
        self.assertEqual(len(stars), 1)
        self.assertEqual(stars[0]["horse_number"], 1)

    def test_buy_recommendation_tier(self):
        self.assertEqual(buy_recommendation_tier({"ev_win": 1.3}), "強推奨")
        self.assertEqual(buy_recommendation_tier({"ev_place": 0.95}), "様子見")


if __name__ == "__main__":
    unittest.main()
