"""race_bet_suggestion のユニットテスト。"""
from __future__ import annotations

import unittest

from src.utils.race_bet_suggestion import suggest_race_bets


class TestRaceBetSuggestion(unittest.TestCase):
    def _base_entries(self) -> list[dict]:
        return [
            {
                "horse_number": 12,
                "horse_name": "A",
                "mark_type": "honmei",
                "win_prob": 0.2,
                "top2_prob": 0.4,
                "top3_prob": 0.55,
                "win_odds": 4.5,
                "ev_win": 0.9,
                "ev_top2": 1.1,
                "ev_place": 1.0,
            },
            {
                "horse_number": 8,
                "horse_name": "B",
                "mark_type": "pair",
                "win_prob": 0.15,
                "top2_prob": 0.38,
                "top3_prob": 0.5,
                "win_odds": 6.0,
                "ev_win": 0.7,
                "ev_top2": 1.05,
                "ev_place": 1.1,
            },
            {
                "horse_number": 7,
                "horse_name": "C",
                "mark_type": "anchor",
                "win_prob": 0.08,
                "top2_prob": 0.22,
                "top3_prob": 0.42,
                "win_odds": 12.0,
                "ev_win": 0.6,
                "ev_top2": 0.9,
                "ev_place": 1.05,
            },
            {
                "horse_number": 3,
                "horse_name": "D",
                "mark_type": "show_val",
                "win_prob": 0.05,
                "top2_prob": 0.15,
                "top3_prob": 0.35,
                "win_odds": 25.0,
                "ev_win": 0.5,
                "ev_top2": 0.8,
                "ev_place": 1.15,
            },
        ]

    def test_suggests_umaren_for_honmei_pair(self):
        out = suggest_race_bets(self._base_entries())
        types = [p["bet_type"] for p in out["picks"]]
        self.assertIn("umaren", types)
        umaren = next(p for p in out["picks"] if p["bet_type"] == "umaren")
        self.assertEqual(umaren["horses"], [12, 8])
        self.assertIn("◎", umaren["marks"])

    def test_suggests_fukusho_for_show_val(self):
        out = suggest_race_bets(self._base_entries())
        fukusho = [p for p in out["picks"] if p["bet_type"] == "fukusho"]
        self.assertTrue(any(p["horses"] == [3] for p in fukusho))

    def test_empty_entries(self):
        out = suggest_race_bets([])
        self.assertEqual(out["picks"], [])


if __name__ == "__main__":
    unittest.main()
