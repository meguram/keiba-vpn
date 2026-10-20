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

    def test_trio_uses_maybe_partners_and_symbol(self):
        def e(hn, mark, ev_top2, ev_place, ev_win=0.5):
            return {"horse_number": hn, "horse_name": str(hn), "mark_type": mark, "win_prob": 0.1, "top2_prob": 0.3,
                    "top3_prob": 0.4, "win_odds": 10.0, "ev_win": ev_win, "ev_top2": ev_top2, "ev_place": ev_place}

        entries = [e(12, "honmei", 0.5, 0.94), e(8, "pair", 0.5, 0.94), e(7, "anchor", 0.5, 1.0), e(5, "anchor", 0.5, 0.99)]
        out = suggest_race_bets(entries)
        trios = [p for p in out["picks"] if p["bet_type"] == "trio"]
        self.assertEqual(len(trios), 2)
        for p in trios:
            self.assertEqual(p["marks"], ["◎", "○", "△"])
            self.assertEqual(p["horses"][:2], [12, 8])
            self.assertIn(p["horses"][2], (7, 5))
            self.assertEqual(len(p["selection"].split("-")), 3)
        self.assertEqual(len({p["selection"] for p in trios}), 2)

    def test_old_saved_data_with_multiple_pairs_still_works(self):
        entries = self._base_entries()
        entries.append({
            "horse_number": 9, "horse_name": "F", "mark_type": "pair", "win_prob": 0.07, "top2_prob": 0.33,
            "top3_prob": 0.45, "win_odds": 9.0, "ev_win": 0.7, "ev_top2": 1.04, "ev_place": 1.0,
        })
        out = suggest_race_bets(entries)
        self.assertTrue(any(p["bet_type"] == "umaren" for p in out["picks"]))

    def test_empty_entries(self):
        out = suggest_race_bets([])
        self.assertEqual(out["picks"], [])


if __name__ == "__main__":
    unittest.main()
