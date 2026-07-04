"""race_result 表示補完のユニットテスト。"""
from __future__ import annotations

import unittest

from src.utils.race_result_display import (
    merge_race_result_entries_lap,
    merge_race_result_payoff,
    normalize_payoff_for_display,
    prepare_race_result_display,
)


class TestRaceResultDisplay(unittest.TestCase):
    def test_merge_entries_lap_fills_missing(self):
        entries = [{"horse_number": 1, "last_3f": 0}]
        lap = [{"horse_number": 1, "passing_order": "3-3-2", "last_3f": 33.5}]
        out = merge_race_result_entries_lap(entries, lap)
        self.assertEqual(out[0]["passing_order"], "3-3-2")
        self.assertEqual(out[0]["last_3f"], 33.5)

    def test_prepare_prefers_race_result(self):
        rr = {
            "entries": [{"horse_number": 1, "finish_position": 1}],
            "pace": {},
        }
        on_time = {"entries": [{"horse_number": 2}]}
        lap = {
            "entries_lap": [{"horse_number": 1, "passing_order": "1-1-1", "last_3f": 34.0}],
            "lap_times": [11.0, 11.5, 12.0, 12.0, 12.0, 12.1, 12.2, 12.3],
            "pace": {"first_half_3f": 35.0},
        }
        out = prepare_race_result_display(rr, on_time, lap)
        self.assertIsNotNone(out)
        self.assertEqual(out["_display_source"], "race_result")
        self.assertEqual(out["entries"][0]["passing_order"], "1-1-1")
        self.assertEqual(len(out["lap_times"]), 8)
        self.assertEqual(out["pace"]["first_half_3f"], 35.0)
        self.assertIn("lap_pattern", out)
        self.assertIn("label", out["lap_pattern"])

    def test_normalize_payoff_japanese_keys(self):
        raw = {
            "単勝": {"numbers": "12", "payout": "190", "popularity": "1"},
            "複勝": [{"numbers": "12", "payout": "110", "popularity": "1"}],
        }
        out = normalize_payoff_for_display(raw)
        self.assertEqual(list(out.keys()), ["単勝", "複勝"])
        self.assertEqual(out["単勝"]["payout"], "190")

    def test_merge_payoff_from_race_result(self):
        rr = {"payoff": {"単勝": {"numbers": "1", "payout": "100"}}}
        out = merge_race_result_payoff(rr, None)
        self.assertIn("単勝", out)

    def test_prepare_merges_payoff(self):
        rr = {
            "entries": [{"horse_number": 1}],
            "payoff": {"単勝": {"numbers": "12", "payout": "1,530"}},
        }
        out = prepare_race_result_display(rr, None, None)
        self.assertEqual(out["payoff"]["単勝"]["payout"], "1,530")

    def test_prepare_fallback_on_time(self):
        lap = {"entries_lap": [], "lap_times": [], "pace": {}}
        out = prepare_race_result_display(None, {"entries": [{"horse_number": 3}]}, lap)
        self.assertEqual(out["_display_source"], "race_result_on_time")


if __name__ == "__main__":
    unittest.main()
