"""race_quality_rules ユニットテスト。"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from src.utils import race_quality_rules as rqr


class TestRaceQualityRules(unittest.TestCase):
    def test_obstacle_race_by_name(self):
        rr = {"race_name": "障害3歳以上未勝利", "surface": "芝"}
        self.assertTrue(rqr.is_obstacle_race(rr))

    def test_audit_obstacle_no_gaps(self):
        rr = {"race_name": "障害OP", "surface": "芝", "distance": 3000, "entries": []}
        self.assertEqual(rqr.audit_gcs_race_result_for_health(rr), [])

    def test_audit_turf_with_lap_gap(self):
        rr = {
            "race_name": "3歳未勝利",
            "surface": "芝",
            "distance": 1600,
            "entries": [{"finish_position": 1, "time_sec": 95.0}],
        }
        gaps = rqr.audit_gcs_race_result_for_health(rr)
        self.assertIn("gcs_no_lap_times", gaps)

    def test_meta_only_gap_na_with_finishers_and_lap(self):
        rr = {
            "race_name": "3歳未勝利",
            "surface": "",
            "distance": 0,
            "lap_times": [12.0, 11.0],
            "entries": [{"finish_position": 1, "time_sec": 95.0}],
        }
        self.assertEqual(rqr.audit_gcs_race_result_for_health(rr), [])

    def test_embedded_lap_na(self):
        storage = MagicMock()
        row = {"race_result": True, "race_result_lap": False}
        rr = {"lap_times": [12.0, 11.5]}
        out = rqr.apply_embedded_lap_presence_na(row, "202001010101", storage, rr=rr)
        self.assertIsNone(out["race_result_lap"])

    def test_obstacle_presence_na(self):
        row = {"race_result_lap": False, "race_barometer": False, "race_shutuba": True}
        out = rqr.apply_obstacle_presence_na(row, obstacle=True)
        self.assertIsNone(out["race_result_lap"])
        self.assertIsNone(out["race_barometer"])
        self.assertTrue(out["race_shutuba"])


if __name__ == "__main__":
    unittest.main()
