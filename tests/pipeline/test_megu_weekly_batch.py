"""compute_megu_for_opening_dates のユニットテスト。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.pipeline.megu_index import compute as mc


class TestComputeMeguForOpeningDates(unittest.TestCase):
    @patch.object(mc, "compute_for_date")
    def test_batch_aggregates_results(self, mock_compute):
        mock_compute.side_effect = [
            {"status": "ok", "megu_valid": 30, "megu_oor": 2},
            {"status": "skipped", "megu_valid": 0, "megu_oor": 0, "reason": "no_data"},
        ]

        out = mc.compute_megu_for_opening_dates(
            ["20260718", "20260719"],
            gcs_canonical=True,
        )

        self.assertEqual(mock_compute.call_count, 2)
        self.assertEqual(out["megu_valid"], 30)
        self.assertEqual(out["megu_oor"], 2)
        self.assertEqual(out["status"], "partial")
        self.assertTrue(out["gcs_canonical"])


if __name__ == "__main__":
    unittest.main()
