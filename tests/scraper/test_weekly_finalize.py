"""weekly-update 後処理（coverage / megu / status）のテスト。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.scraper import auto_scrape as asc


class TestFinalizeWeeklyUpdate(unittest.TestCase):
    @patch.object(asc, "_trigger_pg_sync_for_dates")
    @patch.object(asc, "_trigger_megu_batch_for_dates")
    @patch.object(asc, "_update_dates_coverage_bg")
    def test_finalize_calls_coverage_megu_pg(
        self,
        mock_cov,
        mock_megu,
        mock_pg,
    ):
        mock_megu.return_value = {"status": "ok", "megu_valid": 10, "megu_oor": 1}
        mock_pg.return_value = {"status": "ok", "synced": 5}

        result = asc.finalize_weekly_update(
            ["20260718", "20260719"],
            {"status": "ok", "races": 72},
        )

        mock_cov.assert_called_once_with(["20260718", "20260719"])
        mock_megu.assert_called_once()
        mock_pg.assert_called_once()
        self.assertEqual(result["status"], "ok")
        self.assertIn("last_run", result)
        self.assertIn("megu_batch", result)
        self.assertEqual(result["lookback_days"], asc.WEEKLY_LOOKBACK_DAYS)

    @patch.object(asc, "_trigger_pg_sync_for_dates")
    @patch.object(asc, "_trigger_megu_batch_for_dates")
    @patch.object(asc, "_update_dates_coverage_bg")
    def test_finalize_partial_on_megu_error(
        self,
        mock_cov,
        mock_megu,
        mock_pg,
    ):
        mock_megu.return_value = {"status": "error", "error": "boom"}
        mock_pg.return_value = {"status": "skipped"}

        result = asc.finalize_weekly_update(
            ["20260718"],
            {"status": "ok"},
        )
        self.assertEqual(result["status"], "partial")


if __name__ == "__main__":
    unittest.main()
