"""weekly-update 後処理（coverage / megu / status）のテスト。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.scraper import auto_scrape as asc


class TestFinalizeWeeklyUpdate(unittest.TestCase):
    def setUp(self):
        p = patch.object(asc, "megu_missing_dates", return_value=[])   # 手元の実データの欠けを混ぜない
        p.start()
        self.addCleanup(p.stop)

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


class TestMeguCatchup(unittest.TestCase):
    @patch.object(asc, "_trigger_pg_sync_for_dates", return_value={"status": "ok"})
    @patch.object(asc, "_trigger_megu_batch_for_dates", return_value={"status": "ok"})
    @patch.object(asc, "_update_dates_coverage_bg")
    @patch.object(asc, "megu_missing_dates", return_value=["20261031", "20261101", "20261107"])
    def test_finalize_adds_missing_megu_dates(self, _miss, _cov, mock_megu, _pg):
        out = asc.finalize_weekly_update(["2026-11-07", "20261108"], {"status": "ok"})
        self.assertEqual(mock_megu.call_args[0][0], ["20261031", "20261101", "20261107", "20261108"])
        self.assertEqual(out["megu_catchup_dates"], ["20261031", "20261101"])
        self.assertEqual(_pg.call_args[0][0], ["20261031", "20261101", "20261107", "20261108"])   # PG 同期・索引にも含める
        self.assertEqual(_cov.call_args[0][0], ["20261031", "20261101", "20261107", "20261108"])

    @patch.object(asc, "_trigger_pg_sync_for_dates", return_value={"status": "ok"})
    @patch.object(asc, "_trigger_megu_batch_for_dates", return_value={"status": "ok"})
    @patch.object(asc, "_update_dates_coverage_bg")
    @patch.object(asc, "megu_missing_dates", side_effect=OSError("x"))
    def test_finalize_still_runs_when_catchup_check_fails(self, _miss, _cov, mock_megu, _pg):
        asc.finalize_weekly_update(["20261108"], {"status": "ok"})
        self.assertEqual(mock_megu.call_args[0][0], ["20261108"])


def test_megu_missing_dates_picks_days_after_last_done(tmp_path):
    from datetime import date

    import pandas as pd

    y = tmp_path / "2026"
    y.mkdir()
    days = {"2026-10-25": "r1", "2026-10-31": "r2", "2026-11-01": "r3", "2026-11-08": "r4", "2026-10-11": "r0", "2026-11-12": "r5"}
    pd.DataFrame({"date": list(days), "race_id": list(days.values())}).to_parquet(y / "race_result_flat.parquet")
    pd.DataFrame({"race_id": ["r1"]}).to_parquet(y / "megu_index_flat.parquet")   # 10/25 まで作れた。10/11 は古い欠け
    got = asc.megu_missing_dates(date(2026, 11, 12), tables_dir=tmp_path)
    assert got == ["20261011", "20261031", "20261101", "20261108"]      # 窓の中の欠けは全部（後ろの日が先に埋まっても前の欠けを拾う）・今日は除く
    assert asc.megu_missing_dates(date(2026, 11, 12), tables_dir=tmp_path, max_days=5) == ["20261108"]


def test_megu_missing_dates_across_years_and_bad_dates(tmp_path):
    from datetime import date

    import pandas as pd

    for y, rows in {"2025": [("2025-12-27", "a"), ("2025-12-28", "b")], "2026": [("2026-01-04", "c"), ("bad", "d")]}.items():
        (tmp_path / y).mkdir()
        pd.DataFrame(rows, columns=["date", "race_id"]).to_parquet(tmp_path / y / "race_result_flat.parquet")
    pd.DataFrame({"race_id": ["b"]}).to_parquet(tmp_path / "2025" / "megu_index_flat.parquet")
    assert asc.megu_missing_dates(date(2026, 1, 10), tables_dir=tmp_path) == ["20251227", "20260104"]   # 前年の分も見る・不正な日付は除く


def test_queue_weekly_returns_wait_failed_without_finalize(monkeypatch):
    from src.scraper import auto_scrape_queue as q

    called = []
    monkeypatch.setattr(q, "_storage", lambda: object())
    monkeypatch.setattr(q, "_queue", lambda: type("Q", (), {"bulk_add_jobs": lambda self, j: {"created": len(j)}})())
    monkeypatch.setattr(q, "_ensure_race_list_date", lambda *a: True)
    import src.scraper.period_runners as pr

    monkeypatch.setattr(pr, "collect_jra_race_job_specs_for_period", lambda *a, **k: ([{"race_id": "202605050101"}], None))
    monkeypatch.setattr(q, "_kick_and_wait", lambda *a, **k: (False, "paused", {}))
    monkeypatch.setattr(asc, "finalize_weekly_update", lambda *a, **k: called.append(1))
    out = q.run_weekly_update_for_dates(["20261031"])
    assert out["queue_wait_failed"] and out["reason"] == "queue-wait:paused" and not called


def test_weekly_queue_wait_failure_notifies_and_returns_nonzero(monkeypatch):
    from datetime import date

    sent = []
    monkeypatch.setattr(asc, "_today_jst", lambda: date(2026, 11, 6))          # 金曜
    monkeypatch.setattr(asc, "_load_race_calendar", lambda: {})
    monkeypatch.setattr(asc, "_recent_opening_dates", lambda cal, n: ["20261031", "20261101"])
    monkeypatch.setattr(asc, "_save_status", lambda *a, **k: None)
    monkeypatch.setattr(asc, "run_weekly_update_for_dates",
                        lambda ds: {"status": "error", "reason": "queue-wait:paused", "queue_wait_failed": True})
    import src.utils.notify as notify

    monkeypatch.setattr(notify, "notify_slack", lambda m, **k: sent.append(m) or True)
    assert asc.task_weekly_update() == asc.WEEKLY_QUEUE_WAIT_FAILED_EXIT
    assert sent and "queue-wait:paused" in sent[0]
    monkeypatch.setattr(asc, "run_weekly_update_for_dates", lambda ds: {"status": "ok"})
    assert asc.task_weekly_update() is None


def test_main_passes_task_exit_code_to_cron(monkeypatch):
    import sys

    import pytest

    monkeypatch.setattr(asc, "script_basic_config", lambda *a, **k: None)   # ルートロガーにハンドラを足さない
    monkeypatch.setitem(asc.TASKS, "weekly-update", lambda: asc.WEEKLY_QUEUE_WAIT_FAILED_EXIT)
    monkeypatch.setattr(sys, "argv", ["auto_scrape", "--task", "weekly-update"])
    with pytest.raises(SystemExit) as ei:
        asc.main()
    assert ei.value.code == 7
    monkeypatch.setitem(asc.TASKS, "weekly-update", lambda: None)
    asc.main()                                                              # 返り値なしは従来どおり 0


if __name__ == "__main__":
    unittest.main()
