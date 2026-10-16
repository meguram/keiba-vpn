"""開催日ランナーの直接実行（キューを待たない）のテスト。"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from src.scraper import auto_scrape_queue as aq


class DirectTest(unittest.TestCase):
    def test_env_switch(self):
        with mock.patch.dict(os.environ, {"KEIBA_RACEDAY_DIRECT": "1"}):
            self.assertTrue(aq._raceday_direct_enabled())
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("KEIBA_RACEDAY_DIRECT", None)
            self.assertFalse(aq._raceday_direct_enabled())          # 既定は従来どおり（承認後に有効化）

    def test_runs_tasks_in_order_and_collects_errors(self):
        calls = []

        def fake(runner, rid, task, date, smart_skip=True):
            calls.append((rid, task, smart_skip))
            if task == "race_pair_odds_t60":
                raise RuntimeError("boom")

        with mock.patch("src.scraper.queue_tasks._race_task", fake), mock.patch.object(aq.time, "sleep", lambda *_: None):
            errs = aq._run_race_tasks_direct(object(), "202605040311", "20261017", ["race_odds_t60", "race_pair_odds_t60"])
        self.assertEqual([c[1] for c in calls], ["race_odds_t60", "race_pair_odds_t60"])
        self.assertTrue(all(c[2] is False for c in calls))       # 直前オッズは常に取り直す
        self.assertEqual(len(errs), 1)
        self.assertIn("race_pair_odds_t60", errs[0])

    def test_access_restriction_stops_remaining_tasks(self):
        from src.scraper.access_guard import AccessRestrictionDetected

        calls = []

        def fake(runner, rid, task, date, smart_skip=True):
            calls.append(task)
            raise AccessRestrictionDetected("https://x", 403)

        with mock.patch("src.scraper.queue_tasks._race_task", fake), mock.patch.object(aq.time, "sleep", lambda *_: None):
            errs = aq._run_race_tasks_direct(object(), "r", "20261017", ["a", "b", "c"])
        self.assertEqual(calls, ["a"])                             # 制限の疑いが出たらそのレースの残りは止める
        self.assertIn("access_restriction", errs[0])


if __name__ == "__main__":
    unittest.main()
