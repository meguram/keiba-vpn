"""SLA 仕様（sla_spec）とタスクカタログ・スケジュール定義の整合を検査する。"""

from __future__ import annotations

import importlib
import unittest
from datetime import datetime, timedelta, timezone

from src.scraper import sla_spec
from src.scraper.auto_scrape import TASKS, barometer_due_dates
from src.scraper.queue_tasks import TASK_CATALOG

JST = timezone(timedelta(hours=9))


class SlaSpecTests(unittest.TestCase):
    def test_every_catalog_task_belongs_to_an_endpoint(self):
        covered = {t for e in sla_spec.ENDPOINTS for t in e.task_ids}
        missing = [
            t["id"] for t in TASK_CATALOG
            if t["id"] not in covered and t["id"] not in sla_spec.AGGREGATE_TASK_IDS
        ]
        self.assertEqual(missing, [], "TASK_CATALOG に足したタスクは sla_spec.ENDPOINTS にも載せること")

    def test_spec_task_ids_exist_in_catalog(self):
        ids = {t["id"] for t in TASK_CATALOG}
        bad = [(e.id, t) for e in sla_spec.ENDPOINTS for t in e.task_ids if t not in ids]
        self.assertEqual(bad, [])

    def test_fetch_jobs_are_real(self):
        for e in sla_spec.ENDPOINTS:
            for f in e.fetches:
                ok = f.job in TASKS or f.job.startswith(sla_spec.NON_TASK_JOBS_PREFIXES)
                self.assertTrue(ok, f"{e.id}: 未知のジョブ {f.job}")

    def test_declared_task_lists_contain_the_task(self):
        for e in sla_spec.ENDPOINTS:
            for f in e.fetches:
                if not f.via:
                    continue
                mod, _, name = f.via.rpartition(".")
                tasks = getattr(importlib.import_module(mod), name)
                self.assertIn(f.task, tasks, f"{e.id}: {f.via} に {f.task} が無い（スケジュールと仕様が食い違い）")

    def test_known_gaps_match_endpoints_without_schedule(self):
        self.assertEqual(
            sla_spec.endpoints_without_schedule(),
            set(sla_spec.KNOWN_GAPS),
            "定期取得経路の有無が変わった。KNOWN_GAPS と ENDPOINTS.fetches を揃えること",
        )

    def test_required_endpoints_have_a_schedule(self):
        for e in sla_spec.ENDPOINTS:
            if e.required:
                self.assertTrue(e.fetches, f"{e.id} は UI/予測に必須なのに定期取得が無い")

    def test_endpoint_ids_unique(self):
        ids = [e.id for e in sla_spec.ENDPOINTS]
        self.assertEqual(len(ids), len(set(ids)))


class BarometerRetryTests(unittest.TestCase):
    CAL = {"race_days": [{"date": d} for d in ("2026-10-10", "2026-10-11", "2026-10-12", "2026-10-17")]}

    def _due(self, ts: str, lookback: int = 30):
        return barometer_due_dates(self.CAL, datetime.fromisoformat(ts).replace(tzinfo=JST), lookback)

    def test_not_due_before_next_friday_18(self):
        self.assertEqual(self._due("2026-10-16 17:59"), [])

    def test_due_after_next_friday_18(self):
        self.assertEqual(self._due("2026-10-16 18:00"), ["20261010", "20261011", "20261012"])

    def test_next_week_races_not_due_yet(self):
        self.assertNotIn("20261017", self._due("2026-10-23 17:00"))

    def test_lookback_limits_old_dates(self):
        self.assertEqual(self._due("2026-12-01 12:00", lookback=10), [])


if __name__ == "__main__":
    unittest.main()
