"""quality_auto_remediation ユニットテスト。"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.api import quality_auto_remediation as qar


class TestQualityAutoRemediation(unittest.TestCase):
    def test_build_plan_presence_gcs_missing(self):
        health = {
            "checks": {
                "presence": {
                    "status": "fail",
                    "issues": [
                        {
                            "race_id": "202006010101",
                            "kind": "gcs_missing",
                            "categories": ["race_result_lap_times", "race_result"],
                        },
                    ],
                },
                "raw_content": {"status": "ok", "issues": []},
                "calculated": {"status": "ok", "issues": []},
            },
        }
        with patch.object(qar, "get_health_view", return_value={"date": "20200301", "checks": health["checks"]}):
            with patch("src.scraper.date_coverage.load_not_available", return_value=set()):
                plan = qar.build_remediation_plan("20200301", health)
        scrape = [a for a in plan["actions"] if a["kind"] == "scrape_race"]
        self.assertEqual(len(scrape), 1)
        self.assertIn("race_result", scrape[0]["tasks"])
        self.assertIn("race_result_lap", scrape[0]["tasks"])

    def test_build_plan_skips_no_meeting(self):
        health = {
            "checks": {
                "presence": {
                    "status": "unknown",
                    "summary": {"reason": "no_meeting"},
                    "issues": [],
                },
                "raw_content": {
                    "status": "unknown",
                    "summary": {"reason": "no_meeting"},
                    "issues": [],
                },
                "calculated": {
                    "status": "unknown",
                    "summary": {"reason": "no_meeting"},
                    "issues": [],
                },
            },
        }
        plan = qar.build_remediation_plan("20220110", health)
        self.assertFalse(plan["actionable"])

    def test_build_plan_calculated_no_pg_finishers(self):
        health = {
            "checks": {
                "presence": {"status": "fail", "issues": []},
                "raw_content": {"status": "ok", "issues": []},
                "calculated": {
                    "status": "fail",
                    "summary": {"reason": "no_pg_finishers", "races": 36},
                    "issues": [],
                },
            },
        }
        plan = qar.build_remediation_plan("20260711", health)
        kinds = {a["kind"] for a in plan["actions"]}
        self.assertIn("scrape_date", kinds)

    def test_apply_dry_run(self):
        plan = {
            "date": "20260711",
            "actions": [
                {
                    "kind": "scrape_race",
                    "race_id": "202607110101",
                    "tasks": ["race_result"],
                    "reasons": ["raw:gcs_missing"],
                },
            ],
        }
        result = qar.apply_remediation_plan(plan, dry_run=True)
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(result["scrape_queue"]["jobs"], 1)

    def test_auto_remediate_enabled_stg(self):
        with patch.dict("os.environ", {"KEIBA_ENV": "stg", "KEIBA_QUALITY_AUTO_REMEDIATE": ""}, clear=False):
            self.assertTrue(qar.auto_remediate_enabled())
        with patch.dict("os.environ", {"KEIBA_QUALITY_AUTO_REMEDIATE": "0"}, clear=False):
            self.assertFalse(qar.auto_remediate_enabled())


if __name__ == "__main__":
    unittest.main()
