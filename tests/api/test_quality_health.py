"""quality_health ユニットテスト。"""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from src.api import quality_health as qh


class TestQualityHealth(unittest.TestCase):
    def test_compute_overall_worst(self):
        checks = {
            "presence": {"status": "ok"},
            "raw_content": {"status": "fail"},
        }
        self.assertEqual(qh.compute_overall_status(checks), "fail")

    def test_stale_marks_warn_display(self):
        old = (datetime.now(timezone(timedelta(hours=9))) - timedelta(days=10)).isoformat()
        view = qh.enrich_check_record({"status": "ok", "checked_at": old})
        self.assertTrue(view["stale"])
        self.assertEqual(view["display_status"], "warn")

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            qh.QUALITY_HEALTH_DIR = Path(tmp)
            qh.save_health("20260101", {
                "checks": {
                    "presence": {"status": "ok", "checked_at": "2026-01-01T00:00:00+09:00"},
                },
            })
            loaded = qh.load_health("20260101")
            assert loaded is not None
            self.assertEqual(loaded["overall_status"], "ok")


if __name__ == "__main__":
    unittest.main()
