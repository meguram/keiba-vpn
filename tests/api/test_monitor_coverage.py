"""monitor_coverage ユニットテスト。"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.api import monitor_coverage as mc


class TestMonitorCoverage(unittest.TestCase):
    def test_race_meta_from_race_id(self):
        meta = mc._race_meta(["202606030101"])
        self.assertEqual(meta["202606030101"]["venue"], "中山")
        self.assertEqual(meta["202606030101"]["race_num"], 1)

    def test_aggregation_mode_prod_cached(self):
        with patch.dict(os.environ, {"KEIBA_ENV": "prod"}):
            self.assertEqual(mc.aggregation_mode(), "cached")
        with patch.dict(os.environ, {"KEIBA_ENV": "stg"}):
            self.assertEqual(mc.aggregation_mode(), "realtime")

    def test_dev_raw_matrix_requirement_all_true(self):
        storage = MagicMock()
        storage._local_path.return_value = Path("/nonexistent/a.json")
        storage._legacy_local_path.return_value = Path("/nonexistent/b.json")

        with patch.object(mc, "_load_race_ids_for_date", return_value=["202606030101"]):
            with patch.object(mc, "_local_file_exists", return_value=False):
                out = mc.build_raw_matrix("20260603", view="dev", storage=storage)

        self.assertEqual(out["view"], "dev")
        req = out["requirement"]["matrix"]["202606030101"]
        self.assertTrue(all(req.values()))
        self.assertFalse(out["local_index"]["matrix"]["202606030101"]["race_shutuba"])

    def test_db_coverage_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            mc.DB_COVERAGE_DIR = Path(tmp)
            payload = {
                "date": "20260101",
                "raw": {"matrix": {"202601010101": {"pg_races": True}}},
                "calculated": {"matrix": {"202601010101": {"megu_coverage_ok": True}}},
            }
            mc.save_db_coverage_cache("20260101", payload)
            loaded = mc.load_db_coverage_cache("20260101")
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertTrue(loaded["raw"]["matrix"]["202601010101"]["pg_races"])


if __name__ == "__main__":
    unittest.main()
