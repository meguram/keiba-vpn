"""HybridStorage.save とスキーマ検証（厳格／緩和）の結合テスト。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.scraper.schemas import SchemaValidationError
from src.scraper.storage import HybridStorage


class TestHybridStorageSchema(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.base = Path(self._dir.name)

    def test_strict_raises_before_gcs_on_invalid_shutuba(self) -> None:
        bad: dict = {"entries": []}
        with mock.patch.dict(os.environ, {"KEIBA_SCHEMA_STRICT": "1", "GCS_BUCKET": ""}):
            st = HybridStorage(base_dir=str(self.base), bucket_name="")
            with self.assertRaises(SchemaValidationError):
                st.save("race_shutuba", "202401010101", bad)

    def test_lenient_does_not_raise_sets_meta(self) -> None:
        bad: dict = {"entries": []}
        with mock.patch.dict(os.environ, {"KEIBA_SCHEMA_STRICT": "0", "GCS_BUCKET": ""}):
            st = HybridStorage(base_dir=str(self.base), bucket_name="")
            out = st.save("race_shutuba", "202401010101", bad)
        self.assertFalse(out)
        self.assertEqual(bad["_meta"].get("scrape_validation_status"), "schema_failed")
        self.assertIn("schema_validation", bad["_meta"])
