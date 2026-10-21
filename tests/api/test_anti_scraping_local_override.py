"""config/anti_scraping.local.json があれば base の設定を上書きする（ホスト固有の値をリポジトリに載せないため）。"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.utils import anti_scraping as A


class LocalOverrideTest(unittest.TestCase):
    def test_local_json_overrides_base(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "anti_scraping.json"
            base.write_text(json.dumps({"edge_sources": []}), encoding="utf-8")
            guard = A.AntiScraping(config_path=base) if hasattr(A, "AntiScraping") else None
            self.assertIsNotNone(guard)
            self.assertEqual(guard.config().get("edge_sources"), [])
            (Path(d) / "anti_scraping.local.json").write_text(json.dumps({"edge_sources": ["203.0.113.9"]}), encoding="utf-8")
            guard2 = A.AntiScraping(config_path=base)
            self.assertEqual(guard2.config().get("edge_sources"), ["203.0.113.9"])

    def test_no_local_file_is_fine(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "anti_scraping.json"
            base.write_text(json.dumps({}), encoding="utf-8")
            self.assertEqual(A.AntiScraping(config_path=base).config().get("edge_sources"), [])


if __name__ == "__main__":
    unittest.main()
