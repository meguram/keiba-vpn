"""HybridStorage local_only paths use page_reference."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.scraper.storage import HybridStorage


class TestPageReferenceStorage(unittest.TestCase):
    def test_race_lists_save_load_under_page_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "data" / "page_reference").mkdir(parents=True)
            st = HybridStorage(base_dir=str(base))
            st.save("race_lists", "20990101", {"races": [], "date": "20990101"})
            path = base / "data" / "page_reference" / "race_lists" / "20990101.json"
            self.assertTrue(path.is_file())
            loaded = st.load("race_lists", "20990101")
            self.assertEqual(loaded.get("date"), "20990101")

    def test_race_lists_legacy_fallback_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            legacy = base / "data" / "local" / "race_lists"
            legacy.mkdir(parents=True)
            (legacy / "20990102.json").write_text(
                json.dumps({"date": "20990102"}), encoding="utf-8"
            )
            st = HybridStorage(base_dir=str(base))
            loaded = st.load("race_lists", "20990102")
            self.assertEqual(loaded.get("date"), "20990102")


if __name__ == "__main__":
    unittest.main()
