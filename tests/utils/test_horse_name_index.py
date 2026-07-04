"""src.utils.horse_name_index のユニットテスト。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.utils.horse_name_index import (
    canonical_horse_name_index_path,
    invalidate_horse_name_index_cache,
    load_horse_name_index,
    upsert_horse_name_index_entry,
)


class TestHorseNameIndex(unittest.TestCase):
    def setUp(self) -> None:
        invalidate_horse_name_index_cache()

    def tearDown(self) -> None:
        invalidate_horse_name_index_cache()

    def test_upsert_writes_canonical_and_load_roundtrip(self) -> None:
        base = Path(self._temp())
        kn = base / "data" / "knowledge"
        kn.mkdir(parents=True, exist_ok=True)
        upsert_horse_name_index_entry(base, "2099990001", "テスト馬アルファ", "Test Alpha")
        p = canonical_horse_name_index_path(base)
        self.assertTrue(p.is_file())
        data = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(data["total_horses"], 1)
        self.assertEqual(data["horses"][0]["horse_id"], "2099990001")
        invalidate_horse_name_index_cache()
        loaded = load_horse_name_index(base)
        self.assertEqual(len(loaded.get("horses") or []), 1)

    def test_prefers_newer_file_between_knowledge_and_local(self) -> None:
        base = Path(self._temp())
        old = base / "data" / "local" / "knowledge"
        new = base / "data" / "knowledge"
        old.mkdir(parents=True, exist_ok=True)
        new.mkdir(parents=True, exist_ok=True)
        old_file = old / "horse_name_index.json"
        new_file = new / "horse_name_index.json"
        old_file.write_text(
            json.dumps({"version": "1.0", "total_horses": 1, "horses": [
                {"horse_id": "1", "horse_name": "古い", "name_en": "", "year": "2020"},
            ]}, ensure_ascii=False),
            encoding="utf-8",
        )
        import time

        time.sleep(0.05)
        new_file.write_text(
            json.dumps({"version": "1.0", "total_horses": 1, "horses": [
                {"horse_id": "2", "horse_name": "新しい", "name_en": "", "year": "2026"},
            ]}, ensure_ascii=False),
            encoding="utf-8",
        )
        invalidate_horse_name_index_cache()
        loaded = load_horse_name_index(base)
        names = {h["horse_name"] for h in loaded.get("horses") or []}
        self.assertIn("新しい", names)
        self.assertNotIn("古い", names)

    @staticmethod
    def _temp() -> str:
        import tempfile

        return tempfile.mkdtemp(prefix="hnidx_")


if __name__ == "__main__":
    unittest.main()
