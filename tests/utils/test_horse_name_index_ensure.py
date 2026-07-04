"""ensure_horse_name_index のローカルキャッシュ経由のテスト。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.utils.horse_name_index import (
    ensure_horse_name_index,
    invalidate_horse_name_index_cache,
)


class TestEnsureHorseNameIndex(unittest.TestCase):
    def tearDown(self) -> None:
        invalidate_horse_name_index_cache()

    def test_ensure_builds_from_local_cache(self) -> None:
        import tempfile

        base = Path(tempfile.mkdtemp(prefix="hnensure_"))
        cache = base / "data" / "cache" / "horse_result" / "2099"
        cache.mkdir(parents=True)
        (cache / "2099990001.json").write_text(
            json.dumps(
                {
                    "horse_id": "2099990001",
                    "horse_name": "テスト馬インデックス",
                    "name_en": "",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        invalidate_horse_name_index_cache()
        r = ensure_horse_name_index(base, min_horses=1)
        self.assertIn(r["status"], ("rebuilt_local_cache", "skipped"))
        self.assertGreaterEqual(r.get("total_horses") or 0, 1)
        p = base / "data" / "knowledge" / "horse_name_index.json"
        self.assertTrue(p.is_file())


if __name__ == "__main__":
    unittest.main()
