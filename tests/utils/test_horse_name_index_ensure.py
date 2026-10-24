"""ensure_horse_name_index のローカルキャッシュ経由のテスト。"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from src.utils.horse_name_index import (
    canonical_horse_name_index_path,
    ensure_horse_name_index,
    invalidate_horse_name_index_cache,
)


def _empty_index_paths(base: Path) -> tuple[Path, ...]:
    calc = base / "data" / "calculated_data" / "knowledge" / "horse_name_index.json"
    legacy = base / "data" / "knowledge" / "horse_name_index.json"
    return (calc, legacy)


class TestEnsureHorseNameIndex(unittest.TestCase):
    def tearDown(self) -> None:
        invalidate_horse_name_index_cache()
        os.environ.pop("KEIBA_CALCULATED_DATA_DIR", None)

    def test_ensure_builds_from_local_cache(self) -> None:
        import tempfile

        base = Path(tempfile.mkdtemp(prefix="hnensure_"))
        calc_root = base / "data" / "calculated_data"
        os.environ["KEIBA_CALCULATED_DATA_DIR"] = str(calc_root)
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
        with patch(
            "src.utils.horse_name_index.horse_name_index_candidate_paths",
            side_effect=lambda _b: _empty_index_paths(base),
        ):
            r = ensure_horse_name_index(base, min_horses=1)
        self.assertIn(r["status"], ("rebuilt_local_cache", "skipped"))
        self.assertGreaterEqual(r.get("total_horses") or 0, 1)
        p = canonical_horse_name_index_path(base)
        self.assertTrue(p.is_file())


if __name__ == "__main__":
    unittest.main()
