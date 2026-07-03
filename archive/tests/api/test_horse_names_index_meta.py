"""馬名インデックス meta API・検索・成長曲線の軽い結合テスト。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.utils.horse_name_index import invalidate_horse_name_index_cache

# test_endpoints と同様（GCS オフ・認証オフ）
import os

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("GCS_BUCKET", "")
os.environ.setdefault("HORSE_NAME_INDEX_DISABLE_BOOTSTRAP", "1")

from src.api import app as app_mod

client = TestClient(app_mod.app, raise_server_exceptions=False)


class TestHorseNamesIndexMeta(unittest.TestCase):
    def tearDown(self) -> None:
        invalidate_horse_name_index_cache()

    def test_index_meta_returns_core_fields(self) -> None:
        r = client.get("/api/horse-names/index-meta")
        self.assertEqual(r.status_code, 200, r.text)
        j = r.json()
        for k in ("canonical_path", "total_horses", "resolved_path", "last_ensure", "storage"):
            self.assertIn(k, j, msg=j)

    def test_search_finds_horse_with_patched_base_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            kn = Path(tmp) / "data" / "knowledge"
            kn.mkdir(parents=True)
            (kn / "horse_name_index.json").write_text(
                json.dumps(
                    {
                        "version": "1.0",
                        "total_horses": 1,
                        "generated_at": "2099-01-01 00:00:00",
                        "horses": [
                            {
                                "horse_id": "2019105551",
                                "horse_name": "イクイノックス",
                                "name_en": "Equinox",
                                "year": "2019",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            invalidate_horse_name_index_cache()
            with patch.object(app_mod, "BASE_DIR", tmp):
                invalidate_horse_name_index_cache()
                r = client.get("/api/horse-names/search?q=い&limit=10")
                self.assertEqual(r.status_code, 200, r.text)
                ids = {x.get("horse_id") for x in (r.json().get("results") or [])}
                self.assertIn("2019105551", ids)

    def test_growth_curve_json_shape(self) -> None:
        r = client.get("/api/growth-curve/2019105551")
        self.assertIn(r.status_code, (200, 404), r.text)
        if r.status_code == 404:
            self.assertIn("error", r.json())
            return
        j = r.json()
        self.assertIn("horse_name", j)
        self.assertIn("races", j)
        self.assertIsInstance(j["races"], list)


if __name__ == "__main__":
    unittest.main()
