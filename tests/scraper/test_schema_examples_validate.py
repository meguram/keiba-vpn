"""`src.scraper.schemas` が fixtures と整合すること（構造・型の最低限）。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.scraper import schemas

_FIX_DIR = Path(__file__).resolve().parent / "fixtures" / "schema_examples"


class TestSchemaExamplesValidate(unittest.TestCase):
    def _load(self, name: str) -> dict:
        p = _FIX_DIR / f"{name}.json"
        self.assertTrue(p.is_file(), f"missing {p}")
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        self.assertIsInstance(d, dict)
        return d

    def test_all_examples_pass_schemas_validate(self) -> None:
        for stem in sorted(p.stem for p in _FIX_DIR.glob("*.json")):
            with self.subTest(category=stem):
                data = self._load(stem)
                r = schemas.validate(stem, data)
                self.assertTrue(
                    r.get("passed"),
                    f"{stem}: {r}",
                )

    def test_schema_version_bumped(self) -> None:
        self.assertGreaterEqual(schemas.SCHEMA_VERSION, 2)

    def test_race_shutuba_fixture_matches_jsonschema_file(self) -> None:
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not installed")
        schema_path = (
            Path(__file__).resolve().parents[2]
            / "docs"
            / "requirements"
            / "data"
            / "schemas"
            / "json"
            / "race_shutuba.schema.json"
        )
        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)
        inst = self._load("race_shutuba")
        jsonschema.validate(instance=inst, schema=schema)


if __name__ == "__main__":
    unittest.main()
