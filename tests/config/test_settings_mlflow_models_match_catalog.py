"""config/settings.yaml の mlflow.models が MODEL_CATALOG と一致していること（T-060）。"""
import unittest
from pathlib import Path

import yaml

from src.pipeline.mlflow.catalog import MODEL_CATALOG

SETTINGS = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


class SettingsMatchCatalogTest(unittest.TestCase):
    def setUp(self):
        self.models = yaml.safe_load(SETTINGS.read_text(encoding="utf-8"))["mlflow"]["models"]

    def test_every_catalog_model_is_configured(self):
        self.assertEqual(sorted(self.models), sorted(MODEL_CATALOG))

    def test_ports_match_catalog_defaults(self):
        for key, spec in MODEL_CATALOG.items():
            with self.subTest(model=key):
                cfg = self.models[key]
                self.assertEqual(cfg["serve_port"], spec.default_serve_port)
                self.assertTrue(cfg["serve_uri"].endswith(f":{spec.default_serve_port}"))


if __name__ == "__main__":
    unittest.main()
