"""MLflow プラットフォーム基盤のユニットテスト。"""

from __future__ import annotations

import os
import unittest

from src.pipeline.mlflow.catalog import (
    MODEL_CATALOG,
    ModelLifecycle,
    get_model_spec,
    list_model_keys,
)
from src.pipeline.mlflow.runtime import (
    model_settings,
    platform_health,
    resolve_serve_uri,
)


class TestModelCatalog(unittest.TestCase):
    def test_all_specs_have_required_fields(self):
        for key, spec in MODEL_CATALOG.items():
            self.assertEqual(spec.key, key)
            self.assertTrue(spec.registered_name)
            self.assertTrue(spec.experiment_name)

    def test_planned_models_exist(self):
        planned = list_model_keys(lifecycle=ModelLifecycle.PLANNED)
        self.assertIn("finish_order", planned)
        self.assertIn("final_odds", planned)

    def test_get_model_spec_unknown_raises(self):
        with self.assertRaises(KeyError):
            get_model_spec("nonexistent_model")


class TestServeUriResolution(unittest.TestCase):
    def test_legacy_env_tracking_difficulty(self):
        env = "KEIBA_MLFLOW_SERVE_TRACKING_URI"
        old = os.environ.get(env)
        try:
            os.environ[env] = "http://test-tracking:5999"
            self.assertEqual(
                resolve_serve_uri("tracking_difficulty"),
                "http://test-tracking:5999",
            )
        finally:
            if old is None:
                os.environ.pop(env, None)
            else:
                os.environ[env] = old

    def test_default_port_from_catalog(self):
        env_keys = [
            "KEIBA_MLFLOW_SERVE_TRACKING_DIFFICULTY_URI",
            "KEIBA_MLFLOW_SERVE_TRACKING_URI",
        ]
        saved = {k: os.environ.pop(k, None) for k in env_keys}
        try:
            ms = model_settings("finish_order")
            if not ms.get("serve_uri"):
                uri = resolve_serve_uri("finish_order")
                self.assertEqual(uri, "http://localhost:5002")
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v


class TestPlatformHealth(unittest.TestCase):
    def test_platform_health_structure(self):
        report = platform_health(force_serve_check=False)
        self.assertIn("mlflow_tracking", report)
        self.assertIn("models", report)
        self.assertEqual(len(report["models"]), len(MODEL_CATALOG))
        keys = {m["key"] for m in report["models"]}
        self.assertEqual(keys, set(MODEL_CATALOG.keys()))


if __name__ == "__main__":
    unittest.main()
