"""model_registry（モデルの公開・取得・検証）のテスト。GCS には接続しない。"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.pipeline.models import model_registry as mr
from src.pipeline.models.model_registry import (
    GcsModelStore,
    LocalModelStore,
    ModelIncompatibleError,
    ModelIntegrityError,
    fetch_latest,
    fetch_version,
    open_store,
    publish_model,
    set_latest_version,
)
from src.scripts.maintenance.make_pseudo_ensemble import train_pseudo_ensemble

NAMES_N = 12


class ModelRegistryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.model_dir = cls.root / "trained"
        cls.names = train_pseudo_ensemble(cls.model_dir, NAMES_N, n_rows=300)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        self.store = LocalModelStore(tempfile.mkdtemp(dir=self.root))
        self.cache = Path(tempfile.mkdtemp(dir=self.root))

    def test_publish_then_fetch_latest_roundtrip(self):
        m = publish_model(self.model_dir, "v1", self.store, feature_names=self.names)
        self.assertEqual(m["n_features"], NAMES_N)
        self.assertIn("scikit-learn", m["libraries"])
        got = fetch_latest(self.store, self.cache)
        self.assertEqual(got.name, "v1")
        self.assertTrue((got / "manifest.json").is_file())

    def test_fetch_latest_returns_none_when_nothing_published(self):
        self.assertIsNone(fetch_latest(self.store, self.cache))

    def test_second_fetch_does_not_download_again(self):
        publish_model(self.model_dir, "v1", self.store, feature_names=self.names)
        fetch_latest(self.store, self.cache)
        with patch.object(LocalModelStore, "get_file", side_effect=AssertionError("再ダウンロードした")):
            fetch_latest(self.store, self.cache)

    def test_tampered_file_in_store_is_rejected(self):
        publish_model(self.model_dir, "v1", self.store, feature_names=self.names)
        (self.store.root / "v1" / "lightgbm_model.pkl").write_bytes(b"tampered")
        with self.assertRaises(ModelIntegrityError):
            fetch_latest(self.store, self.cache)
        self.assertFalse((self.cache / "v1").exists())      # 検証に失敗した版は残さない

    def test_locally_corrupted_cache_is_refetched(self):
        publish_model(self.model_dir, "v1", self.store, feature_names=self.names)
        got = fetch_latest(self.store, self.cache)
        (got / "meta_model.pkl").write_bytes(b"broken")
        again = fetch_latest(self.store, self.cache)
        mm = json.loads((again / "manifest.json").read_text())
        self.assertEqual(mr._sha256(again / "meta_model.pkl"), mm["files"]["meta_model.pkl"]["sha256"])

    def test_library_version_mismatch_is_rejected_when_strict(self):
        publish_model(self.model_dir, "v1", self.store, feature_names=self.names)
        manifest = self.store.read_json("v1/manifest.json")
        manifest["libraries"]["scikit-learn"] = "0.1.0"
        self.store.write_json("v1/manifest.json", manifest)
        with self.assertRaises(ModelIncompatibleError):
            fetch_latest(self.store, self.cache)
        # strict でなければ警告のみで取得できる
        self.assertIsNotNone(fetch_latest(self.store, self.cache, strict_versions=False))

    def test_rollback_by_switching_latest(self):
        publish_model(self.model_dir, "v1", self.store, feature_names=self.names)
        publish_model(self.model_dir, "v2", self.store, feature_names=self.names)
        self.assertEqual(fetch_latest(self.store, self.cache).name, "v2")
        set_latest_version(self.store, "v1")
        self.assertEqual(fetch_latest(self.store, self.cache).name, "v1")

    def test_cannot_point_latest_to_unpublished_version(self):
        with self.assertRaises(FileNotFoundError):
            set_latest_version(self.store, "nope")

    def test_publish_without_feature_names_uses_ensemble_meta(self):
        m = publish_model(self.model_dir, "v1", self.store)
        self.assertEqual(m["feature_names"], self.names)

    def test_open_store_dispatch(self):
        self.assertIsInstance(open_store("/tmp/some/path"), LocalModelStore)
        with patch.object(mr.GcsModelStore, "__init__", return_value=None) as init:
            self.assertIsInstance(open_store("gs://bkt/models/ensemble"), GcsModelStore)
            init.assert_called_once_with("bkt", "models/ensemble")


class GcsModelStoreTest(unittest.TestCase):
    def test_keys_use_prefix_and_json_roundtrip(self):
        bucket = MagicMock()
        blob = bucket.blob.return_value
        blob.exists.return_value = True
        blob.download_as_text.return_value = '{"version": "v9"}'
        store = GcsModelStore("bkt", "models/ensemble/", bucket=bucket)

        self.assertEqual(store.read_json("latest.json"), {"version": "v9"})
        bucket.blob.assert_called_with("models/ensemble/latest.json")

        store.write_json("v9/manifest.json", {"a": 1})
        bucket.blob.assert_called_with("models/ensemble/v9/manifest.json")
        blob.upload_from_string.assert_called_once()

    def test_read_json_returns_none_when_blob_missing(self):
        bucket = MagicMock()
        bucket.blob.return_value.exists.return_value = False
        self.assertIsNone(GcsModelStore("bkt", "", bucket=bucket).read_json("latest.json"))


if __name__ == "__main__":
    unittest.main()
