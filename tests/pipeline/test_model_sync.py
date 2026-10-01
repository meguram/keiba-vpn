"""src.pipeline.models.model_sync のユニットテスト（google.cloud.storage をモック）。

GCS 認証ファイルが無い環境でも実行できるよう、``google.cloud.storage.Client`` と
``HybridStorage._build_credentials`` をモックして検証する。実際の GCS 接続は行わない。
"""

from __future__ import annotations

import os
import time
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.pipeline.models.model_sync import (
    sync_latest_model_from_gcs,
    upload_model_to_gcs,
)


def _mock_client_with_bucket(blob: MagicMock) -> MagicMock:
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = blob
    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    return mock_client


class TestSyncLatestModelFromGcs(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.local_path = Path(self._tmp.name) / "models" / "keiba_model.pkl"

    def test_downloads_when_local_missing(self):
        blob = MagicMock()
        blob.updated = datetime.now(timezone.utc)

        def _fake_download(path):
            Path(path).write_bytes(b"model-bytes")

        blob.download_to_filename.side_effect = _fake_download
        mock_client = _mock_client_with_bucket(blob)

        with patch.dict(os.environ, {"GCS_BUCKET": "test-bucket"}, clear=False):
            with patch("google.cloud.storage.Client", return_value=mock_client):
                with patch(
                    "src.scraper.storage.HybridStorage._build_credentials",
                    return_value=None,
                ):
                    result = sync_latest_model_from_gcs(local_path=self.local_path)

        self.assertTrue(result)
        self.assertTrue(self.local_path.is_file())
        self.assertEqual(self.local_path.read_bytes(), b"model-bytes")
        blob.download_to_filename.assert_called_once()

    def test_downloads_when_gcs_is_newer_than_local(self):
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        self.local_path.write_bytes(b"old-model")
        old_ts = time.time() - 3600
        os.utime(self.local_path, (old_ts, old_ts))

        blob = MagicMock()
        blob.updated = datetime.now(timezone.utc)

        def _fake_download(path):
            Path(path).write_bytes(b"new-model")

        blob.download_to_filename.side_effect = _fake_download
        mock_client = _mock_client_with_bucket(blob)

        with patch.dict(os.environ, {"GCS_BUCKET": "test-bucket"}, clear=False):
            with patch("google.cloud.storage.Client", return_value=mock_client):
                with patch(
                    "src.scraper.storage.HybridStorage._build_credentials",
                    return_value=None,
                ):
                    result = sync_latest_model_from_gcs(local_path=self.local_path)

        self.assertTrue(result)
        self.assertEqual(self.local_path.read_bytes(), b"new-model")

    def test_skips_when_local_is_already_up_to_date(self):
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        self.local_path.write_bytes(b"current-model")

        blob = MagicMock()
        blob.updated = datetime.now(timezone.utc) - timedelta(days=1)
        mock_client = _mock_client_with_bucket(blob)

        with patch.dict(os.environ, {"GCS_BUCKET": "test-bucket"}, clear=False):
            with patch("google.cloud.storage.Client", return_value=mock_client):
                with patch(
                    "src.scraper.storage.HybridStorage._build_credentials",
                    return_value=None,
                ):
                    result = sync_latest_model_from_gcs(local_path=self.local_path)

        self.assertFalse(result)
        blob.download_to_filename.assert_not_called()
        self.assertEqual(self.local_path.read_bytes(), b"current-model")

    def test_returns_false_when_blob_missing_on_gcs(self):
        class _NotFound(Exception):
            pass

        blob = MagicMock()
        blob.reload.side_effect = _NotFound("404 Not Found")
        mock_client = _mock_client_with_bucket(blob)

        with patch.dict(os.environ, {"GCS_BUCKET": "test-bucket"}, clear=False):
            with patch("google.cloud.storage.Client", return_value=mock_client):
                with patch(
                    "src.scraper.storage.HybridStorage._build_credentials",
                    return_value=None,
                ):
                    result = sync_latest_model_from_gcs(local_path=self.local_path)

        self.assertFalse(result)
        self.assertFalse(self.local_path.exists())

    def test_returns_false_without_exception_when_gcs_bucket_unset(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GCS_BUCKET", None)
            result = sync_latest_model_from_gcs(local_path=self.local_path)

        self.assertFalse(result)

    def test_returns_false_without_exception_on_client_error(self):
        with patch.dict(os.environ, {"GCS_BUCKET": "test-bucket"}, clear=False):
            with patch(
                "google.cloud.storage.Client",
                side_effect=RuntimeError("network down"),
            ):
                with patch(
                    "src.scraper.storage.HybridStorage._build_credentials",
                    return_value=None,
                ):
                    result = sync_latest_model_from_gcs(local_path=self.local_path)

        self.assertFalse(result)


class TestUploadModelToGcs(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.local_path = Path(self._tmp.name) / "models" / "keiba_model.pkl"

    def test_uploads_when_local_file_exists_and_gcs_enabled(self):
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        self.local_path.write_bytes(b"trained-model")

        blob = MagicMock()
        mock_client = _mock_client_with_bucket(blob)

        with patch.dict(os.environ, {"GCS_BUCKET": "test-bucket"}, clear=False):
            with patch("google.cloud.storage.Client", return_value=mock_client):
                with patch(
                    "src.scraper.storage.HybridStorage._build_credentials",
                    return_value=None,
                ):
                    result = upload_model_to_gcs(local_path=self.local_path)

        self.assertTrue(result)
        blob.upload_from_filename.assert_called_once_with(str(self.local_path))

    def test_returns_false_when_local_file_missing(self):
        with patch.dict(os.environ, {"GCS_BUCKET": "test-bucket"}, clear=False):
            result = upload_model_to_gcs(local_path=self.local_path)

        self.assertFalse(result)

    def test_returns_false_without_exception_when_gcs_bucket_unset(self):
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        self.local_path.write_bytes(b"trained-model")

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GCS_BUCKET", None)
            result = upload_model_to_gcs(local_path=self.local_path)

        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
