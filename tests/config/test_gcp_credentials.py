"""src.config.gcp_credentials のユニットテスト（.env の GCS_* から構築する方式）。"""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.config.gcp_credentials import (
    build_gcp_credentials,
    gcp_credentials_available,
    gcp_project_id,
    gcp_service_account_info,
)

_FAKE_ENV = {
    "GCS_PROJECT_ID": "masked-project",
    "GCS_PRIVATE_KEY_ID": "0" * 40,
    "GCS_PRIVATE_KEY": "-----BEGIN PRIVATE KEY-----\\nMASKED\\n-----END PRIVATE KEY-----\\n",
    "GCS_CLIENT_EMAIL": "masked@masked-project.iam.gserviceaccount.com",
    "GCS_CLIENT_ID": "0" * 21,
}


class TestGcpCredentials(unittest.TestCase):
    def test_info_is_none_without_private_key(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(gcp_service_account_info())
            self.assertFalse(gcp_credentials_available())
            self.assertIsNone(build_gcp_credentials())

    def test_info_built_from_gcs_env_vars(self):
        with patch.dict(os.environ, _FAKE_ENV, clear=True):
            info = gcp_service_account_info()
            self.assertIsNotNone(info)
            self.assertEqual(info["type"], "service_account")
            self.assertEqual(info["project_id"], "masked-project")
            self.assertEqual(info["client_email"], _FAKE_ENV["GCS_CLIENT_EMAIL"])
            # .env 上のエスケープされた \n が実際の改行に戻る
            self.assertIn("\n", info["private_key"])
            self.assertNotIn("\\n", info["private_key"])
            self.assertEqual(info["token_uri"], "https://oauth2.googleapis.com/token")
            self.assertTrue(gcp_credentials_available())

    def test_build_credentials_delegates_to_google_auth(self):
        with patch.dict(os.environ, _FAKE_ENV, clear=True):
            with patch(
                "src.config.gcp_credentials._service_account.Credentials.from_service_account_info",
                return_value="FAKE_CREDS",
            ) as mock_from_info:
                self.assertEqual(build_gcp_credentials(), "FAKE_CREDS")
                passed_info = mock_from_info.call_args.args[0]
                self.assertEqual(passed_info["project_id"], "masked-project")

    def test_project_id_prefers_gcs_then_gcp(self):
        with patch.dict(os.environ, {"GCP_PROJECT_ID": "gcp-only"}, clear=True):
            self.assertEqual(gcp_project_id(), "gcp-only")
        with patch.dict(
            os.environ,
            {"GCS_PROJECT_ID": "from-gcs", "GCP_PROJECT_ID": "from-gcp"},
            clear=True,
        ):
            self.assertEqual(gcp_project_id(), "from-gcs")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(gcp_project_id(), "")


if __name__ == "__main__":
    unittest.main()
