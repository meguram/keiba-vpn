"""src.config.gcp_credentials のユニットテスト。"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config.gcp_credentials import (
    ensure_google_application_credentials,
    gcp_credentials_available,
)


class TestGcpCredentials(unittest.TestCase):
    def test_returns_false_when_no_env_and_no_default_file(self):
        with patch.dict(os.environ, {}, clear=True):
            with tempfile.TemporaryDirectory() as tmp:
                self.assertFalse(ensure_google_application_credentials(tmp))
                self.assertFalse(gcp_credentials_available())

    def test_sets_env_from_default_path_when_file_exists(self):
        with patch.dict(os.environ, {}, clear=True):
            with tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                (base / "config").mkdir()
                cred_file = base / "config" / "gcp-service-account.json"
                cred_file.write_text("{}", encoding="utf-8")

                self.assertTrue(ensure_google_application_credentials(base))
                self.assertEqual(
                    os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
                    str(cred_file.resolve()),
                )
                self.assertTrue(gcp_credentials_available())

    def test_existing_env_var_takes_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            real_cred = base / "real-creds.json"
            real_cred.write_text("{}", encoding="utf-8")
            with patch.dict(
                os.environ,
                {"GOOGLE_APPLICATION_CREDENTIALS": str(real_cred)},
                clear=True,
            ):
                self.assertTrue(ensure_google_application_credentials(base))
                self.assertEqual(
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"], str(real_cred)
                )

    def test_existing_env_var_pointing_to_missing_file_returns_false(self):
        with patch.dict(
            os.environ,
            {"GOOGLE_APPLICATION_CREDENTIALS": "/nonexistent/path.json"},
            clear=True,
        ):
            self.assertFalse(ensure_google_application_credentials())
            self.assertFalse(gcp_credentials_available())

    def test_env_specific_file_takes_precedence_over_generic(self):
        with patch.dict(os.environ, {"KEIBA_ENV": "stg"}, clear=True):
            with tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                (base / "config").mkdir()
                (base / "config" / "gcp-service-account.json").write_text(
                    "{}", encoding="utf-8"
                )
                stg_cred = base / "config" / "gcp-service-account.stg.json"
                stg_cred.write_text("{}", encoding="utf-8")

                self.assertTrue(ensure_google_application_credentials(base))
                self.assertEqual(
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
                    str(stg_cred.resolve()),
                )

    def test_falls_back_to_generic_file_when_env_specific_missing(self):
        with patch.dict(os.environ, {"KEIBA_ENV": "prod"}, clear=True):
            with tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                (base / "config").mkdir()
                generic_cred = base / "config" / "gcp-service-account.json"
                generic_cred.write_text("{}", encoding="utf-8")

                self.assertTrue(ensure_google_application_credentials(base))
                self.assertEqual(
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
                    str(generic_cred.resolve()),
                )


if __name__ == "__main__":
    unittest.main()
