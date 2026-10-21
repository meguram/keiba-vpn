"""外に出す応答（/api/health など）に保存先（GCS のバケット名）を含めないこと。"""
from __future__ import annotations

import os
import unittest
from unittest import mock

from src.config.deployment import deployment_info


class DeploymentInfoTest(unittest.TestCase):
    def test_public_info_has_no_storage(self):
        with mock.patch.dict(os.environ, {"GCS_BUCKET": "secret-bucket", "GCS_PREFIX": "pfx", "KEIBA_ENV": "stg"}):
            pub = deployment_info(include_storage=False)
            self.assertNotIn("gcs_bucket", pub)
            self.assertNotIn("gcs_prefix", pub)
            self.assertNotIn("secret-bucket", str(pub))
            self.assertEqual(pub["keiba_env"], "stg")

    def test_internal_info_keeps_storage(self):
        with mock.patch.dict(os.environ, {"GCS_BUCKET": "b", "GCS_PREFIX": "p"}):
            full = deployment_info()
            self.assertEqual(full["gcs_bucket"], "b")
            self.assertEqual(full["gcs_prefix"], "p")


if __name__ == "__main__":
    unittest.main()
