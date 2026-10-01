"""load_project_dotenv の環境別オーバーレイ（.env=dev / .env.stg / .env.prod）のテスト。"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.utils.project_env import load_project_dotenv


class TestEnvOverlay(unittest.TestCase):
    def _write_env_files(self, root: Path) -> None:
        (root / ".env").write_text(
            "GCS_PROJECT_ID=dev-project\nGCS_BUCKET=dev-bucket\nONLY_IN_BASE=1\n",
            encoding="utf-8",
        )
        (root / ".env.stg").write_text("GCS_PROJECT_ID=stg-project\n", encoding="utf-8")
        (root / ".env.prod").write_text(
            "GCS_PROJECT_ID=prod-project\nGCS_BUCKET=prod-bucket\n", encoding="utf-8"
        )

    def _load(self, keiba_env: str | None) -> dict[str, str]:
        env = {} if keiba_env is None else {"KEIBA_ENV": keiba_env}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_env_files(root)
            with patch.dict(os.environ, env, clear=True):
                load_project_dotenv(dotenv_path=root / ".env")
                return {
                    k: os.environ.get(k, "")
                    for k in ("GCS_PROJECT_ID", "GCS_BUCKET", "ONLY_IN_BASE")
                }

    def test_dev_uses_only_base_env(self):
        got = self._load(None)
        self.assertEqual(got["GCS_PROJECT_ID"], "dev-project")
        self.assertEqual(got["GCS_BUCKET"], "dev-bucket")

    def test_stg_overlays_only_differences(self):
        got = self._load("stg")
        self.assertEqual(got["GCS_PROJECT_ID"], "stg-project")
        self.assertEqual(got["GCS_BUCKET"], "dev-bucket")  # .env.stg に無い値はベースを継承
        self.assertEqual(got["ONLY_IN_BASE"], "1")

    def test_prod_overrides_project_and_bucket(self):
        got = self._load("prod")
        self.assertEqual(got["GCS_PROJECT_ID"], "prod-project")
        self.assertEqual(got["GCS_BUCKET"], "prod-bucket")


if __name__ == "__main__":
    unittest.main()
