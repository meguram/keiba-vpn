"""FastAPI 旧ルート（/api/...）の書き込み系が開発者限定になることの回帰テスト（T-010）。"""
import json
import subprocess
import sys
import unittest
from pathlib import Path

from src.api.auth import requires_auth

ROOT = Path(__file__).resolve().parents[2]
WRITE = {"POST", "PUT", "PATCH", "DELETE"}


class LegacyWriteAuthTest(unittest.TestCase):
    def test_legacy_write_routes_require_developer(self):
        for method, path in [
            ("POST", "/api/train"),
            ("POST", "/api/scrape-queue/kick"),
            ("POST", "/api/scrape-queue/add"),
            ("POST", "/api/admin/git-pull"),
            ("POST", "/api/backfill"),
            ("POST", "/api/auto-scrape/start"),
            ("DELETE", "/api/scrape-queue/jobs/1"),
        ]:
            with self.subTest(method=method, path=path):
                self.assertTrue(requires_auth(path, method))

    def test_legacy_reads_stay_open_for_monitor(self):
        for path in ["/api/health", "/api/scrape-jobs", "/api/coverage-calendar", "/api/date-race-matrix"]:
            with self.subTest(path=path):
                self.assertFalse(requires_auth(path, "GET"))

    def test_v1_dev_only_still_blocked_for_any_method(self):
        self.assertTrue(requires_auth("/api/v1/scrape-queue/kick", "GET"))
        self.assertTrue(requires_auth("/api/v1/admin/x", "POST"))

    def test_public_v1_and_unlisted_paths_unchanged(self):
        self.assertFalse(requires_auth("/api/v1/races/202606010101", "GET"))
        self.assertFalse(requires_auth("/api/internal/cloud-tasks/process-job", "POST"))

    def test_default_method_is_read(self):
        self.assertFalse(requires_auth("/api/train"))

    def test_no_dev_only_write_route_is_left_open(self):
        out = subprocess.check_output(
            [sys.executable, str(ROOT / ".claude/skills/evaluate-keiba-architecture/scripts/collect_endpoints.py")],
            text=True,
        )
        routes = next(l for l in json.loads(out) if l["layer"] == "fastapi_legacy")["routes"]
        from src.api.auth import is_dev_only_path, is_public_path

        leaked = []
        for r in routes:
            if not set(r["method"]) & WRITE or not r["path"].startswith("/api/") or r["path"].startswith("/api/v1/"):
                continue
            v1 = "/api/v1/" + r["path"][len("/api/"):]
            if is_dev_only_path(v1) and not is_public_path(v1) and not requires_auth(r["path"], sorted(set(r["method"]) & WRITE)[0]):
                leaked.append(r["path"])
        self.assertEqual(leaked, [])


if __name__ == "__main__":
    unittest.main()


class SecretKeyTest(unittest.TestCase):
    """DEV_SECRET_KEY の既定値フォールバックは dev のみ（T-010）。"""

    def test_prod_and_stg_require_secret(self):
        from unittest.mock import patch

        from src.api import auth

        for env in ("stg", "staging", "prod", "production", "PROD"):
            with self.subTest(env=env), patch.dict("os.environ", {"KEIBA_ENV": env}, clear=True):
                with self.assertRaises(RuntimeError):
                    auth._get_secret_key()

    def test_explicit_secret_wins_in_any_env(self):
        from unittest.mock import patch

        from src.api import auth

        with patch.dict("os.environ", {"KEIBA_ENV": "prod", "DEV_SECRET_KEY": "x" * 32}, clear=True):
            self.assertEqual(auth._get_secret_key(), "x" * 32)

    def test_dev_falls_back_with_default(self):
        from unittest.mock import patch

        from src.api import auth

        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(auth._get_secret_key(), auth._INSECURE_DEV_SECRET)
