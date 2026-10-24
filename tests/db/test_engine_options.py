import os
import unittest
from unittest.mock import patch

from src.db.session import engine_options


class EngineOptionsTest(unittest.TestCase):
    def test_postgres_gets_timeouts(self):
        with patch.dict(os.environ, {}, clear=True):
            o = engine_options("postgresql+psycopg://u:p@host/db")
        self.assertTrue(o["pool_pre_ping"])
        self.assertEqual(o["pool_timeout"], 10)
        self.assertEqual(o["connect_args"], {"connect_timeout": 5})

    def test_env_overrides_and_bad_values_fall_back(self):
        with patch.dict(os.environ, {"DB_POOL_TIMEOUT_SEC": "3", "DB_CONNECT_TIMEOUT_SEC": "abc"}, clear=True):
            o = engine_options("postgresql://u:p@host/db")
        self.assertEqual(o["pool_timeout"], 3)
        self.assertEqual(o["connect_args"], {"connect_timeout": 5})

    def test_sqlite_is_untouched(self):
        self.assertEqual(engine_options("sqlite:///:memory:"), {"pool_pre_ping": True})


if __name__ == "__main__":
    unittest.main()
