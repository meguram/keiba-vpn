"""src.db.cloud_sql / src.db.session の KEIBA_DB_BACKEND 分岐のユニットテスト。

実際のCloud SQLインスタンスには接続しない。``google.cloud.sql.connector.Connector``を
モックし、SQLAlchemyエンジンが正しいcreator関数（pg8000ドライバ経由）で構築されることと、
``src.db.session.init_engine``がKEIBA_DB_BACKENDの値によって正しく分岐することのみを検証する。
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from sqlalchemy.engine import Engine

import src.db.session as session


class TestGetCloudSqlEngine(unittest.TestCase):
    def test_builds_engine_with_pg8000_url(self):
        from src.db.cloud_sql import get_cloud_sql_engine

        with patch("google.cloud.sql.connector.Connector") as mock_connector_cls:
            mock_connector = mock_connector_cls.return_value
            mock_connector.connect.return_value = "FAKE_DBAPI_CONN"

            engine = get_cloud_sql_engine(
                instance_connection_name="my-project:asia-northeast1:keiba-db",
                user="keiba_user",
                password="keiba_pass",
                db="keiba_db",
            )

            self.assertIsInstance(engine, Engine)
            # creator パターン: URLはドライバのみでホスト情報を持たない
            self.assertEqual(str(engine.url), "postgresql+pg8000://")
            mock_connector_cls.assert_called_once_with()

    def test_creator_calls_connector_connect_with_expected_args(self):
        from src.db.cloud_sql import get_cloud_sql_engine

        with patch("google.cloud.sql.connector.Connector") as mock_connector_cls:
            mock_connector = mock_connector_cls.return_value
            mock_connector.connect.return_value = "FAKE_DBAPI_CONN"

            engine = get_cloud_sql_engine(
                instance_connection_name="my-project:asia-northeast1:keiba-db",
                user="keiba_user",
                password="keiba_pass",
                db="keiba_db",
            )

            creator = engine.pool._creator
            result = creator()

            self.assertEqual(result, "FAKE_DBAPI_CONN")
            mock_connector.connect.assert_called_once_with(
                "my-project:asia-northeast1:keiba-db",
                "pg8000",
                user="keiba_user",
                password="keiba_pass",
                db="keiba_db",
            )


class TestInitEngineBackendBranching(unittest.TestCase):
    def setUp(self):
        # グローバルなシングルトンをテストごとにリセットする
        session._engine = None
        session._SessionLocal = None

    def tearDown(self):
        session._engine = None
        session._SessionLocal = None

    def test_default_backend_uses_database_url(self):
        env = {"DATABASE_URL": "postgresql+psycopg://u:p@localhost:5432/db"}
        with patch.dict(os.environ, env, clear=True):
            with patch("src.db.session.create_engine") as mock_create_engine:
                mock_create_engine.return_value = MagicMock(spec=Engine)
                engine = session.init_engine()

                mock_create_engine.assert_called_once_with(
                    "postgresql+psycopg://u:p@localhost:5432/db", pool_pre_ping=True
                )
                self.assertIs(engine, mock_create_engine.return_value)

    def test_unset_backend_defaults_to_database_url_even_if_cloud_sql_vars_present(self):
        # KEIBA_DB_BACKEND を明示的に設定しない限り、Cloud SQL用変数があっても
        # 従来通りDATABASE_URL直結を維持する（後方互換）。
        env = {
            "DATABASE_URL": "postgresql+psycopg://u:p@localhost:5432/db",
            "CLOUD_SQL_INSTANCE_CONNECTION_NAME": "proj:region:inst",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch("src.db.session.create_engine") as mock_create_engine:
                mock_create_engine.return_value = MagicMock(spec=Engine)
                session.init_engine()
                mock_create_engine.assert_called_once()

    def test_cloud_sql_backend_calls_get_cloud_sql_engine(self):
        env = {
            "KEIBA_DB_BACKEND": "cloud_sql",
            "CLOUD_SQL_INSTANCE_CONNECTION_NAME": "my-project:asia-northeast1:keiba-db",
            "CLOUD_SQL_DB_USER": "keiba_user",
            "CLOUD_SQL_DB_PASSWORD": "keiba_pass",
            "CLOUD_SQL_DB_NAME": "keiba_db",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch("src.db.cloud_sql.get_cloud_sql_engine") as mock_get_engine:
                mock_get_engine.return_value = MagicMock(spec=Engine)
                engine = session.init_engine()

                mock_get_engine.assert_called_once_with(
                    instance_connection_name="my-project:asia-northeast1:keiba-db",
                    user="keiba_user",
                    password="keiba_pass",
                    db="keiba_db",
                )
                self.assertIs(engine, mock_get_engine.return_value)

    def test_explicit_url_bypasses_cloud_sql_backend(self):
        # init_engine(url=...) を明示的に渡した場合は KEIBA_DB_BACKEND を無視し、
        # 従来通りそのURLでcreate_engineする（テスト等での上書き用途を壊さない）。
        env = {
            "KEIBA_DB_BACKEND": "cloud_sql",
            "CLOUD_SQL_INSTANCE_CONNECTION_NAME": "proj:region:inst",
            "CLOUD_SQL_DB_USER": "u",
            "CLOUD_SQL_DB_PASSWORD": "p",
            "CLOUD_SQL_DB_NAME": "d",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch("src.db.session.create_engine") as mock_create_engine, patch(
                "src.db.cloud_sql.get_cloud_sql_engine"
            ) as mock_get_engine:
                mock_create_engine.return_value = MagicMock(spec=Engine)
                engine = session.init_engine(url="sqlite:///:memory:")

                mock_create_engine.assert_called_once_with(
                    "sqlite:///:memory:", pool_pre_ping=True
                )
                mock_get_engine.assert_not_called()
                self.assertIs(engine, mock_create_engine.return_value)

    def test_cloud_sql_backend_missing_required_env_raises_keyerror(self):
        env = {"KEIBA_DB_BACKEND": "cloud_sql"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(KeyError):
                session.init_engine()


if __name__ == "__main__":
    unittest.main()
