"""Cloud SQL Python Connector 経由の SQLAlchemy エンジン生成。

VPS（サービング）・GCP（スクレイピング/ML/スケジュール実行）の両方から
同一の Cloud SQL (PostgreSQL) インスタンスへ接続する必要があるため、
``DATABASE_URL`` 直結の代わりに Cloud SQL Python Connector を使う経路を提供する。

``src/db/session.py`` から ``KEIBA_DB_BACKEND=cloud_sql`` のときのみ呼ばれる。
GCP 認証は事前に ``src.config.gcp_credentials.ensure_google_application_credentials()``
が呼ばれ、ADC（Application Default Credentials）が利用可能になっている前提。
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def get_cloud_sql_engine(
    instance_connection_name: str,
    user: str,
    password: str,
    db: str,
) -> Engine:
    """Cloud SQL Python Connector を使って SQLAlchemy エンジンを構築する。

    Args:
        instance_connection_name: ``project:region:instance`` 形式のインスタンス接続名。
        user: DB ユーザー名。
        password: DB パスワード。
        db: DB 名。

    Returns:
        ``postgresql+pg8000://`` ドライバで Cloud SQL Connector 経由に
        接続する SQLAlchemy ``Engine``。
    """
    from google.cloud.sql.connector import Connector

    connector = Connector()

    def getconn():
        return connector.connect(
            instance_connection_name,
            "pg8000",
            user=user,
            password=password,
            db=db,
        )

    return create_engine(
        "postgresql+pg8000://",
        creator=getconn,
        pool_pre_ping=True,
    )
