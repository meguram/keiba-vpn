"""SQLAlchemy セッション管理。"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

_engine = None
_SessionLocal: sessionmaker[Session] | None = None


def database_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg://keiba:keiba@localhost:5432/keiba",
    )


def init_engine(url: str | None = None):
    global _engine, _SessionLocal
    if url is None and not os.environ.get("DATABASE_URL") and not os.environ.get("KEIBA_DB_BACKEND"):
        from src.utils.project_env import load_project_dotenv

        load_project_dotenv()

    backend = os.environ.get("KEIBA_DB_BACKEND", "").strip().lower()
    if url is None and backend == "cloud_sql":
        from src.db.cloud_sql import get_cloud_sql_engine

        _engine = get_cloud_sql_engine(
            instance_connection_name=os.environ["CLOUD_SQL_INSTANCE_CONNECTION_NAME"],
            user=os.environ["CLOUD_SQL_DB_USER"],
            password=os.environ["CLOUD_SQL_DB_PASSWORD"],
            db=os.environ["CLOUD_SQL_DB_NAME"],
        )
    else:
        # 既定（KEIBA_DB_BACKEND 未設定）: 従来通り DATABASE_URL から直結する。
        _engine = create_engine(url or database_url(), pool_pre_ping=True)

    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    if _SessionLocal is None:
        init_engine()
    assert _SessionLocal is not None
    return _SessionLocal


@contextmanager
def get_session() -> Generator[Session, None, None]:
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
