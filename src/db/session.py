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


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def engine_options(url: str) -> dict:
    """直結 PostgreSQL 向けのエンジン設定。

    接続・プール待ちに上限を置き、DB 不調時にリクエストが既定の 30 秒以上固まるのを防ぐ。
    PostgreSQL 以外（テストの SQLite など）には何も足さない。
    """
    opts: dict = {"pool_pre_ping": True}
    if url.startswith("postgresql"):
        opts["pool_timeout"] = _env_int("DB_POOL_TIMEOUT_SEC", 10)
        opts["connect_args"] = {"connect_timeout": _env_int("DB_CONNECT_TIMEOUT_SEC", 5)}
    return opts


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
        target = url or database_url()
        _engine = create_engine(target, **engine_options(target))

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
