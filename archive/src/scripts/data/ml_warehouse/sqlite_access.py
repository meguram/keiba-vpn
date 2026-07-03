"""シャード SQLite への読み取り（1頭・複数頭・レース出走表）。"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from src.scripts.data.ml_warehouse.paths import (
    catalog_sqlite_path,
    horse_shard_db_path,
    horse_shard_key,
)


def shard_for_horse_id(horse_id: str) -> str:
    return horse_shard_key(horse_id)


def lookup_shard(base: Path, horse_id: str) -> str | None:
    cat = catalog_sqlite_path(base)
    if not cat.exists():
        return shard_for_horse_id(horse_id)
    con = sqlite3.connect(str(cat))
    try:
        row = con.execute(
            "SELECT shard FROM horse_lookup WHERE horse_id = ?",
            (str(horse_id).strip(),),
        ).fetchone()
        return row[0] if row else None
    finally:
        con.close()


@contextmanager
def open_shard(base: Path, shard: str) -> Iterator[sqlite3.Connection]:
    p = horse_shard_db_path(base, shard)
    if not p.exists():
        raise FileNotFoundError(p)
    con = sqlite3.connect(str(p))
    con.row_factory = sqlite3.Row
    try:
        yield con
    finally:
        con.close()


@contextmanager
def open_horse(base: Path, horse_id: str) -> Iterator[sqlite3.Connection]:
    shard = lookup_shard(base, horse_id) or shard_for_horse_id(horse_id)
    with open_shard(base, shard) as con:
        yield con


@contextmanager
def open_horses(base: Path, horse_ids: list[str]) -> Iterator[sqlite3.Connection]:
    """複数頭: 必要なシャードだけ ATTACH した接続を返す。"""
    shards = sorted({shard_for_horse_id(h) for h in horse_ids if str(h).strip()})
    if not shards:
        con = sqlite3.connect(":memory:")
        con.row_factory = sqlite3.Row
        try:
            yield con
        finally:
            con.close()
        return

    primary = horse_shard_db_path(base, shards[0])
    if not primary.exists():
        raise FileNotFoundError(primary)
    con = sqlite3.connect(str(primary))
    con.row_factory = sqlite3.Row
    try:
        for i, sh in enumerate(shards[1:], start=1):
            p = horse_shard_db_path(base, sh)
            if p.exists():
                con.execute(f"ATTACH DATABASE ? AS s{i}", (str(p),))
        yield con
    finally:
        con.close()


def fetch_profile(base: Path, horse_id: str) -> dict | None:
    with open_horse(base, horse_id) as con:
        row = con.execute(
            "SELECT * FROM horse_profile WHERE horse_id = ?",
            (str(horse_id).strip(),),
        ).fetchone()
        return dict(row) if row else None


def fetch_training(
    base: Path,
    horse_id: str,
    *,
    date_from: str | None = None,
    limit: int = 500,
) -> list[dict]:
    q = "SELECT * FROM horse_training_row WHERE horse_id = ?"
    params: list = [str(horse_id).strip()]
    if date_from:
        q += " AND training_date >= ?"
        params.append(date_from)
    q += " ORDER BY training_date DESC, seq LIMIT ?"
    params.append(limit)
    with open_horse(base, horse_id) as con:
        return [dict(r) for r in con.execute(q, params).fetchall()]
