from __future__ import annotations

from pathlib import Path

DEFAULT_YEARS = ("2020", "2021", "2022", "2023", "2024", "2025")

# レース開催年ではなく horse_id 先頭4桁（生年コホート。GCS 馬パスと同じ）
HORSE_SHARD_LEN = 4


def warehouse_root(base: str | Path) -> Path:
    return Path(base) / "data" / "local" / "ml" / "warehouse"


def by_year_dir(base: str | Path, year: str) -> Path:
    return warehouse_root(base) / "by_year" / year


def horse_shards_dir(base: str | Path) -> Path:
    return warehouse_root(base) / "sqlite" / "horses"


def horse_shard_key(horse_id: str) -> str:
    """horse_id 先頭4桁。短い ID はゼロ埋めしない（そのまま先頭）。"""
    hid = str(horse_id).strip()
    if len(hid) < HORSE_SHARD_LEN:
        return hid or "_"
    return hid[:HORSE_SHARD_LEN]


def horse_shard_db_path(base: str | Path, shard: str) -> Path:
    return horse_shards_dir(base) / f"{shard}.sqlite3"


def catalog_sqlite_path(base: str | Path) -> Path:
    return warehouse_root(base) / "sqlite" / "catalog.sqlite3"


def legacy_monolith_sqlite_path(base: str | Path) -> Path:
    """旧単一 DB（移行元）。"""
    return warehouse_root(base) / "sqlite" / "keiba_horse_ml.sqlite3"


def sqlite_path(base: str | Path) -> Path:
    """後方互換: カタログ DB を指す。"""
    return catalog_sqlite_path(base)


def manifest_path(base: str | Path) -> Path:
    return warehouse_root(base) / "manifest.json"


def local_tables_year(base: str | Path, year: str) -> Path:
    return Path(base) / "data" / "local" / "tables" / year
