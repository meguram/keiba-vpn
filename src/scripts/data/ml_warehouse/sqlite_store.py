"""馬系 SQLite シャード（horse_id 先頭4桁）のスキーマと接続。"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from src.scripts.data.ml_warehouse.paths import catalog_sqlite_path, horse_shard_db_path, horse_shards_dir

SCHEMA_VERSION = 1

HORSE_TABLES_DDL = """
CREATE TABLE IF NOT EXISTS _meta (
  key TEXT PRIMARY KEY,
  value TEXT
);

CREATE TABLE IF NOT EXISTS horse_profile (
  horse_id TEXT PRIMARY KEY,
  horse_name TEXT,
  name_en TEXT,
  status TEXT,
  sex TEXT,
  birthday TEXT,
  trainer TEXT,
  owner TEXT,
  breeder TEXT,
  birthplace TEXT,
  sire TEXT,
  dam TEXT,
  dam_sire TEXT,
  total_earnings TEXT,
  career TEXT,
  source_path TEXT,
  source_mtime REAL,
  ingested_at REAL
);

CREATE TABLE IF NOT EXISTS horse_race_history (
  horse_id TEXT NOT NULL,
  race_id TEXT NOT NULL,
  date TEXT,
  venue TEXT,
  race_name TEXT,
  race_round TEXT,
  surface TEXT,
  distance TEXT,
  track_condition TEXT,
  weather TEXT,
  field_size TEXT,
  bracket_number TEXT,
  horse_number TEXT,
  finish_position TEXT,
  jockey_name TEXT,
  jockey_weight TEXT,
  odds TEXT,
  popularity TEXT,
  finish_time TEXT,
  time_sec REAL,
  margin TEXT,
  passing_order TEXT,
  last_3f TEXT,
  weight TEXT,
  weight_change TEXT,
  winner TEXT,
  source_path TEXT,
  ingested_at REAL,
  PRIMARY KEY (horse_id, race_id)
);
CREATE INDEX IF NOT EXISTS idx_hrh_race_id ON horse_race_history(race_id);
CREATE INDEX IF NOT EXISTS idx_hrh_date ON horse_race_history(date);

CREATE TABLE IF NOT EXISTS horse_training_row (
  horse_id TEXT NOT NULL,
  training_date TEXT NOT NULL,
  seq INTEGER NOT NULL,
  race_info TEXT,
  day_of_week TEXT,
  course TEXT,
  track_condition TEXT,
  rider TEXT,
  time_raw TEXT,
  lap_times TEXT,
  position TEXT,
  leg_color TEXT,
  evaluation TEXT,
  rank TEXT,
  comment TEXT,
  pages_fetched INTEGER,
  source_path TEXT,
  source_mtime REAL,
  ingested_at REAL,
  PRIMARY KEY (horse_id, training_date, seq)
);
CREATE INDEX IF NOT EXISTS idx_ht_date ON horse_training_row(training_date);
"""

CATALOG_DDL = """
CREATE TABLE IF NOT EXISTS shard_registry (
  shard TEXT PRIMARY KEY,
  db_path TEXT NOT NULL,
  profile_count INTEGER NOT NULL DEFAULT 0,
  history_count INTEGER NOT NULL DEFAULT 0,
  training_count INTEGER NOT NULL DEFAULT 0,
  file_bytes INTEGER NOT NULL DEFAULT 0,
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS horse_lookup (
  horse_id TEXT PRIMARY KEY,
  shard TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_horse_lookup_shard ON horse_lookup(shard);
"""


def connect_shard(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA cache_size=-64000")
    return con


def init_shard_schema(con: sqlite3.Connection, shard: str) -> None:
    con.executescript(HORSE_TABLES_DDL)
    con.execute(
        "INSERT OR REPLACE INTO _meta(key, value) VALUES (?, ?)",
        ("schema_version", str(SCHEMA_VERSION)),
    )
    con.execute(
        "INSERT OR REPLACE INTO _meta(key, value) VALUES (?, ?)",
        ("horse_shard", shard),
    )
    con.commit()


class ShardWriter:
    """1 シャード DB へのバッファ付き upsert。"""

    def __init__(self, base: Path, shard: str):
        self.shard = shard
        self.db_path = horse_shard_db_path(base, shard)
        self.con = connect_shard(self.db_path)
        init_shard_schema(self.con, shard)
        self.stats = {"profiles": 0, "history": 0, "training": 0}

    def close(self) -> dict[str, int]:
        self.con.commit()
        self.con.close()
        try:
            nbytes = self.db_path.stat().st_size
        except OSError:
            nbytes = 0
        self.stats["file_bytes"] = nbytes
        return self.stats

    def upsert_profile(self, row: tuple[Any, ...]) -> None:
        self.con.execute(
            """INSERT OR REPLACE INTO horse_profile (
              horse_id, horse_name, name_en, status, sex, birthday, trainer, owner, breeder,
              birthplace, sire, dam, dam_sire, total_earnings, career, source_path, source_mtime, ingested_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            row,
        )
        self.stats["profiles"] += 1

    def upsert_history(self, row: tuple[Any, ...]) -> None:
        self.con.execute(
            """INSERT OR REPLACE INTO horse_race_history (
              horse_id, race_id, date, venue, race_name, race_round, surface, distance, track_condition,
              weather, field_size, bracket_number, horse_number, finish_position, jockey_name, jockey_weight,
              odds, popularity, finish_time, time_sec, margin, passing_order, last_3f, weight, weight_change,
              winner, source_path, ingested_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            row,
        )
        self.stats["history"] += 1

    def upsert_training(self, row: tuple[Any, ...]) -> None:
        self.con.execute(
            """INSERT OR REPLACE INTO horse_training_row (
              horse_id, training_date, seq, race_info, day_of_week, course, track_condition, rider,
              time_raw, lap_times, position, leg_color, evaluation, rank, comment, pages_fetched,
              source_path, source_mtime, ingested_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            row,
        )
        self.stats["training"] += 1


def rebuild_catalog(base: Path, shard_stats: dict[str, dict[str, int]]) -> Path:
    """shard_registry / horse_lookup を再構築。"""
    cat_path = catalog_sqlite_path(base)
    cat_path.parent.mkdir(parents=True, exist_ok=True)
    con = connect_shard(cat_path)
    con.executescript(CATALOG_DDL)
    now = time.time()
    con.execute("DELETE FROM shard_registry")
    con.execute("DELETE FROM horse_lookup")

    shards_dir = horse_shards_dir(base)
    for shard, st in sorted(shard_stats.items()):
        rel = f"sqlite/horses/{shard}.sqlite3"
        con.execute(
            """INSERT INTO shard_registry(
              shard, db_path, profile_count, history_count, training_count, file_bytes, updated_at
            ) VALUES (?,?,?,?,?,?,?)""",
            (
                shard,
                rel,
                st.get("profiles", 0),
                st.get("history", 0),
                st.get("training", 0),
                st.get("file_bytes", 0),
                now,
            ),
        )
        dbp = shards_dir / f"{shard}.sqlite3"
        if not dbp.exists():
            continue
        scon = connect_shard(dbp)
        try:
            seen: set[str] = set()
            for sql in (
                "SELECT horse_id FROM horse_profile",
                "SELECT DISTINCT horse_id FROM horse_race_history",
                "SELECT DISTINCT horse_id FROM horse_training_row",
            ):
                for (hid,) in scon.execute(sql):
                    h = str(hid).strip()
                    if not h or h in seen:
                        continue
                    seen.add(h)
                    con.execute(
                        "INSERT OR REPLACE INTO horse_lookup(horse_id, shard) VALUES (?, ?)",
                        (h, shard),
                    )
        finally:
            scon.close()

    con.commit()
    con.close()
    return cat_path
