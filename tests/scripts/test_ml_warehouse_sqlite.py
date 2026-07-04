"""ML warehouse 馬 SQLite シャードのユニットテスト。"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.scripts.data.ml_warehouse.paths import catalog_sqlite_path, horse_shard_db_path, horse_shard_key
from src.scripts.data.ml_warehouse.sqlite_access import fetch_training, lookup_shard, open_horse
from src.scripts.data.ml_warehouse.sqlite_builder import rebuild_horse_sqlite
from src.scripts.data.ml_warehouse.sqlite_store import ShardWriter, rebuild_catalog


class TestHorseSqliteShards(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        hid = "2023105347"
        sh = horse_shard_key(hid)
        train_dir = self.base / "data" / "local" / "horse_training" / sh
        train_dir.mkdir(parents=True)
        payload = {
            "horse_id": hid,
            "entries": [
                {"date": "2025-03-01", "course": "栗東", "time_raw": "85.0"},
                {"date": "2025-03-01", "course": "栗東", "time_raw": "86.0"},
            ],
        }
        (train_dir / f"{hid}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_shard_writer_and_catalog(self) -> None:
        st = rebuild_horse_sqlite(self.base, ["2025"], include_training=True)
        self.assertEqual(st["layout"], "sharded_by_horse_id_prefix4")
        self.assertGreater(st["training_rows"], 0)
        db = horse_shard_db_path(self.base, "2023")
        self.assertTrue(db.exists())
        self.assertTrue(catalog_sqlite_path(self.base).exists())
        self.assertEqual(lookup_shard(self.base, "2023105347"), "2023")

    def test_training_primary_key_seq(self) -> None:
        rebuild_horse_sqlite(self.base, ["2025"], include_training=True)
        with open_horse(self.base, "2023105347") as con:
            n = con.execute(
                "SELECT COUNT(*) FROM horse_training_row WHERE horse_id = ?",
                ("2023105347",),
            ).fetchone()[0]
            self.assertEqual(n, 2)
            dup = con.execute(
                """SELECT COUNT(*) - COUNT(DISTINCT horse_id || '|' || training_date || '|' || seq)
                   FROM horse_training_row"""
            ).fetchone()[0]
            self.assertEqual(dup, 0)

    def test_fetch_training_api(self) -> None:
        rebuild_horse_sqlite(self.base, ["2025"], include_training=True)
        rows = fetch_training(self.base, "2023105347", limit=5)
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
