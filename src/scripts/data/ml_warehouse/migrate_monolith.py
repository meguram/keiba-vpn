"""旧 keiba_horse_ml.sqlite3 → シャード DB へ移行。"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from src.scripts.data.ml_warehouse.paths import horse_shard_key, legacy_monolith_sqlite_path
from src.scripts.data.ml_warehouse.sqlite_store import ShardWriter, rebuild_catalog

logger = logging.getLogger(__name__)


def migrate_monolith_to_shards(base: Path) -> dict:
    src = legacy_monolith_sqlite_path(base)
    if not src.exists():
        return {"migrated": False, "reason": "no monolith"}

    con = sqlite3.connect(str(src))
    writers: dict[str, ShardWriter] = {}
    shard_stats: dict[str, dict[str, int]] = {}

    def w(shard: str) -> ShardWriter:
        if shard not in writers:
            writers[shard] = ShardWriter(base, shard)
        return writers[shard]

    n_prof = n_hist = n_train = 0
    for row in con.execute("SELECT * FROM horse_profile"):
        hid = str(row[0])
        sh = horse_shard_key(hid)
        w(sh).upsert_profile(tuple(row))
        n_prof += 1

    cols_hist = [d[1] for d in con.execute("PRAGMA table_info(horse_race_history)")]
    for row in con.execute(f"SELECT {','.join(cols_hist)} FROM horse_race_history"):
        hid = str(row[0])
        w(horse_shard_key(hid)).upsert_history(tuple(row))
        n_hist += 1

    cols_tr = [d[1] for d in con.execute("PRAGMA table_info(horse_training_row)")]
    for row in con.execute(f"SELECT {','.join(cols_tr)} FROM horse_training_row"):
        hid = str(row[0])
        w(horse_shard_key(hid)).upsert_training(tuple(row))
        n_train += 1

    con.close()
    for sh, wr in writers.items():
        shard_stats[sh] = wr.close()

    rebuild_catalog(base, shard_stats)
    logger.info(
        "monolith 移行: profiles=%d history=%d training=%d shards=%d",
        n_prof, n_hist, n_train, len(shard_stats),
    )
    return {
        "migrated": True,
        "profiles": n_prof,
        "history": n_hist,
        "training": n_train,
        "shards": len(shard_stats),
    }
