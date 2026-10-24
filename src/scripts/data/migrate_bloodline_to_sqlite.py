#!/usr/bin/env python3
"""bloodline.db マイグレーションスクリプト。

parquet から SQLite を構築し、GCS にアップロードする。

スキーマ変更への対応:
  各テーブルは parquet のカラム情報をハッシュして DB 内に保存する。
  parquet のカラムが追加/変更された場合、そのテーブルだけ自動で再構築する。
  再構築 = DROP → CREATE → INSERT (その他テーブルは影響なし)。

更新頻度の違いへの対応:
  --update-tables race_results,horse_names  # 指定テーブルのみ再構築
  --full                                     # 全テーブルを強制再構築
  (デフォルト)                               # スキーマ差分を検出して再構築

Usage:
    python -m src.scripts.data.migrate_bloodline_to_sqlite
    python -m src.scripts.data.migrate_bloodline_to_sqlite --full
    python -m src.scripts.data.migrate_bloodline_to_sqlite --update-tables race_results,horse_names
    python -m src.scripts.data.migrate_bloodline_to_sqlite --no-upload
    python -m src.scripts.data.migrate_bloodline_to_sqlite --upload-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sqlite3
import sys
import time
from glob import glob
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parents[3]
ART_DIR    = ROOT / "data/page_reference/note_aptitude_race"
IDX_DIR    = ROOT / "data/page_reference/pedigree_race_index"
PED10_DIR  = ROOT / "data/local/research/pedigree_10gen"
TABLES_DIR = ROOT / "data/page_reference/tables"

from src.pipeline.data.bloodline_sqlite import DB_LOCAL, upload_to_gcs


# ── スキーマバージョン管理 ────────────────────────────────────────

_META_INIT = """
CREATE TABLE IF NOT EXISTS _schema_meta (
    table_name  TEXT PRIMARY KEY,
    col_hash    TEXT NOT NULL,
    built_at    TEXT NOT NULL,
    row_count   INTEGER
);
"""


def _col_hash(df: pd.DataFrame) -> str:
    """DataFrame のカラムリストからハッシュを生成する。"""
    key = ",".join(sorted(df.columns))
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _stored_hash(conn: sqlite3.Connection, table_name: str) -> str | None:
    cur = conn.execute(
        "SELECT col_hash FROM _schema_meta WHERE table_name = ?", (table_name,)
    )
    row = cur.fetchone()
    return row[0] if row else None


def _store_meta(conn: sqlite3.Connection, table_name: str, df: pd.DataFrame) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO _schema_meta(table_name, col_hash, built_at, row_count)"
        " VALUES(?,?,datetime('now'),?)",
        (table_name, _col_hash(df), len(df)),
    )


def _needs_rebuild(conn: sqlite3.Connection, table_name: str, df: pd.DataFrame) -> bool:
    """parquet のカラムハッシュが DB と異なれば True (スキーマ変更あり)。"""
    stored = _stored_hash(conn, table_name)
    current = _col_hash(df)
    if stored != current:
        if stored:
            logger.info(
                "  [%s] スキーマ変更を検出 (old=%s, new=%s) → 再構築",
                table_name, stored, current,
            )
        return True
    return False


def _rebuild_table(
    conn: sqlite3.Connection,
    table_name: str,
    df: pd.DataFrame,
    chunksize: int = 8000,
) -> int:
    """テーブルを DROP → INSERT して _schema_meta を更新する。"""
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    df.to_sql(table_name, conn, if_exists="replace", index=False, chunksize=chunksize)
    _store_meta(conn, table_name, df)
    return len(df)


# ── インデックス再構築 ────────────────────────────────────────────

_INDEXES = {
    "race_results": [
        "CREATE INDEX IF NOT EXISTS idx_rr_stallion ON race_results(stallion_id)",
        "CREATE INDEX IF NOT EXISTS idx_rr_horse    ON race_results(horse_id)",
        "CREATE INDEX IF NOT EXISTS idx_rr_date     ON race_results(date)",
        "CREATE INDEX IF NOT EXISTS idx_rr_stn_date ON race_results(stallion_id, date, surface, venue)",
    ],
    "pedigree_cats": [
        "CREATE INDEX IF NOT EXISTS idx_pc_stl_cat_gen ON pedigree_cats(stallion_id, cat, gen)",
        "CREATE INDEX IF NOT EXISTS idx_pc_horse        ON pedigree_cats(horse_id)",
    ],
    "race_result_slim": [
        "CREATE INDEX IF NOT EXISTS idx_rrs_horse ON race_result_slim(horse_id)",
        "CREATE INDEX IF NOT EXISTS idx_rrs_date  ON race_result_slim(date)",
    ],
    "ancestor_horses": [
        "CREATE INDEX IF NOT EXISTS idx_ah_anc ON ancestor_horses(ancestor_id)",
    ],
}


def _rebuild_indexes(conn: sqlite3.Connection, table_name: str) -> None:
    for sql in _INDEXES.get(table_name, []):
        conn.execute(sql)


# ── テーブル構築関数 ────────────────────────────────────────────────

def build_race_results(
    conn: sqlite3.Connection, force: bool = False
) -> tuple[int, str]:
    """race_records + horse_to_sire → race_results。頻繁に更新されるテーブル。"""
    t0 = time.time()
    rec_path = ART_DIR / "race_records.parquet"
    h2s_path = ART_DIR / "horse_to_sire.parquet"
    if not rec_path.exists():
        logger.warning("race_records.parquet 未存在: %s", rec_path)
        return 0, "skipped"

    rec = pd.read_parquet(rec_path)
    rec["race_id"]  = rec["race_id"].astype(str)
    rec["horse_id"] = rec["horse_id"].astype(str)

    if h2s_path.exists():
        h2s = pd.read_parquet(h2s_path, columns=["horse_id", "stallion_id", "stallion_name"])
        h2s["horse_id"] = h2s["horse_id"].astype(str)
        rec = rec.merge(h2s, on="horse_id", how="left")

    if not force and not _needs_rebuild(conn, "race_results", rec):
        logger.info("  [race_results] スキーマ変更なし → スキップ")
        del rec
        return 0, "skipped"

    n = _rebuild_table(conn, "race_results", rec)
    _rebuild_indexes(conn, "race_results")
    del rec
    logger.info("  [race_results] %d 行 (%.1fs)", n, time.time() - t0)
    return n, "rebuilt"


def build_horse_names(
    conn: sqlite3.Connection, force: bool = False
) -> tuple[int, str]:
    """全年の race_result_flat.parquet (glob) → horse_names。頻繁に更新。"""
    t0 = time.time()
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for f in sorted(glob(str(TABLES_DIR / "*/race_result_flat.parquet"))):
        try:
            df = pd.read_parquet(f, columns=["horse_id", "horse_name"])
            df = df.dropna(subset=["horse_id", "horse_name"])
            for hid, nm in zip(df["horse_id"].astype(str), df["horse_name"].astype(str)):
                if hid and nm and hid not in seen:
                    rows.append((hid, nm))
                    seen.add(hid)
        except Exception as exc:
            logger.warning("  glob スキップ %s: %s", f, exc)

    dummy = pd.DataFrame(rows, columns=["horse_id", "horse_name"])
    if not force and not _needs_rebuild(conn, "horse_names", dummy):
        logger.info("  [horse_names] スキーマ変更なし → スキップ")
        return 0, "skipped"

    conn.execute("DROP TABLE IF EXISTS horse_names")
    conn.execute(
        "CREATE TABLE horse_names (horse_id TEXT PRIMARY KEY, horse_name TEXT)"
    )
    conn.executemany(
        "INSERT OR REPLACE INTO horse_names(horse_id, horse_name) VALUES(?,?)", rows
    )
    _store_meta(conn, "horse_names", dummy)
    n = len(rows)
    logger.info("  [horse_names] %d 件 (%.1fs)", n, time.time() - t0)
    return n, "rebuilt"


def build_ancestor_horses(
    conn: sqlite3.Connection, force: bool = False
) -> tuple[int, str]:
    """ancestor_to_horses.parquet → ancestor_horses (horse_ids を JSON 格納)。"""
    t0 = time.time()
    inv_path = PED10_DIR / "ancestor_to_horses.parquet"
    if not inv_path.exists():
        logger.warning("  ancestor_to_horses.parquet 未存在")
        return 0, "skipped"

    inv = pd.read_parquet(inv_path)
    if not force and not _needs_rebuild(conn, "ancestor_horses", inv):
        logger.info("  [ancestor_horses] スキーマ変更なし → スキップ")
        return 0, "skipped"

    conn.execute("DROP TABLE IF EXISTS ancestor_horses")
    conn.execute("""
        CREATE TABLE ancestor_horses (
            ancestor_id TEXT NOT NULL,
            side        TEXT NOT NULL,
            horse_ids   TEXT NOT NULL,
            PRIMARY KEY (ancestor_id, side)
        )
    """)
    rows = [
        (str(r.ancestor_id), str(r.side),
         json.dumps([str(h) for h in r.horse_ids] if r.horse_ids is not None else []))
        for r in inv.itertuples(index=False)
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO ancestor_horses(ancestor_id, side, horse_ids) VALUES(?,?,?)",
        rows,
    )
    _store_meta(conn, "ancestor_horses", inv)
    _rebuild_indexes(conn, "ancestor_horses")
    n = len(rows)
    del inv, rows
    logger.info("  [ancestor_horses] %d 行 (%.1fs)", n, time.time() - t0)
    return n, "rebuilt"


def build_ancestor_names(
    conn: sqlite3.Connection, force: bool = False
) -> tuple[int, str]:
    t0 = time.time()
    path = PED10_DIR / "ancestor_id_to_name.parquet"
    if not path.exists():
        logger.warning("  ancestor_id_to_name.parquet 未存在")
        return 0, "skipped"

    df = pd.read_parquet(path).rename(columns={"horse_id": "ancestor_id"})
    if not force and not _needs_rebuild(conn, "ancestor_names", df):
        logger.info("  [ancestor_names] スキーマ変更なし → スキップ")
        return 0, "skipped"

    _rebuild_table(conn, "ancestor_names", df)
    n = len(df)
    del df
    logger.info("  [ancestor_names] %d 件 (%.1fs)", n, time.time() - t0)
    return n, "rebuilt"


def build_pedigree_cats(
    conn: sqlite3.Connection, force: bool = False
) -> tuple[int, str]:
    """horse_pedigree_cats.parquet → pedigree_cats。大きいが変更頻度は低い。"""
    t0 = time.time()
    path = IDX_DIR / "horse_pedigree_cats.parquet"
    if not path.exists():
        logger.warning("  horse_pedigree_cats.parquet 未存在")
        return 0, "skipped"

    logger.info("  [pedigree_cats] 読み込み中 (大ファイル)...")
    df = pd.read_parquet(path)
    if not force and not _needs_rebuild(conn, "pedigree_cats", df):
        logger.info("  [pedigree_cats] スキーマ変更なし → スキップ")
        del df
        return 0, "skipped"

    n = _rebuild_table(conn, "pedigree_cats", df, chunksize=10000)
    _rebuild_indexes(conn, "pedigree_cats")
    del df
    logger.info("  [pedigree_cats] %d 行 (%.1fs)", n, time.time() - t0)
    return n, "rebuilt"


def build_race_result_slim(
    conn: sqlite3.Connection, force: bool = False
) -> tuple[int, str]:
    t0 = time.time()
    path = IDX_DIR / "race_result_slim.parquet"
    if not path.exists():
        logger.warning("  race_result_slim.parquet 未存在")
        return 0, "skipped"

    df = pd.read_parquet(path)
    if not force and not _needs_rebuild(conn, "race_result_slim", df):
        logger.info("  [race_result_slim] スキーマ変更なし → スキップ")
        del df
        return 0, "skipped"

    n = _rebuild_table(conn, "race_result_slim", df)
    _rebuild_indexes(conn, "race_result_slim")
    del df
    logger.info("  [race_result_slim] %d 行 (%.1fs)", n, time.time() - t0)
    return n, "rebuilt"


def build_horse_bms(
    conn: sqlite3.Connection, force: bool = False
) -> tuple[int, str]:
    t0 = time.time()
    path = IDX_DIR / "horse_bms.parquet"
    if not path.exists():
        logger.warning("  horse_bms.parquet 未存在")
        return 0, "skipped"

    df = pd.read_parquet(path)
    if not force and not _needs_rebuild(conn, "horse_bms", df):
        logger.info("  [horse_bms] スキーマ変更なし → スキップ")
        return 0, "skipped"

    n = _rebuild_table(conn, "horse_bms", df)
    del df
    logger.info("  [horse_bms] %d 件 (%.1fs)", n, time.time() - t0)
    return n, "rebuilt"


def build_stallion_lineage(
    conn: sqlite3.Connection, force: bool = False
) -> tuple[int, str]:
    t0 = time.time()
    path = IDX_DIR / "stallion_lineage.parquet"
    if not path.exists():
        logger.warning("  stallion_lineage.parquet 未存在")
        return 0, "skipped"

    df = pd.read_parquet(path)
    if not force and not _needs_rebuild(conn, "stallion_lineage", df):
        logger.info("  [stallion_lineage] スキーマ変更なし → スキップ")
        return 0, "skipped"

    n = _rebuild_table(conn, "stallion_lineage", df)
    del df
    logger.info("  [stallion_lineage] %d 件 (%.1fs)", n, time.time() - t0)
    return n, "rebuilt"


# ── テーブルビルダー登録 ────────────────────────────────────────────
# 更新頻度の目安: high=毎週 / medium=月次 / low=不定期

_BUILDERS: dict[str, tuple] = {
    # table_name: (builder_fn, update_freq)
    "race_results":     (build_race_results,     "high"),
    "horse_names":      (build_horse_names,       "high"),
    "ancestor_horses":  (build_ancestor_horses,   "medium"),
    "ancestor_names":   (build_ancestor_names,    "low"),
    "pedigree_cats":    (build_pedigree_cats,     "low"),
    "race_result_slim": (build_race_result_slim,  "high"),
    "horse_bms":        (build_horse_bms,         "low"),
    "stallion_lineage": (build_stallion_lineage,  "low"),
}


# ── メイン ────────────────────────────────────────────────────────

def build_db(
    targets: list[str] | None = None,
    force: bool = False,
) -> Path:
    """bloodline.db を構築または更新して DB パスを返す。

    targets: 再構築するテーブル名リスト。None = スキーマ差分のみ再構築。
    force:   True = スキーマ変更検出をスキップして全テーブルを強制再構築。
    """
    DB_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_LOCAL))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_META_INIT)
    conn.commit()

    results: dict[str, str] = {}
    for tname, (builder_fn, freq) in _BUILDERS.items():
        should_force = force or (targets is not None and tname in targets)
        n, status = builder_fn(conn, force=should_force)
        results[tname] = f"{status} ({n} rows)" if n else status

    conn.execute("ANALYZE")
    conn.commit()
    conn.close()

    size_mb = DB_LOCAL.stat().st_size / 1024 ** 2
    logger.info("bloodline.db 完成: %.1fMB → %s", size_mb, DB_LOCAL)
    for tname, res in results.items():
        logger.info("  %-20s %s", tname, res)
    return DB_LOCAL


def main() -> int:
    parser = argparse.ArgumentParser(description="bloodline.db マイグレーション")
    parser.add_argument(
        "--full", action="store_true",
        help="全テーブルを強制再構築 (スキーマ差分チェックをスキップ)",
    )
    parser.add_argument(
        "--update-tables", default="",
        help="指定テーブルのみ再構築 (例: race_results,horse_names)",
    )
    parser.add_argument(
        "--no-upload", action="store_true",
        help="GCS アップロードをスキップ",
    )
    parser.add_argument(
        "--upload-only", action="store_true",
        help="DB 再構築せず GCS のみアップロード",
    )
    args = parser.parse_args()

    if args.upload_only:
        if not DB_LOCAL.exists():
            logger.error("bloodline.db が見つかりません")
            return 1
    else:
        targets = (
            [t.strip() for t in args.update_tables.split(",") if t.strip()]
            if args.update_tables else None
        )
        t_start = time.time()
        build_db(targets=targets, force=args.full)
        logger.info("構築完了: %.1f 分", (time.time() - t_start) / 60)

    if not args.no_upload:
        ok = upload_to_gcs()
        if not ok:
            logger.warning("GCS アップロード失敗（ローカル DB は完成しています）")
    else:
        logger.info("GCS アップロードをスキップしました")

    return 0


if __name__ == "__main__":
    sys.exit(main())
