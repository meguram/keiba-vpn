"""bloodline.db SQLite アクセス層。

parquet を全件展開する代わりに SQLite の行単位クエリを使い、
API プロセスの定常メモリを削減する。

  race_records.parquet       235MB in-mem → SELECT 結果分のみ
  horse_pedigree_cats.parquet 846MB in-mem → INDEX 付き SELECT で数MB以下
  race_result_slim.parquet    78MB in-mem  → SELECT 結果分のみ

ローカルパス : data/bloodline/bloodline.db
GCS パス     : {GCS_OTHERS}/bloodline_db/bloodline.db
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
DB_LOCAL = ROOT / "data" / "bloodline" / "bloodline.db"
GCS_BLOB  = "chuou/data/others/bloodline_db/bloodline.db"

_conn_lock = threading.Lock()
_connection: sqlite3.Connection | None = None


# ── GCS 連携 ────────────────────────────────────────────────────

def upload_to_gcs(storage=None) -> bool:
    """bloodline.db を GCS にアップロードする。マイグレーション後に呼ぶ。"""
    if not DB_LOCAL.exists():
        logger.error("bloodline.db が存在しません: %s", DB_LOCAL)
        return False
    try:
        if storage is None:
            from src.scraper.storage import HybridStorage
            storage = HybridStorage()
        bucket = storage._get_bucket()
        blob = bucket.blob(GCS_BLOB)
        blob.upload_from_filename(
            str(DB_LOCAL), content_type="application/x-sqlite3"
        )
        size_mb = DB_LOCAL.stat().st_size / 1024 ** 2
        logger.info(
            "bloodline.db を GCS にアップロード: %.1fMB → gs://%s/%s",
            size_mb, storage._bucket_name, GCS_BLOB,
        )
        return True
    except Exception as exc:
        logger.warning("GCS アップロード失敗: %s", exc)
        return False


def download_from_gcs(storage=None) -> bool:
    """GCS から bloodline.db をダウンロードする。"""
    try:
        if storage is None:
            from src.scraper.storage import HybridStorage
            storage = HybridStorage()
        bucket = storage._get_bucket()
        blob = bucket.blob(GCS_BLOB)
        if not blob.exists():
            logger.warning("GCS に bloodline.db がありません: %s", GCS_BLOB)
            return False
        DB_LOCAL.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(DB_LOCAL))
        size_mb = DB_LOCAL.stat().st_size / 1024 ** 2
        logger.info("bloodline.db を GCS からダウンロード: %.1fMB", size_mb)
        return True
    except Exception as exc:
        logger.warning("GCS ダウンロード失敗: %s", exc)
        return False


# ── 接続管理 ────────────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    """プロセス内シングルトン接続を返す。DB が無ければ GCS からダウンロード。"""
    global _connection
    with _conn_lock:
        if _connection is not None:
            return _connection
        if not DB_LOCAL.exists():
            logger.info("bloodline.db 未存在 → GCS からダウンロードを試みます")
            download_from_gcs()
        if not DB_LOCAL.exists():
            raise FileNotFoundError(
                f"bloodline.db が見つかりません: {DB_LOCAL}\n"
                "python -m src.scripts.data.migrate_bloodline_to_sqlite を実行してください。"
            )
        conn = sqlite3.connect(str(DB_LOCAL), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # WAL モードで並行読み取りを最適化
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA cache_size=-32000")   # 32MB ページキャッシュ
        conn.execute("PRAGMA temp_store=MEMORY")
        _connection = conn
        size_mb = DB_LOCAL.stat().st_size / 1024 ** 2
        logger.info("bloodline.db 接続確立: %.1fMB (%s)", size_mb, DB_LOCAL)
        return _connection


def close_connection() -> None:
    global _connection
    with _conn_lock:
        if _connection is not None:
            _connection.close()
            _connection = None


def db_exists() -> bool:
    return DB_LOCAL.exists()


# ── クエリヘルパー ────────────────────────────────────────────────

def load_race_results(
    conn: sqlite3.Connection | None = None,
    stallion_ids: list[str] | None = None,
    columns: list[str] | None = None,
) -> "pd.DataFrame":
    """race_results テーブルを DataFrame として読み込む。

    stallion_ids を指定すると WHERE IN フィルタを適用して小さな結果を返す。
    """
    import pandas as pd

    if conn is None:
        conn = get_connection()

    col_str = ", ".join(columns) if columns else "*"
    if stallion_ids:
        placeholders = ",".join("?" * len(stallion_ids))
        sql = f"SELECT {col_str} FROM race_results WHERE stallion_id IN ({placeholders})"
        return pd.read_sql(sql, conn, params=stallion_ids)
    return pd.read_sql(f"SELECT {col_str} FROM race_results", conn)


def load_horse_names(conn: sqlite3.Connection | None = None) -> dict[str, str]:
    """horse_id → horse_name マップを返す。"""
    if conn is None:
        conn = get_connection()
    cur = conn.execute("SELECT horse_id, horse_name FROM horse_names")
    return {row["horse_id"]: row["horse_name"] for row in cur}


def load_ancestor_horses(
    conn: sqlite3.Connection | None = None,
    ancestor_ids: list[str] | None = None,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """(ancestor_to_father, ancestor_to_mother) を返す。

    ancestor_ids を指定すると対象 ancestor のみ取得する。
    """
    if conn is None:
        conn = get_connection()

    if ancestor_ids:
        placeholders = ",".join("?" * len(ancestor_ids))
        sql = (
            "SELECT ancestor_id, side, horse_ids FROM ancestor_horses "
            f"WHERE ancestor_id IN ({placeholders})"
        )
        rows = conn.execute(sql, ancestor_ids).fetchall()
    else:
        rows = conn.execute(
            "SELECT ancestor_id, side, horse_ids FROM ancestor_horses"
        ).fetchall()

    father: dict[str, set[str]] = {}
    mother: dict[str, set[str]] = {}
    for row in rows:
        ancestor_id = row["ancestor_id"]
        side = row["side"]
        horse_ids = set(json.loads(row["horse_ids"]))
        if side == "father":
            father[ancestor_id] = horse_ids
        else:
            mother[ancestor_id] = horse_ids
    return father, mother


def load_ancestor_names(conn: sqlite3.Connection | None = None) -> dict[str, str]:
    """ancestor_id → name マップを返す。"""
    if conn is None:
        conn = get_connection()
    cur = conn.execute("SELECT ancestor_id, name FROM ancestor_names")
    return {row["ancestor_id"]: row["name"] for row in cur}


def load_pedigree_cats(
    conn: sqlite3.Connection | None = None,
    stallion_id: str | None = None,
    cat: int | None = None,
    gen: int | None = None,
    gen_min: int | None = None,
    gen_max: int | None = None,
    horse_ids: list[str] | None = None,
    columns: list[str] | None = None,
) -> "pd.DataFrame":
    """horse_pedigree_cats テーブルを絞り込んで返す。

    全件取得は 846MB だが、horse_ids + cat + gen で絞ると数MB 以下。
    """
    import pandas as pd

    if conn is None:
        conn = get_connection()

    col_str = ", ".join(columns) if columns else "*"
    conds: list[str] = []
    params: list = []
    if stallion_id is not None:
        conds.append("stallion_id = ?")
        params.append(stallion_id)
    if cat is not None:
        conds.append("cat = ?")
        params.append(cat)
    if gen is not None:
        conds.append("gen = ?")
        params.append(gen)
    if gen_min is not None:
        conds.append("gen >= ?")
        params.append(gen_min)
    if gen_max is not None:
        conds.append("gen <= ?")
        params.append(gen_max)
    if horse_ids is not None:
        placeholders = ",".join("?" * len(horse_ids))
        conds.append(f"horse_id IN ({placeholders})")
        params.extend(horse_ids)

    where = f"WHERE {' AND '.join(conds)}" if conds else ""
    return pd.read_sql(
        f"SELECT {col_str} FROM pedigree_cats {where}", conn, params=params
    )


def load_race_result_slim(
    conn: sqlite3.Connection | None = None,
    horse_ids: list[str] | None = None,
) -> "pd.DataFrame":
    """race_result_slim テーブルを返す (horse_ids 指定で絞り込み)。"""
    import pandas as pd

    if conn is None:
        conn = get_connection()

    if horse_ids:
        placeholders = ",".join("?" * len(horse_ids))
        return pd.read_sql(
            f"SELECT * FROM race_result_slim WHERE horse_id IN ({placeholders})",
            conn, params=horse_ids,
        )
    return pd.read_sql("SELECT * FROM race_result_slim", conn)


def load_horse_bms(
    conn: sqlite3.Connection | None = None,
    horse_ids: list[str] | None = None,
    columns: list[str] | None = None,
) -> "pd.DataFrame":
    """horse_bms テーブルを返す。"""
    import pandas as pd

    if conn is None:
        conn = get_connection()

    col_str = ", ".join(columns) if columns else "*"
    if horse_ids:
        placeholders = ",".join("?" * len(horse_ids))
        return pd.read_sql(
            f"SELECT {col_str} FROM horse_bms WHERE horse_id IN ({placeholders})",
            conn, params=horse_ids,
        )
    return pd.read_sql(f"SELECT {col_str} FROM horse_bms", conn)


def load_stallion_lineage(conn: sqlite3.Connection | None = None) -> "pd.DataFrame":
    """stallion_lineage テーブルを返す (件数が少ないので全件)。"""
    import pandas as pd

    if conn is None:
        conn = get_connection()
    return pd.read_sql("SELECT * FROM stallion_lineage", conn)
