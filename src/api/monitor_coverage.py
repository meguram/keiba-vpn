"""
/monitor 向けデータカバレッジ API（GCS 存在確認 + PostgreSQL 集計）。

方針（課金安全）:
  - GCS: batch_list_blobs / ローカル date_coverage のみ（JSON download しない）
  - PostgreSQL: DEV/STG はリアルタイム SQL、prod は db_coverage キャッシュ
  - Calculated: PG + ローカル flat parquet（GCS 不使用）
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_COVERAGE_DIR = _PROJECT_ROOT / "data/local/meta/db_coverage"
RACE_LIST_DIR = _PROJECT_ROOT / "data/page_reference/race_lists"
TABLES_DIR = _PROJECT_ROOT / "data/page_reference/tables"

PG_RAW_COLUMNS = ("pg_races", "pg_entries", "pg_race_results")
CALCULATED_COLUMNS = (
    "flat_local",
    "megu_v2",
    "megu_coverage_ok",
)

_VENUE_MAP = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}


def current_keiba_env() -> str:
    return (os.environ.get("KEIBA_ENV") or "dev").strip().lower()


def aggregation_mode() -> str:
    """prod は日次キャッシュ、それ以外はリアルタイム。"""
    return "cached" if current_keiba_env() == "prod" else "realtime"


def _now_jst_iso() -> str:
    jst = timezone(timedelta(hours=9))
    return datetime.now(jst).isoformat(timespec="seconds")


def _db_coverage_path(date: str) -> Path:
    return DB_COVERAGE_DIR / date[:4] / f"{date}.json"


def load_db_coverage_cache(date: str) -> dict | None:
    path = _db_coverage_path(date)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("db_coverage load failed [%s]: %s", date, e)
        return None


def save_db_coverage_cache(date: str, payload: dict) -> None:
    path = _db_coverage_path(date)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload.setdefault("updated_at", _now_jst_iso())
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def _load_race_ids_for_date(date: str) -> list[str]:
    from src.utils.race_list_for_date import load_jra_race_ids_for_opening_date

    return load_jra_race_ids_for_opening_date(date, require_monitor_eligible=True)


def _race_meta(race_ids: list[str]) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    for rid in race_ids:
        if len(rid) >= 12:
            meta[rid] = {
                "race_num": int(rid[10:12]),
                "venue": _VENUE_MAP.get(rid[4:6], rid[4:6]),
            }
    return meta


def _build_gcs_matrix(date: str, race_ids: list[str], storage: Any) -> dict[str, Any]:
    from src.scraper.date_coverage import (
        TRACK_CATEGORIES,
        load_date_coverage,
        load_not_available,
    )
    from src.utils.race_quality_rules import enrich_presence_row

    cov = load_date_coverage(date)
    updated_at = (cov or {}).get("updated_at", "")
    year = date[:4]

    cat_keys: dict[str, set[str]] = {}
    for cat in TRACK_CATEGORIES:
        try:
            cat_keys[cat] = set(storage.batch_list_blobs(cat, year).keys())
        except Exception:
            cat_keys[cat] = set()

    cat_na: dict[str, set[str]] = {
        cat: load_not_available(cat, year) for cat in TRACK_CATEGORIES
    }

    matrix: dict[str, dict[str, bool | None]] = {}
    for rid in race_ids:
        row: dict[str, bool | None] = {}
        for cat in TRACK_CATEGORIES:
            if rid in cat_na[cat]:
                row[cat] = None
            elif rid in cat_keys[cat]:
                row[cat] = True
            else:
                row[cat] = False
        matrix[rid] = enrich_presence_row(row, rid, storage)

    return {
        "categories": TRACK_CATEGORIES,
        "matrix": matrix,
        "local_index_updated_at": updated_at,
    }


def _local_file_exists(storage: Any, category: str, race_id: str) -> bool:
    """GCS を使わずローカル JSON の存在のみ確認。"""
    try:
        paths = [storage._local_path(category, race_id)]
        legacy = storage._legacy_local_path(category, race_id)
        if legacy != paths[0]:
            paths.append(legacy)
        return any(p.exists() and p.stat().st_size > 2 for p in paths)
    except Exception:
        return False


def _build_dev_local_matrix(date: str, race_ids: list[str], storage: Any) -> dict[str, Any]:
    from src.scraper.date_coverage import TRACK_CATEGORIES, load_date_coverage

    cov = load_date_coverage(date)
    matrix: dict[str, dict[str, bool]] = {}
    for rid in race_ids:
        row: dict[str, bool] = {}
        for cat in TRACK_CATEGORIES:
            row[cat] = _local_file_exists(storage, cat, rid)
        matrix[rid] = row

    return {
        "categories": TRACK_CATEGORIES,
        "matrix": matrix,
        "local_index_updated_at": (cov or {}).get("updated_at", ""),
        "note": "ローカル JSON のみ（GCS list / download 不使用）",
    }


def _build_dev_raw_matrix(date: str, race_ids: list[str], storage: Any) -> dict[str, Any]:
    """DEV: 要件（全カテゴリ必須）+ ローカルファイル索引。"""
    from src.scraper.date_coverage import TRACK_CATEGORIES

    requirement: dict[str, dict[str, bool]] = {}
    for rid in race_ids:
        requirement[rid] = {cat: True for cat in TRACK_CATEGORIES}

    local_index = _build_dev_local_matrix(date, race_ids, storage)

    return {
        "mode": "dev",
        "requirement": {
            "categories": TRACK_CATEGORIES,
            "matrix": requirement,
            "note": "scrape_process.md / TRACK_CATEGORIES 準拠のあるべき姿",
        },
        "local_index": local_index,
    }


def _query_pg_raw_for_date(session: Session, date: str, race_ids: list[str]) -> dict[str, dict]:
    if not race_ids:
        return {}

    rows = session.execute(
        text("""
            SELECT r.race_id,
                   1 AS has_race,
                   (SELECT COUNT(*) FROM entries e WHERE e.race_id = r.race_id) AS entry_cnt,
                   (SELECT COUNT(*) FROM race_results rr
                    WHERE rr.race_id = r.race_id
                      AND rr.finish_pos > 0 AND rr.finish_time_sec > 0) AS finisher_cnt
            FROM races r
            WHERE r.race_id = ANY(:ids)
        """),
        {"ids": race_ids},
    ).fetchall()

    by_rid: dict[str, dict] = {}
    for row in rows:
        by_rid[row.race_id] = {
            "pg_races": bool(row.has_race),
            "pg_entries": int(row.entry_cnt or 0),
            "pg_race_results": int(row.finisher_cnt or 0),
        }

    result: dict[str, dict] = {}
    for rid in race_ids:
        rec = by_rid.get(rid, {})
        result[rid] = {
            "pg_races": rec.get("pg_races", False),
            "pg_entries": rec.get("pg_entries", 0),
            "pg_race_results": rec.get("pg_race_results", 0),
        }
    return result


def _query_pg_calculated_for_date(session: Session, date: str, race_ids: list[str]) -> dict[str, dict]:
    if not race_ids:
        return {}

    rows = session.execute(
        text("""
            SELECT r.race_id,
                   r.race_name,
                   r.surface,
                   (SELECT COUNT(*) FROM race_results rr
                    WHERE rr.race_id = r.race_id
                      AND rr.finish_pos > 0 AND rr.finish_time_sec > 0) AS finisher_cnt,
                   (SELECT COUNT(*) FROM megu_index mi
                    WHERE mi.race_id = r.race_id AND mi.model_version = 'v2'
                      AND mi.computation_status IN ('valid', 'out_of_range')) AS megu_cnt,
                   (SELECT COUNT(*) FROM race_results rr
                    WHERE rr.race_id = r.race_id
                      AND rr.finish_pos > 0 AND rr.finish_time_sec > 0
                      AND EXISTS (
                        SELECT 1 FROM megu_index mi
                        WHERE mi.race_id = rr.race_id AND mi.horse_id = rr.horse_id
                          AND mi.model_version = 'v2'
                          AND mi.computation_status IN ('valid', 'out_of_range')
                      )) AS megu_covered_cnt
            FROM races r
            WHERE r.race_id = ANY(:ids)
        """),
        {"ids": race_ids},
    ).fetchall()

    flat_ids = _flat_race_ids_for_date(date, race_ids)

    result: dict[str, dict] = {}
    for row in rows:
        fin = int(row.finisher_cnt or 0)
        megu = int(row.megu_cnt or 0)
        covered = int(row.megu_covered_cnt or 0)
        result[row.race_id] = {
            "flat_local": row.race_id in flat_ids,
            "megu_v2": megu,
            "megu_covered": covered,
            "megu_coverage_ok": fin > 0 and covered >= fin,
            "finisher_count": fin,
            "race_name": row.race_name,
            "surface": row.surface,
        }

    for rid in race_ids:
        if rid not in result:
            result[rid] = {
                "flat_local": rid in flat_ids,
                "megu_v2": 0,
                "megu_coverage_ok": False,
                "finisher_count": 0,
                "race_name": None,
                "surface": None,
            }
    return result


def _flat_race_ids_for_date(date: str, race_ids: list[str]) -> set[str]:
    """ローカル race_result_flat.parquet から該当日の race_id を取得（GCS 不使用）。"""
    if not race_ids:
        return set()

    year = date[:4]
    path = TABLES_DIR / year / "race_result_flat.parquet"
    if not path.exists():
        return set()

    iso_date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(
            path,
            columns=["race_id", "date"],
            filters=[("date", "=", iso_date)],
        )
        found = {str(x) for x in table.column("race_id").to_pylist()}
        return found & set(race_ids)
    except Exception as e:
        logger.warning("flat parquet read failed [%s]: %s", date, e)
        return set()


def _pg_layer_from_cache(cached: dict, race_ids: list[str], *, layer: str) -> dict[str, dict]:
    matrix = (cached.get(layer) or {}).get("matrix") or {}
    return {rid: matrix.get(rid, {}) for rid in race_ids}


def build_raw_matrix(date: str, *, view: str, storage: Any | None = None) -> dict[str, Any]:
    race_ids = _load_race_ids_for_date(date)
    race_meta = _race_meta(race_ids)

    if view == "dev":
        if storage is None:
            raise ValueError("storage required for dev view")
        dev = _build_dev_raw_matrix(date, race_ids, storage)
        return {
            "date": date,
            "view": "dev",
            "race_ids": race_ids,
            "race_meta": race_meta,
            "gcs": None,
            "postgresql": None,
            **dev,
            "aggregation_mode": "local_index",
        }

    if storage is None:
        raise ValueError("storage required for stg view")

    gcs = _build_gcs_matrix(date, race_ids, storage)
    pg_matrix: dict[str, dict] = {}
    pg_mode = aggregation_mode()
    pg_error: str | None = None

    cached = load_db_coverage_cache(date) if pg_mode == "cached" else None
    if cached and cached.get("raw"):
        pg_matrix = _pg_layer_from_cache(cached, race_ids, layer="raw")
    else:
        try:
            from src.db.session import get_session, init_engine

            init_engine()
            with get_session() as session:
                pg_matrix = _query_pg_raw_for_date(session, date, race_ids)
        except Exception as e:
            pg_error = str(e)
            logger.warning("PG raw query failed [%s]: %s", date, e)

    return {
        "date": date,
        "view": "stg",
        "race_ids": race_ids,
        "race_meta": race_meta,
        "gcs": gcs,
        "postgresql": {
            "columns": list(PG_RAW_COLUMNS),
            "matrix": pg_matrix,
            "mode": pg_mode,
            "error": pg_error,
        },
        "aggregation_mode": pg_mode,
    }


def build_calculated_matrix(date: str, *, view: str) -> dict[str, Any]:
    race_ids = _load_race_ids_for_date(date)
    race_meta = _race_meta(race_ids)

    if view == "dev":
        return {
            "date": date,
            "view": "dev",
            "race_ids": race_ids,
            "race_meta": race_meta,
            "requirement": {
                "columns": list(CALCULATED_COLUMNS),
                "note": "DEV では PG/flat 実測なし。STG タブで確認してください。",
            },
            "postgresql": None,
            "aggregation_mode": "none",
        }

    pg_matrix: dict[str, dict] = {}
    pg_mode = aggregation_mode()
    pg_error: str | None = None

    cached = load_db_coverage_cache(date) if pg_mode == "cached" else None
    if cached and cached.get("calculated"):
        pg_matrix = _pg_layer_from_cache(cached, race_ids, layer="calculated")
    else:
        try:
            from src.db.session import get_session, init_engine

            init_engine()
            with get_session() as session:
                pg_matrix = _query_pg_calculated_for_date(session, date, race_ids)
        except Exception as e:
            pg_error = str(e)
            logger.warning("PG calculated query failed [%s]: %s", date, e)

    return {
        "date": date,
        "view": "stg",
        "race_ids": race_ids,
        "race_meta": race_meta,
        "postgresql": {
            "columns": list(CALCULATED_COLUMNS) + ["finisher_count"],
            "matrix": pg_matrix,
            "mode": pg_mode,
            "error": pg_error,
        },
        "aggregation_mode": pg_mode,
    }


def build_db_coverage_cache_for_date(date: str) -> dict:
    """prod 向け: 1 日分の PG カバレッジを構築してローカル保存。"""
    from src.db.session import get_session, init_engine

    race_ids = _load_race_ids_for_date(date)
    init_engine()
    with get_session() as session:
        raw = _query_pg_raw_for_date(session, date, race_ids)
        calculated = _query_pg_calculated_for_date(session, date, race_ids)

    payload = {
        "date": date,
        "race_ids": race_ids,
        "raw": {"columns": list(PG_RAW_COLUMNS), "matrix": raw},
        "calculated": {"columns": list(CALCULATED_COLUMNS), "matrix": calculated},
        "updated_at": _now_jst_iso(),
    }
    save_db_coverage_cache(date, payload)
    return payload


def monitor_context(*, gcs_enabled: bool | None = None, db_available: bool | None = None) -> dict[str, Any]:
    if gcs_enabled is None:
        try:
            from src.scraper.storage import HybridStorage
            gcs_enabled = bool(HybridStorage().gcs_enabled)
        except Exception:
            gcs_enabled = False

    if db_available is None:
        try:
            from src.db.session import get_session, init_engine

            init_engine()
            with get_session() as session:
                session.execute(text("SELECT 1"))
                db_available = True
        except Exception:
            db_available = False

    return {
        "keiba_env": current_keiba_env(),
        "gcs_enabled": gcs_enabled,
        "db_available": db_available,
        "aggregation_mode": aggregation_mode(),
    }
