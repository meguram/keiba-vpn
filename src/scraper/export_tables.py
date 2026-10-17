"""
年 x カテゴリ単位で GCS データを Parquet テーブルにエクスポートする。

メモリ安全設計: カテゴリごとにチャンク処理 + キャッシュクリアで
28GB マシンでも OOM しない。

フラット版 (_flat.parquet): 1行 = 1馬。ML / 分析向き
レース単位版 (_race.parquet): 1行 = 1レース。元構造保持

Usage:
    python -m src.scraper.export_tables --years 2024
    python -m src.scraper.export_tables --years 2021,2022,2023,2024,2025
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.keiba_logging import script_basic_config

logger = logging.getLogger(__name__)

JRA_PLACE_CODES = frozenset(f"{i:02d}" for i in range(1, 11))

JRA_VENUE_NAMES = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}

FLAT_CATEGORIES = [
    "race_result", "race_shutuba", "race_index", "race_shutuba_past",
    "race_odds", "race_paddock", "race_barometer", "race_oikiri",
    "race_trainer_comment", "race_result_lap", "smartrc_race",
]

# ML / 分析向け: 1レース=複数行(entries) とみなせる追加カテゴリ（欠損レース多めのため export 時は COMPLETENESS_EXEMPT 扱い）
ML_EXTRA_FLAT_CATEGORIES = [
    "race_info",
    "race_lap",
    "race_lap_on_time",
    "race_return",
    "race_detail",
    "race_result_on_time",
    "horse_lap",
    "race_predictions",
]

ALL_CATEGORIES = FLAT_CATEGORIES + ML_EXTRA_FLAT_CATEGORIES + ["race_pair_odds"]

RACE_META_FIELDS = [
    "race_id", "date", "venue", "surface", "distance", "direction",
    "grade", "race_class", "weather", "track_condition", "start_time",
    "field_size", "race_name",
]

ENTRY_KEY_MAP = {
    "smartrc_race": "runners",
    "race_result_lap": "entries_lap",
}

PAIR_ODDS_SECTIONS = ["umaren", "wide", "umatan"]

OUTPUT_DIR = Path("data/page_reference/tables")

CHUNK_SIZE = 200

FLAT_MAX_JSON_BYTES = 1024


def is_jra_race(race_id: str) -> bool:
    return len(race_id) >= 6 and race_id[4:6] in JRA_PLACE_CODES


def _extract_race_meta(data: dict, meta_index: dict[str, dict] | None = None) -> dict:
    """レースレベルのメタデータを抽出する。"""
    meta = {}
    race_id = data.get("race_id", "")
    fallback = meta_index.get(race_id, {}) if meta_index else {}

    for field in RACE_META_FIELDS:
        val = data.get(field)
        # 距離・頭数の 0 は「取れていない」。db.netkeiba の結果ページは障害レースの距離を取れず 0 で保存されている
        # （0 のまま特徴量に入ると log(0) で無限大になる）。出馬表のメタで補う。
        if (val is None or (isinstance(val, str) and val.strip() == "")
                or (field in ("distance", "field_size") and (val == 0 or (isinstance(val, float) and val != val)))):
            val = fallback.get(field)
        if field in ("distance", "field_size") and isinstance(val, float) and val == val and val.is_integer():
            val = int(val)  # 年によって列の型が Float64 と Int64 に分かれ、年をまたぐ読み込みが落ちるのを防ぐ
        meta[field] = val

    meta["venue_code"] = race_id[4:6] if len(race_id) >= 6 else ""
    if not meta.get("round") and len(race_id) == 12 and race_id.isdigit():
        meta["round"] = int(race_id[-2:])   # 古い結果データは round が 0 のまま。race_id の末尾 2 桁がレース番号
    if not meta.get("venue"):
        meta["venue"] = JRA_VENUE_NAMES.get(meta["venue_code"], "")
    return meta


def _build_meta_index(year: str, storage: Any) -> dict[str, dict]:
    """race_shutuba からレースメタデータのインデックスを構築する。
    メタフィールドのみ保持するため軽量 (~3,500 x ~200B = ~700KB)。
    """
    logger.info("[%s] メタデータインデックス構築 (race_shutuba) ...", year)
    blobs = storage.batch_list_blobs("race_shutuba", year)
    jra_keys = [k for k in blobs if is_jra_race(k)]

    index: dict[str, dict] = {}
    for key in jra_keys:
        data = storage.load("race_shutuba", key)
        if not data:
            continue
        meta = {}
        for field in RACE_META_FIELDS:
            meta[field] = data.get(field)
        index[key] = meta

    _clear_storage_cache(storage)
    logger.info("[%s] メタデータインデックス: %d races", year, len(index))
    return index


def _add_shutuba_meta(storage: Any, race_ids: Any, index: dict[str, dict]) -> int:
    """指定レースのメタを出馬表から index に足す（既にあるレースは触らない）。--refresh で追記する新しいレースの、
    障害レースの距離 0 などを出馬表のメタで補うために使う。足した件数を返す。"""
    n = 0
    for rid in race_ids:
        if rid in index:
            continue
        data = storage.load("race_shutuba", rid)
        if not data:
            continue
        index[rid] = {f: data.get(f) for f in RACE_META_FIELDS}
        n += 1
    return n


def _build_meta_index_from_race_result_flat(year: str) -> dict[str, dict]:
    """ローカル race_result_flat.parquet からメタを構築（race_shutuba の GCS 列挙・load を避ける）。"""
    p = OUTPUT_DIR / year / "race_result_flat.parquet"
    if not p.exists():
        return {}
    try:
        schema = pq.read_schema(p)
        names = set(schema.names)
        cols = ["race_id"] + [f for f in RACE_META_FIELDS if f in names and f != "race_id"]
        tbl = pq.read_table(p, columns=cols)
        df = tbl.to_pandas()
        if "race_id" not in df.columns or df.empty:
            return {}
        df = df.drop_duplicates(subset=["race_id"], keep="first")
        index: dict[str, dict] = {}
        for rec in df.to_dict("records"):
            rid = str(rec.get("race_id") or "").strip()
            if not rid:
                continue
            meta: dict[str, Any] = {"race_id": rid}
            for field in RACE_META_FIELDS:
                if field == "race_id":
                    continue
                if field in rec:
                    meta[field] = rec.get(field)
            index[rid] = meta
        logger.info(
            "[%s] メタデータ: ローカル race_result_flat から %d races（GCS shutuba なし）",
            year, len(index),
        )
        return index
    except Exception as e:
        logger.warning("[%s] race_result_flat メタ構築失敗: %s", year, e)
        return {}


def _clear_storage_cache(storage: Any) -> None:
    """HybridStorage のインメモリキャッシュを解放する。"""
    if hasattr(storage, "_load_cache"):
        with storage._load_cache_lock:
            storage._load_cache.clear()
    gc.collect()


def _data_to_flat_rows(
    data: dict, category: str,
    meta_index: dict[str, dict] | None = None,
) -> list[dict]:
    """1レース分のデータからフラット行を生成する。"""
    meta = _extract_race_meta(data, meta_index)
    entry_key = ENTRY_KEY_MAP.get(category, "entries")

    top_extra: dict[str, Any] = {}
    for k, v in data.items():
        if k in ("_meta", entry_key, "entries"):
            continue
        if k in meta:
            continue
        if isinstance(v, (list, dict)):
            s = json.dumps(v, ensure_ascii=False) if v else None
            if s and len(s) > FLAT_MAX_JSON_BYTES:
                continue
            top_extra[k] = s
        else:
            top_extra[k] = v

    entries = data.get(entry_key, [])
    rows = []

    if isinstance(entries, list) and entries:
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            row = dict(meta)
            row.update(top_extra)
            row.update(entry)
            rows.append(row)
    else:
        row = dict(meta)
        row.update(top_extra)
        rows.append(row)

    return rows


def _data_to_pair_odds_flat_rows(
    data: dict,
    meta_index: dict[str, dict] | None = None,
) -> list[dict]:
    """1レース分の race_pair_odds からフラット行を生成する。"""
    meta = _extract_race_meta(data, meta_index)
    rows = []
    for section in PAIR_ODDS_SECTIONS:
        items = data.get(section, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            row = dict(meta)
            row["odds_type"] = section
            row.update(item)
            rows.append(row)
    return rows


def _data_to_race_row(data: dict) -> dict:
    """1レース分のデータからレース単位行を生成する。"""
    row = {}
    for k, v in data.items():
        if k == "_meta":
            continue
        if isinstance(v, (list, dict)):
            row[k] = json.dumps(v, ensure_ascii=False) if v else None
        else:
            row[k] = v
    row.setdefault("race_id", data.get("race_id"))
    return row


def _safe_parquet_write(rows: list[dict], path: Path) -> None:
    """行リストからParquetファイルを書き出す。混合型カラムを文字列変換。"""
    if not rows:
        return
    df = pd.DataFrame(rows)
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(20)
            has_complex = any(isinstance(v, (dict, list)) for v in sample)
            if has_complex:
                df[col] = df[col].apply(
                    lambda x: json.dumps(x, ensure_ascii=False)
                    if isinstance(x, (dict, list)) else x
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")


def _merge_parquet_chunks(chunks: list[Path], output: Path) -> None:
    """チャンク Parquet ファイルを統合スキーマで結合する。"""
    if not chunks:
        return
    tables = []
    for p in chunks:
        tables.append(pq.read_table(p))

    try:
        merged_schema = pa.unify_schemas([t.schema for t in tables])
    except pa.ArrowTypeError:
        merged_schema = _resolve_schema_conflicts(
            [t.schema for t in tables]
        )

    unified = []
    for t in tables:
        cols = {}
        for field in merged_schema:
            if field.name in t.column_names:
                col = t.column(field.name)
                if col.type != field.type:
                    col = col.cast(field.type, safe=False)
                cols[field.name] = col
            else:
                cols[field.name] = pa.nulls(len(t), type=field.type)
        unified.append(pa.table(cols, schema=merged_schema))
        del t
    result = pa.concat_tables(unified)
    pq.write_table(result, output)
    del result, unified, tables
    for p in chunks:
        p.unlink(missing_ok=True)
    gc.collect()


def _resolve_schema_conflicts(schemas: list[pa.Schema]) -> pa.Schema:
    """型が衝突するフィールドを安全に統合する。int vs double → double 等。"""
    field_types: dict[str, list[pa.DataType]] = {}
    for schema in schemas:
        for field in schema:
            field_types.setdefault(field.name, []).append(field.type)

    resolved_fields = []
    for name, types in field_types.items():
        unique = list(set(types))
        if len(unique) == 1:
            resolved_fields.append(pa.field(name, unique[0]))
        else:
            all_numeric = all(
                pa.types.is_integer(t) or pa.types.is_floating(t)
                for t in unique
            )
            if all_numeric:
                resolved_fields.append(pa.field(name, pa.float64()))
            elif any(pa.types.is_large_string(t) or pa.types.is_string(t)
                     for t in unique):
                resolved_fields.append(pa.field(name, pa.string()))
            else:
                resolved_fields.append(pa.field(name, pa.string()))
            logger.info("スキーマ競合解決: %s %s → %s",
                        name, unique, resolved_fields[-1].type)
    return pa.schema(resolved_fields)


def _validate_export(flat_path: Path, category: str, n_races: int) -> dict:
    """エクスポート結果のバリデーション。Parquetメタデータから読み取るため軽量。"""
    v: dict[str, Any] = {}
    try:
        pf = pq.ParquetFile(flat_path)
        n_rows = pf.metadata.num_rows
        if n_races > 0:
            avg = n_rows / n_races
            v["avg_entries_per_race"] = round(avg, 1)
            if avg < 1.0 and category not in COMPLETENESS_EXEMPT:
                v["warning"] = f"entries が想定より少ない (avg={avg:.1f})"

        df_sample = pf.read_row_group(0).to_pandas()
        meta_nulls = {}
        for col in ("date", "surface", "distance", "venue"):
            if col in df_sample.columns:
                n = int(df_sample[col].isna().sum())
                if n > 0:
                    meta_nulls[col] = n
        if meta_nulls:
            v["meta_null_counts_sample"] = meta_nulls
    except Exception as e:
        v["validation_error"] = str(e)
    return v


COMPLETENESS_EXEMPT = frozenset({
    "race_barometer",
    "race_paddock",
    "race_trainer_comment",
    *ML_EXTRA_FLAT_CATEGORIES,
})


def _count_parquet_races(path: Path) -> int:
    """既存 Parquet ファイルのユニーク race_id 数を返す。"""
    return len(_existing_race_ids(path))


BRIDGE_COL = "result_source"
BRIDGE_VALUE = "on_time"


def _existing_race_ids(path: Path) -> set[str]:
    """flat に入っているレース ID。速報から橋渡しした行（result_source == on_time）は『確定結果ではない』ので数えない
    （数えると、確定結果が公開されても『既にエクスポート済み』になって入れ替わらない）。"""
    if not path.exists():
        return set()
    try:
        cols = ["race_id"] + ([BRIDGE_COL] if BRIDGE_COL in pq.read_schema(path).names else [])
        df = pq.read_table(path, columns=cols).to_pandas()
        if BRIDGE_COL in df.columns:
            df = df[df[BRIDGE_COL].fillna("") != BRIDGE_VALUE]
        return {str(x) for x in df["race_id"].tolist() if x}
    except Exception:
        return set()


def bridge_on_time_results(year: str) -> dict:
    """確定結果（db.netkeiba.com の race_result）の公開が数日遅れる間、速報（race_result_on_time。着順・タイム・通過順・上がりなど同じ列）
    から race_result_flat に行を足す。履歴系の特徴量（直近の開催の結果）が数日古くなるのを避けるため。
    橋渡しした行には result_source='on_time' を付け、確定結果が出たら入れ替わる（確定結果の取り込み対象から外れず、同じ馬番の行は後勝ち、
    確定結果に無い馬の行は掃除する）。race_result_race（レース単位の表）には足さない。"""
    out_dir = OUTPUT_DIR / year
    rr, ot = out_dir / "race_result_flat.parquet", out_dir / "race_result_on_time_flat.parquet"
    if not rr.exists() or not ot.exists():
        return {"bridged_races": 0, "reason": "flat が無い"}
    cur = pq.read_table(rr).to_pandas()
    if BRIDGE_COL not in cur.columns:
        cur[BRIDGE_COL] = None
    is_b = cur[BRIDGE_COL].fillna("") == BRIDGE_VALUE
    official = set(cur.loc[~is_b, "race_id"].astype(str))
    bridged = set(cur.loc[is_b, "race_id"].astype(str))
    stale = bridged & official
    changed = False
    if stale:
        cur = cur[~(is_b & cur["race_id"].astype(str).isin(stale))]
        changed = True
    new_df = pq.read_table(ot).to_pandas()
    new_df["race_id"] = new_df["race_id"].astype(str)
    need = set(new_df["race_id"]) - official - (bridged - stale)
    if need:
        add = new_df[new_df["race_id"].isin(need)].drop(columns=["finish_position_raw", "finish_status"], errors="ignore").copy()
        add = add[add.get("finish_position", 0).notna()] if "finish_position" in add.columns else add
        add[BRIDGE_COL] = BRIDGE_VALUE
        cur = pd.concat([cur, add], ignore_index=True)
        changed = True
    if changed:
        cur.to_parquet(rr, index=False, engine="pyarrow")
    stats = {"bridged_races": len(need), "removed_stale_races": len(stale)}
    if need:
        logger.warning("[%s] 確定結果(db.netkeiba.com)が未公開のレースが %d 件。速報で代用中（公開されると自動で入れ替わる）: %s", year, len(need), sorted(need)[:3])
    logger.info("[%s] 速報から race_result_flat へ橋渡し: %s", year, stats)
    return stats


def _append_parquet_rows(path: Path, new_rows: list[dict]) -> int:
    """既存 flat Parquet に行を追記して書き戻す。"""
    if not new_rows:
        return 0
    if path.exists():
        old = pq.read_table(path).to_pandas()
        merged = pd.concat([old, pd.DataFrame(new_rows)], ignore_index=True)
        if "race_id" in merged.columns:
            merged = merged.drop_duplicates(
                subset=["race_id", "horse_number"] if "horse_number" in merged.columns else ["race_id"],
                keep="last",
            )
    else:
        merged = pd.DataFrame(new_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, index=False, engine="pyarrow")
    return len(new_rows)


def export_category_chunked(
    year: str, category: str, storage: Any,
    meta_index: dict[str, dict] | None = None,
    expected_keys: set[str] | None = None,
) -> dict:
    """1年1カテゴリをチャンク処理でエクスポートする。OOM防止。
    expected_keys が指定された場合、カテゴリのレース数が一致しなければスキップ。
    既にエクスポート済みでレース数が一致している場合はスキップ (増分チェック)。
    """
    out_dir = OUTPUT_DIR / year
    out_dir.mkdir(parents=True, exist_ok=True)

    flat_path = out_dir / f"{category}_flat.parquet"
    race_path = out_dir / f"{category}_race.parquet"

    if (
        expected_keys
        and category not in COMPLETENESS_EXEMPT
        and flat_path.exists()
    ):
        ex_ids = _existing_race_ids(flat_path)
        if expected_keys <= ex_ids:
            stats: dict[str, Any] = {
                "category": category,
                "races": len(ex_ids),
                "flat_rows": -1,
                "race_rows": -1,
                "skipped": True,
                "reason": "ローカル flat が基準レースを包含。GCS batch_list スキップ",
            }
            if race_path.exists():
                stats["race_size_mb"] = round(race_path.stat().st_size / (1024 * 1024), 2)
            if flat_path.exists():
                stats["flat_size_mb"] = round(flat_path.stat().st_size / (1024 * 1024), 2)
            logger.info("[%s/%s] SKIP: %s", year, category, stats["reason"])
            return stats

    blobs = storage.batch_list_blobs(category, year)
    jra_keys = sorted(k for k in blobs if is_jra_race(k))
    n_total = len(jra_keys)

    stats: dict[str, Any] = {"category": category, "races": 0}

    if not jra_keys:
        stats["flat_rows"] = 0
        stats["race_rows"] = 0
        stats["note"] = "no data"
        return stats

    if expected_keys and category not in COMPLETENESS_EXEMPT:
        cat_keys = set(jra_keys)
        missing = expected_keys - cat_keys
        if missing:
            stats["flat_rows"] = 0
            stats["race_rows"] = 0
            stats["skipped"] = True
            stats["reason"] = (
                f"レース不足: {len(missing)}/{len(expected_keys)} 欠損 "
                f"(GCS上に{n_total}件, 基準{len(expected_keys)}件)"
            )
            logger.warning("[%s/%s] SKIP: %s", year, category, stats["reason"])
            return stats

    existing_ids = _existing_race_ids(flat_path)
    if existing_ids:
        missing_keys = [k for k in jra_keys if k not in existing_ids]
        if not missing_keys:
            stats["races"] = len(existing_ids)
            stats["flat_rows"] = -1
            stats["race_rows"] = -1
            stats["skipped"] = True
            stats["reason"] = f"既にエクスポート済み (Parquet={len(existing_ids)}, GCS={n_total})"
            logger.info("[%s/%s] SKIP: %s", year, category, stats["reason"])
            return stats
        if len(missing_keys) < len(jra_keys):
            jra_keys = missing_keys
            n_total = len(jra_keys)
            logger.info(
                "[%s/%s] 差分のみ %d レースを追記 (既存 %d / GCS合計 %d)",
                year, category, n_total, len(existing_ids), len(existing_ids) + n_total,
            )

    logger.info("[%s/%s] %d JRA races → チャンク処理 (chunk=%d)",
                year, category, n_total, CHUNK_SIZE)

    race_path = out_dir / f"{category}_race.parquet"
    flat_path = out_dir / f"{category}_flat.parquet"

    race_chunks: list[Path] = []
    flat_chunks: list[Path] = []
    total_flat_rows = 0
    total_race_rows = 0
    load_errors = 0
    races_loaded = 0
    chunk_dir = out_dir / f"_chunks_{category}"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    for chunk_start in range(0, n_total, CHUNK_SIZE):
        chunk_keys = jra_keys[chunk_start:chunk_start + CHUNK_SIZE]
        flat_rows: list[dict] = []
        race_rows: list[dict] = []

        for key in chunk_keys:
            data = storage.load(category, key)
            if not data:
                load_errors += 1
                continue
            races_loaded += 1

            race_rows.append(_data_to_race_row(data))

            if category == "race_pair_odds":
                flat_rows.extend(_data_to_pair_odds_flat_rows(data, meta_index))
            elif category in FLAT_CATEGORIES or category in ML_EXTRA_FLAT_CATEGORIES:
                flat_rows.extend(_data_to_flat_rows(data, category, meta_index))

        chunk_idx = chunk_start // CHUNK_SIZE

        if race_rows:
            chunk_path = chunk_dir / f"race_{chunk_idx:04d}.parquet"
            _safe_parquet_write(race_rows, chunk_path)
            race_chunks.append(chunk_path)
            total_race_rows += len(race_rows)

        if flat_rows:
            chunk_path = chunk_dir / f"flat_{chunk_idx:04d}.parquet"
            _safe_parquet_write(flat_rows, chunk_path)
            flat_chunks.append(chunk_path)
            total_flat_rows += len(flat_rows)

        del flat_rows, race_rows
        _clear_storage_cache(storage)

        done = min(chunk_start + CHUNK_SIZE, n_total)
        logger.info("[%s/%s] chunk %d/%d done (flat=%d, race=%d)",
                    year, category, done, n_total,
                    total_flat_rows, total_race_rows)

    if race_chunks:
        _merge_parquet_chunks(race_chunks, race_path)
    if flat_chunks:
        if existing_ids and flat_path.exists():
            # 差分追記モード: チャンクを読み込んで既存にマージ
            incr_rows: list[dict] = []
            for p in flat_chunks:
                incr_rows.extend(pq.read_table(p).to_pandas().to_dict("records"))
            added = _append_parquet_rows(flat_path, incr_rows)
            total_flat_rows = added
            for p in flat_chunks:
                p.unlink(missing_ok=True)
            flat_chunks = []
        else:
            _merge_parquet_chunks(flat_chunks, flat_path)

    import shutil
    shutil.rmtree(chunk_dir, ignore_errors=True)

    stats["races"] = races_loaded
    stats["load_errors"] = load_errors
    stats["race_rows"] = total_race_rows
    stats["flat_rows"] = total_flat_rows

    if race_path.exists():
        stats["race_file"] = str(race_path)
        stats["race_size_mb"] = round(race_path.stat().st_size / (1024 * 1024), 2)
    if flat_path.exists():
        stats["flat_file"] = str(flat_path)
        stats["flat_size_mb"] = round(flat_path.stat().st_size / (1024 * 1024), 2)
        stats["validation"] = _validate_export(flat_path, category, races_loaded)

    _clear_storage_cache(storage)
    return stats


def export_year(
    year: str, *,
    categories: list[str] | None = None,
    storage: Any = None,
    workers: int = 6,
    force: bool = False,
    refresh: bool = False,
) -> dict:
    """1年分の全カテゴリをエクスポートする。refresh=True は日次の増分用で、各カテゴリが「自分の flat に無いレース」を
    GCS から追記する（基準レース集合による完全性チェックを外す。既定のままだと race_result は自分の flat が基準になり、新しいレースが永久に入らない）。
    force=True は、対象カテゴリの既存ファイルを消して GCS から作り直す
    （GCS 側のデータを直した後など、レース数が同じで中身だけが変わった場合。増分チェックは中身の変更を検知しない）。"""
    from src.scraper.storage import HybridStorage

    if storage is None:
        storage = HybridStorage()

    cats = categories or (FLAT_CATEGORIES + ["race_pair_odds"])
    t0 = time.time()
    all_stats: list[dict] = []

    local_rr = OUTPUT_DIR / year / "race_result_flat.parquet"
    # 基準レース数は、消す前のローカル race_result_flat から取る（force で作り直すときも同じレース集合で作る）
    base_ids = frozenset(_existing_race_ids(local_rr)) if (force and local_rr.exists()) else None
    if force:
        for cat in cats:
            for suffix in ("flat", "race"):
                f = OUTPUT_DIR / year / f"{cat}_{suffix}.parquet"
                if f.exists():
                    f.unlink()
                    logger.info("[%s/%s] --force: 既存の %s を削除して作り直す", year, cat, f.name)
    if refresh and not force:
        expected_keys = frozenset()
        logger.info("[%s] --refresh: 各カテゴリが自分の flat に無いレースを GCS から追記する", year)
    elif base_ids is not None:
        expected_keys = base_ids
        logger.info("[%s] 基準レース数 (force 前のローカル race_result_flat): %d", year, len(expected_keys))
    elif local_rr.exists():
        expected_keys = frozenset(_existing_race_ids(local_rr))
        logger.info(
            "[%s] 基準レース数 (ローカル race_result_flat): %d",
            year, len(expected_keys),
        )
    else:
        base_blobs = storage.batch_list_blobs("race_result", year)
        expected_keys = frozenset(k for k in base_blobs if is_jra_race(k))
        logger.info("[%s] 基準レース数 (GCS race_result): %d", year, len(expected_keys))

    # メタ: 増分 export があるときだけ。ローカル Parquet から組むことを最優先し GCS shutuba を避ける。
    needs_meta = bool(refresh and not force)
    out_dir = OUTPUT_DIR / year
    for cat in cats:
        fp = out_dir / f"{cat}_flat.parquet"
        if not fp.exists():
            needs_meta = True
            break
        if cat in COMPLETENESS_EXEMPT:
            continue
        ex = _existing_race_ids(fp)
        if expected_keys and not expected_keys <= ex:
            needs_meta = True
            break

    if needs_meta:
        meta_index = _build_meta_index_from_race_result_flat(year)
        if not meta_index:
            meta_index = _build_meta_index(year, storage)
        elif refresh and not force:
            new_ids = [k for k in storage.batch_list_blobs("race_result", year) if is_jra_race(k) and k not in meta_index]
            logger.info("[%s] --refresh: 新しいレース %d 件のメタを出馬表から補う（%d 件）", year, len(new_ids),
                        _add_shutuba_meta(storage, new_ids, meta_index))
    else:
        meta_index = {}
        logger.info("[%s] 全カテゴリ export 済み → メタインデックス構築スキップ", year)

    for cat in cats:
        stats = export_category_chunked(
            year, cat, storage, meta_index, expected_keys=expected_keys,
        )
        all_stats.append(stats)
        if refresh and not force and cat == "race_result" and not stats.get("skipped"):
            # 追記した新しいレースのメタ（日付・場・距離など）を、後続カテゴリが使えるようにする
            meta_index = _build_meta_index_from_race_result_flat(year) or meta_index

        flat_info = f"flat={stats.get('flat_rows', 0)}"
        race_info = f"race={stats.get('race_rows', 0)}"
        size_info = ""
        if stats.get("flat_size_mb"):
            size_info += f" flat={stats['flat_size_mb']}MB"
        if stats.get("race_size_mb"):
            size_info += f" race={stats['race_size_mb']}MB"
        logger.info("[%s/%s] 完了: %s, %s%s", year, cat, flat_info, race_info, size_info)

    if "race_result" in cats:
        try:
            _b = bridge_on_time_results(year)
            all_stats.append({"category": "race_result_bridge_on_time", "races": _b.get("bridged_races", 0), "flat_rows": 0, "race_rows": 0, **_b})
        except Exception as exc:  # noqa: BLE001  橋渡しの失敗で日次エクスポート全体を止めない
            logger.warning("[%s] 速報の橋渡しに失敗: %s", year, exc)

    elapsed = time.time() - t0

    report = {
        "year": year,
        "elapsed_seconds": round(elapsed, 1),
        "categories": all_stats,
        "total_flat_rows": sum(s.get("flat_rows", 0) for s in all_stats),
        "total_race_rows": sum(s.get("race_rows", 0) for s in all_stats),
    }

    report_path = OUTPUT_DIR / year / "_export_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    logger.info("[%s] エクスポート完了: %.1fs, flat=%d, race=%d → %s",
                year, elapsed,
                report["total_flat_rows"], report["total_race_rows"],
                report_path)

    return report


def main() -> int:
    script_basic_config()
    ap = argparse.ArgumentParser(description="Parquet テーブルエクスポート")
    ap.add_argument("--years", required=True,
                    help="対象年 (カンマ区切り: 2022,2023,2024)")
    ap.add_argument("--categories",
                    help="対象カテゴリ (カンマ区切り、省略時: 全12カテゴリ)")
    ap.add_argument("--workers", type=int, default=6, help="並列数 (default: 6)")
    ap.add_argument("--refresh", action="store_true",
                    help="日次の増分用: 各カテゴリが自分の flat に無いレースを GCS から追記する（基準レースの完全性チェックを外す）")
    ap.add_argument("--force", action="store_true",
                    help="対象カテゴリの既存ファイルを消して GCS から作り直す（レース数が同じで中身だけ変わったとき）")
    args = ap.parse_args()

    years = [y.strip() for y in args.years.split(",")]
    cats = [c.strip() for c in args.categories.split(",")] if args.categories else None

    from src.scraper.storage import HybridStorage
    storage = HybridStorage()

    for year in years:
        print(f"\n{'='*60}")
        print(f"  {year}年 テーブルエクスポート")
        print(f"{'='*60}")

        report = export_year(year, categories=cats, storage=storage, workers=args.workers, force=args.force, refresh=args.refresh)

        print(f"\n  フラット行: {report['total_flat_rows']:,}")
        print(f"  レース行: {report['total_race_rows']:,}")
        print(f"  所要時間: {report['elapsed_seconds']}s")
        print()

        for s in report["categories"]:
            if s.get("skipped"):
                print(f"  {s['category']:25s}  SKIPPED: {s.get('reason','')}")
                continue
            flat_mb = s.get("flat_size_mb", 0)
            race_mb = s.get("race_size_mb", 0)
            print(f"  {s['category']:25s}  races={s['races']:>5d}"
                  f"  flat={s.get('flat_rows', 0):>7,}"
                  f"  race={s.get('race_rows', 0):>5,}"
                  f"  flat={flat_mb:>6.1f}MB  race={race_mb:>6.1f}MB")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
