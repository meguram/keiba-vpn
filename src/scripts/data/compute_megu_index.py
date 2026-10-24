"""
めぐ指数計算・DB投入スクリプト（STG版）

STG DBの2026年データを使い、以下の簡易版めぐ指数を計算してDBに格納する。

  megu_index = 100 + (par_time - finish_time_sec) × 10

par_time は (venue, surface, distance, track_condition) セルごとの中央値走破タイム。
補正（Δpace, Δtrack, Δweight, Δlevel）はSTGデータ不足のため 0 として扱う。

Usage:
    python3 -m src.scripts.data.compute_megu_index [--model-version stg-v1] [--year 2026]
"""

from __future__ import annotations

import argparse
import logging
import sys
from decimal import Decimal

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_VERSION = "stg-v1"


def fetch_data(engine) -> pd.DataFrame:
    """race_results + races から走破タイム入りの行を取得。"""
    sql = """
        SELECT
            rr.race_id,
            rr.horse_id,
            rr.finish_pos,
            rr.finish_time_sec,
            rr.last_3f_sec,
            r.venue,
            r.surface,
            r.distance,
            r.track_condition,
            r.race_date,
            r.grade
        FROM race_results rr
        JOIN races r ON r.race_id = rr.race_id
        WHERE rr.finish_time_sec IS NOT NULL
          AND rr.finish_pos IS NOT NULL
          AND rr.finish_pos BETWEEN 1 AND 18
          AND r.surface IN ('芝', 'ダート')
          AND r.distance > 0
          AND (r.track_condition IS NOT NULL)
        ORDER BY r.race_date, rr.race_id, rr.finish_pos
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    logger.info("取得行数: %d", len(df))
    return df


def compute_par_times(df: pd.DataFrame) -> pd.DataFrame:
    """セル別基準タイム（中央値）を計算。"""
    group_cols = ["venue", "surface", "distance", "track_condition"]
    par = (
        df.groupby(group_cols)["finish_time_sec"]
        .agg(par_time_sec="median", sample_count="count")
        .reset_index()
    )
    # 最低サンプル数フィルタ
    par = par[par["sample_count"] >= 3].copy()
    logger.info("基準タイムセル数: %d", len(par))
    return par


def compute_megu_index(df: pd.DataFrame, par: pd.DataFrame) -> pd.DataFrame:
    """めぐ指数を計算。"""
    group_cols = ["venue", "surface", "distance", "track_condition"]
    merged = df.merge(par[group_cols + ["par_time_sec"]], on=group_cols, how="left")

    # par_time が存在しないセルは除外
    merged = merged.dropna(subset=["par_time_sec"])

    # 全補正量は 0（STGデータ不足のため）
    merged["delta_pace_sec"] = 0.0
    merged["delta_track_sec"] = 0.0
    merged["delta_weight_sec"] = 0.0
    merged["delta_level_sec"] = 0.0

    merged["adjusted_time_sec"] = merged["finish_time_sec"]  # 補正なし
    merged["megu_index"] = (
        100.0 + (merged["par_time_sec"] - merged["finish_time_sec"]) * 10.0
    ).round(1)

    logger.info("めぐ指数計算行数: %d (par未設定除外後)", len(merged))
    logger.info("  megu_index 統計: mean=%.1f, std=%.1f, min=%.1f, max=%.1f",
                merged["megu_index"].mean(),
                merged["megu_index"].std(),
                merged["megu_index"].min(),
                merged["megu_index"].max())
    return merged


def upsert_par_times(engine, par: pd.DataFrame, model_version: str) -> int:
    """megu_par_time テーブルに基準タイムを投入。"""
    from src.db.models import MeguParTime

    rows = []
    for _, r in par.iterrows():
        rows.append({
            "distance": int(r["distance"]),
            "course": str(r["venue"]),
            "surface": str(r["surface"]),
            "track_condition": str(r["track_condition"]),
            "par_time_sec": float(r["par_time_sec"]),
            "sample_count": int(r["sample_count"]),
            "model_version": model_version,
        })

    if not rows:
        return 0

    with engine.begin() as conn:
        stmt = insert(MeguParTime).values(rows)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_megu_par_time",
            set_={
                "par_time_sec": stmt.excluded.par_time_sec,
                "sample_count": stmt.excluded.sample_count,
            },
        )
        conn.execute(stmt)
    logger.info("megu_par_time 投入: %d セル", len(rows))
    return len(rows)


def upsert_megu_index(engine, df_result: pd.DataFrame, model_version: str) -> int:
    """megu_index テーブルにバルク投入（1000件ずつ）。"""
    from src.db.models import MeguIndex

    cols = [
        "race_id", "horse_id", "finish_time_sec", "par_time_sec",
        "delta_pace_sec", "delta_track_sec", "delta_weight_sec", "delta_level_sec",
        "adjusted_time_sec", "megu_index",
    ]
    inserted = 0
    chunk_size = 1000

    for i in range(0, len(df_result), chunk_size):
        chunk = df_result.iloc[i:i + chunk_size]
        rows = []
        for _, r in chunk.iterrows():
            rows.append({
                "race_id": str(r["race_id"]),
                "horse_id": str(r["horse_id"]),
                "finish_time_sec": float(r["finish_time_sec"]),
                "par_time_sec": float(r["par_time_sec"]),
                "delta_pace_sec": 0.0,
                "delta_track_sec": 0.0,
                "delta_weight_sec": 0.0,
                "delta_level_sec": 0.0,
                "adjusted_time_sec": float(r["adjusted_time_sec"]),
                "megu_index": float(r["megu_index"]),
                "model_version": model_version,
            })
        with engine.begin() as conn:
            stmt = insert(MeguIndex).values(rows)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_megu_index",
                set_={
                    "finish_time_sec": stmt.excluded.finish_time_sec,
                    "par_time_sec": stmt.excluded.par_time_sec,
                    "adjusted_time_sec": stmt.excluded.adjusted_time_sec,
                    "megu_index": stmt.excluded.megu_index,
                },
            )
            conn.execute(stmt)
        inserted += len(rows)
        if i % 5000 == 0:
            logger.info("  投入済み %d / %d", inserted, len(df_result))

    logger.info("megu_index 投入完了: %d 行", inserted)
    return inserted


def main():
    parser = argparse.ArgumentParser(description="めぐ指数計算・DB投入")
    parser.add_argument("--model-version", default=MODEL_VERSION)
    args = parser.parse_args()

    from src.db.session import init_engine
    engine = init_engine()

    logger.info("=== Step 1: 走破タイムデータ取得 ===")
    df = fetch_data(engine)
    if len(df) == 0:
        logger.error("走破タイムデータが0件です。ETLが未実行の可能性があります。")
        sys.exit(1)

    logger.info("=== Step 2: 基準タイム計算 ===")
    par = compute_par_times(df)

    logger.info("=== Step 3: めぐ指数計算 ===")
    df_result = compute_megu_index(df, par)

    logger.info("=== Step 4: 基準タイムDB投入 ===")
    upsert_par_times(engine, par, args.model_version)

    logger.info("=== Step 5: めぐ指数DB投入 ===")
    upsert_megu_index(engine, df_result, args.model_version)

    logger.info("=== 完了 ===")
    logger.info("基準タイムセル: %d, 指数計算行: %d", len(par), len(df_result))


if __name__ == "__main__":
    main()
