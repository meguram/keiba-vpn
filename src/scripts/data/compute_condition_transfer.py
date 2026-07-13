"""
条件転換係数（megu_condition_transfer）計算スクリプト

連続レースで条件が変化した際の megu_index 差を秒換算（÷10）して集計。
delta_mean 列 = adjusted_time 空間への加算量（秒）。
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

from src.pipeline.megu_index.common import MEGU_POINTS_PER_SEC, dist_band

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_VERSION = "v1"
MIN_SAMPLES = 30


def fetch_megu_history(engine) -> pd.DataFrame:
    """馬ごとの megu 履歴（adjusted_time 含む）。"""
    sql = """
        SELECT
            mi.horse_id,
            mi.race_id,
            mi.megu_index,
            mi.adjusted_time_sec,
            mi.par_time_sec,
            r.race_date,
            r.surface,
            r.distance
        FROM megu_index mi
        JOIN races r ON r.race_id = mi.race_id
        WHERE mi.megu_index IS NOT NULL
          AND mi.computation_status = 'valid'
          AND r.surface IN ('芝', 'ダート')
          AND r.distance > 0
        ORDER BY mi.horse_id, r.race_date ASC, mi.race_id ASC
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    logger.info("megu 履歴取得: %d 行, %d 頭", len(df), df["horse_id"].nunique())
    return df


def compute_transfer_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """連続レース対の megu 差を秒換算して収集（1点=0.1秒）。"""
    records = []
    for horse_id, grp in df.groupby("horse_id", sort=False):
        races = grp.reset_index(drop=True)
        for i in range(1, len(races)):
            prev = races.iloc[i - 1]
            curr = races.iloc[i]

            sf = prev["surface"]
            st = curr["surface"]
            db_from = dist_band(int(prev["distance"]))
            db_to = dist_band(int(curr["distance"]))

            if sf == st and db_from == db_to:
                continue

            delta_sec = (float(curr["megu_index"]) - float(prev["megu_index"])) / MEGU_POINTS_PER_SEC
            records.append({
                "surface_from": sf,
                "surface_to": st,
                "dist_band_from": db_from,
                "dist_band_to": db_to,
                "delta_sec": delta_sec,
            })

    result = pd.DataFrame(records)
    logger.info("条件転換対: %d 件", len(result))
    return result


def aggregate_transfer_coeffs(deltas: pd.DataFrame, min_samples: int) -> pd.DataFrame:
    group_cols = ["surface_from", "surface_to", "dist_band_from", "dist_band_to"]
    if deltas.empty:
        return pd.DataFrame()

    agg = (
        deltas.groupby(group_cols)["delta_sec"]
        .agg(
            delta_mean="mean",
            delta_std="std",
            sample_count="count",
        )
        .reset_index()
    )
    agg["delta_mean"] = agg["delta_mean"].round(3)
    agg["delta_std"] = agg["delta_std"].round(3)

    before = len(agg)
    agg = agg[agg["sample_count"] >= min_samples].copy()
    logger.info(
        "転換係数: %d ペア（最低 %d サンプル未満 %d ペア除外）",
        len(agg), min_samples, before - len(agg),
    )

    for _, row in agg.iterrows():
        logger.info(
            "  %s %s → %s %s : Δsec=%.3f ± %.3f (n=%d)",
            row["surface_from"], row["dist_band_from"],
            row["surface_to"], row["dist_band_to"],
            row["delta_mean"], row["delta_std"] if not np.isnan(row["delta_std"]) else 0,
            row["sample_count"],
        )
    return agg


def upsert_transfer_coeffs(engine, agg: pd.DataFrame, model_version: str) -> int:
    from src.db.models import MeguConditionTransfer  # noqa: F401

    if agg.empty:
        logger.warning("upsert 対象データが0件です")
        return 0

    rows = []
    for _, r in agg.iterrows():
        rows.append({
            "surface_from": str(r["surface_from"]),
            "surface_to": str(r["surface_to"]),
            "dist_band_from": str(r["dist_band_from"]),
            "dist_band_to": str(r["dist_band_to"]),
            "delta_mean": float(r["delta_mean"]),
            "delta_std": None if np.isnan(r["delta_std"]) else float(r["delta_std"]),
            "sample_count": int(r["sample_count"]),
            "model_version": model_version,
        })

    with engine.begin() as conn:
        stmt = insert(MeguConditionTransfer).values(rows)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_megu_condition_transfer",
            set_={
                "delta_mean": stmt.excluded.delta_mean,
                "delta_std": stmt.excluded.delta_std,
                "sample_count": stmt.excluded.sample_count,
            },
        )
        conn.execute(stmt)

    logger.info("megu_condition_transfer 投入: %d ペア", len(rows))
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="条件転換係数計算・DB投入（秒ベース）")
    parser.add_argument("--model-version", default=MODEL_VERSION)
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES)
    args = parser.parse_args()

    from src.db.session import init_engine
    engine = init_engine()

    logger.info("=== Step 1: megu 履歴取得 ===")
    df = fetch_megu_history(engine)
    if df.empty:
        logger.error("megu 履歴が0件です。compute_megu_index を先に実行してください。")
        sys.exit(1)

    logger.info("=== Step 2: 条件転換 delta_sec 計算 ===")
    deltas = compute_transfer_deltas(df)
    if deltas.empty:
        logger.warning("条件転換対が0件でした。")
        sys.exit(0)

    logger.info("=== Step 3: 転換係数集計 ===")
    agg = aggregate_transfer_coeffs(deltas, args.min_samples)

    logger.info("=== Step 4: DB 投入 ===")
    upsert_transfer_coeffs(engine, agg, args.model_version)
    logger.info("=== 完了: %d ペアの転換係数を更新 ===", len(agg))


if __name__ == "__main__":
    main()
