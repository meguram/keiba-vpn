"""race_result_flat.parquet の surface/distance 欠損修復。"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.pipeline.megu_index.compute import (
    TABLES_DIR,
    _enrich_flat_from_gcs_race_result,
)

logger = logging.getLogger(__name__)


def _needs_metadata_repair(df: pd.DataFrame) -> pd.Series:
    dist = pd.to_numeric(df.get("distance"), errors="coerce").fillna(0)
    surf = df.get("surface")
    bad_surf = surf.isna() if surf is not None else pd.Series(True, index=df.index)
    if surf is not None:
        bad_surf = bad_surf | ~surf.astype(str).isin(["芝", "ダート"])
    return (dist <= 0) | bad_surf


def is_obstacle_race_name(race_name: str | None) -> bool:
    """障害・ジャンプ系レース名（めぐ指数対象外）。"""
    n = str(race_name or "")
    markers = ("障害", "ジャンプ", "J・G", "JGI", "J.G", "グランドジャンプ", "JS(", "JS（")
    return any(m in n for m in markers)


def classify_bad_flat_races(df: pd.DataFrame) -> dict:
    """メタ欠損レースを障害（skip可）と要 backfill に分類。"""
    bad = _needs_metadata_repair(df)
    if not bad.any():
        return {
            "bad_races": 0,
            "obstacle_races": 0,
            "backfill_races": 0,
            "backfill_race_ids": [],
            "obstacle_race_ids": [],
        }
    bad_ids = sorted(df.loc[bad, "race_id"].unique())
    obstacle_ids: list[str] = []
    backfill_ids: list[str] = []
    for rid in bad_ids:
        row = df[df["race_id"] == rid].iloc[0]
        name = row.get("race_name")
        if is_obstacle_race_name(name):
            obstacle_ids.append(str(rid))
        else:
            backfill_ids.append(str(rid))
    return {
        "bad_races": len(bad_ids),
        "obstacle_races": len(obstacle_ids),
        "backfill_races": len(backfill_ids),
        "backfill_race_ids": backfill_ids,
        "obstacle_race_ids": obstacle_ids,
    }


def audit_flat_metadata(year: int) -> dict:
    """年別 flat のメタデータ欠損を集計。"""
    p = TABLES_DIR / str(year) / "race_result_flat.parquet"
    if not p.exists():
        return {"year": year, "exists": False, "bad_rows": 0, "bad_races": 0, "total_races": 0}
    df = pd.read_parquet(p)
    bad = _needs_metadata_repair(df)
    classified = classify_bad_flat_races(df)
    return {
        "year": year,
        "exists": True,
        "path": str(p),
        "total_rows": len(df),
        "total_races": int(df["race_id"].nunique()) if "race_id" in df.columns else 0,
        "bad_rows": int(bad.sum()),
        "bad_races": int(df.loc[bad, "race_id"].nunique()) if bad.any() else 0,
        **classified,
    }


def repair_race_result_flat_metadata(year: int, *, dry_run: bool = False) -> dict:
    """
    lap_times 推定・venue 参照・GCS race_card で surface/distance を補完し parquet を上書き。

    Returns:
        {"year", "repaired_rows", "remaining_bad_rows", "remaining_bad_races"}
    """
    p = TABLES_DIR / str(year) / "race_result_flat.parquet"
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_parquet(p)
    before_bad = _needs_metadata_repair(df)
    if not before_bad.any():
        return {
            "year": year,
            "repaired_rows": 0,
            "remaining_bad_rows": 0,
            "remaining_bad_races": 0,
        }

    enriched = _enrich_flat_from_gcs_race_result(df, df_ref=df)
    after_bad = _needs_metadata_repair(enriched)
    repaired_rows = int((before_bad & ~after_bad).sum())

    if not dry_run:
        enriched.to_parquet(p, index=False, engine="pyarrow")
        logger.info(
            "repair flat %d: repaired_rows=%d remaining_bad_races=%d",
            year,
            repaired_rows,
            int(enriched.loc[after_bad, "race_id"].nunique()) if after_bad.any() else 0,
        )

    return {
        "year": year,
        "repaired_rows": repaired_rows,
        "remaining_bad_rows": int(after_bad.sum()),
        "remaining_bad_races": int(enriched.loc[after_bad, "race_id"].nunique()) if after_bad.any() else 0,
        "dry_run": dry_run,
    }
