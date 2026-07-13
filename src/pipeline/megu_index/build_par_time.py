"""
クラス別 par_time（v2）の推定と megu_par_time への投入。

学習: 2020–2025 の race_result_flat、2着馬の
  finish_time - Δpace - Δtrack - Δweight
をセル平均。Δlevel は含めない。

Usage:
  KEIBA_ENV=stg python -m src.pipeline.megu_index.build_par_time --model-version v2
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

from src.pipeline.megu_index.class_bucket import par_class_bucket
from src.pipeline.megu_index.compute import (
    CUSHION_DIR,
    TABLES_DIR,
    _load_cushion,
    _load_result_flat,
    _parse_lap_times,
    _select_split_point,
)

logger = logging.getLogger(__name__)

MIN_SAMPLES = 20
TRAIN_YEARS = list(range(2020, 2026))
PARAM_FALLBACK_VERSION = "v1"


def _load_regression_params(session, model_version: str) -> dict[str, float]:
    rows = session.execute(
        text("SELECT param_name, param_value FROM megu_regression_params WHERE model_version=:mv"),
        {"mv": model_version},
    ).fetchall()
    params = {r[0]: float(r[1]) for r in rows}
    if params:
        return params
    rows = session.execute(
        text("SELECT param_name, param_value FROM megu_regression_params WHERE model_version=:mv"),
        {"mv": PARAM_FALLBACK_VERSION},
    ).fetchall()
    return {r[0]: float(r[1]) for r in rows}


def _load_v1_front_split(session) -> pd.DataFrame:
    rows = session.execute(
        text("""
            SELECT distance, course, surface, track_condition, par_front_split_sec
            FROM megu_par_time WHERE model_version=:mv AND par_front_split_sec IS NOT NULL
        """),
        {"mv": PARAM_FALLBACK_VERSION},
    ).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["distance", "course", "surface", "track_condition", "par_front_split_sec"])
    df = df.rename(columns={"course": "direction", "track_condition": "track_cat"})
    df["par_front_split_sec"] = pd.to_numeric(df["par_front_split_sec"], errors="coerce")
    return df


def _attach_splits(df: pd.DataFrame) -> pd.DataFrame:
    splits = []
    for _, row in df[["race_id", "distance", "lap_times"]].drop_duplicates("race_id").iterrows():
        lap_dict = _parse_lap_times(row["lap_times"], int(row["distance"]))
        if not lap_dict:
            splits.append({"race_id": row["race_id"], "front_split_sec": np.nan})
            continue
        sp = _select_split_point(int(row["distance"]), list(lap_dict.keys()))
        splits.append({
            "race_id": row["race_id"],
            "front_split_sec": lap_dict[sp] if sp is not None else np.nan,
        })
    return df.merge(pd.DataFrame(splits), on="race_id", how="left")


def _prepare_training_frame(years: list[int], params: dict[str, float]):
    beta_pace = params.get("beta_pace", 0.0)
    beta_track = params.get("beta_track", 0.0)
    beta_weight = params.get("beta_weight", 0.0)
    tsi_mean = params.get("tsi_mean", 0.0)

    dfs = []
    for yr in years:
        df_y = _load_result_flat(yr)
        if df_y.empty:
            continue
        df_cushion = _load_cushion(yr)
        if not df_cushion.empty:
            df_y["date_str"] = pd.to_datetime(df_y["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            df_y["venue_code"] = df_y["venue_code"].astype(str).str.strip()
            c_merge = df_cushion[["date_str", "venue_code", "cushion_value", "dirt_moisture_goal"]].drop_duplicates(
                subset=["date_str", "venue_code"]
            )
            df_y = df_y.merge(c_merge, on=["date_str", "venue_code"], how="left")
            df_y["tsi_raw"] = np.where(
                df_y["surface"] == "芝",
                df_y["cushion_value"].fillna(tsi_mean),
                -df_y["dirt_moisture_goal"].fillna(-tsi_mean),
            )
        else:
            df_y["tsi_raw"] = tsi_mean
        dfs.append(df_y)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df = df[df["surface"].isin(["芝", "ダート"])].copy()
    df["finish_time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")
    df = df[df["finish_time_sec"].notna() & (df["finish_time_sec"] > 0)]
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df = df[df["distance"] > 0]

    df = _attach_splits(df)
    track_map = {"良": "良", "稍重": "稍重", "重": "重・不良", "不良": "重・不良"}
    df["track_cat"] = df["track_condition"].map(track_map).fillna("良")
    df["direction"] = df["direction"].fillna("").astype(str)

    df["sex"] = df["sex_age"].astype(str).str.extract(r"^(牡|牝|セン)", expand=False).fillna("牡")
    df["base_weight"] = np.where(df["sex"] == "牝", 53.0, 55.0)
    df["jockey_weight_num"] = pd.to_numeric(df["jockey_weight"], errors="coerce")
    df["weight_dev"] = df["jockey_weight_num"] - df["base_weight"]
    df["dist_scale"] = df["distance"] / 2000.0
    df["tsi_normalized"] = df["tsi_raw"].fillna(tsi_mean) - tsi_mean

    df["class_bucket"] = df.apply(
        lambda r: par_class_bucket(r.get("grade"), r.get("race_name"), r.get("race_class")),
        axis=1,
    )

    finish_col = "finish_position" if "finish_position" in df.columns else "finish_pos"
    df["finish_pos_num"] = pd.to_numeric(df[finish_col], errors="coerce")

    return df, beta_pace, beta_track, beta_weight, tsi_mean, finish_col


def _second_place_rows(df: pd.DataFrame, finish_col: str) -> pd.DataFrame:
    df_2 = df[df["finish_pos_num"] == 2].copy()
    df_1 = df[df["finish_pos_num"] == 1][["race_id", "finish_time_sec"]].rename(
        columns={"finish_time_sec": "time_1st"}
    )
    if df_2.empty:
        return df_2
    have_2nd = set(df_2["race_id"])
    df_1_only = df[df["finish_pos_num"] == 1].copy()
    df_1_only = df_1_only[~df_1_only["race_id"].isin(have_2nd)]
    return pd.concat([df_2, df_1_only], ignore_index=True)


def build_par_time_cells(
    df: pd.DataFrame,
    *,
    beta_pace: float,
    beta_track: float,
    beta_weight: float,
    df_split: pd.DataFrame,
    finish_col: str,
    min_samples: int = MIN_SAMPLES,
) -> pd.DataFrame:
    """2着基準の adjusted_time（level なし）から par セルを構築。"""
    if df_split is not None and not df_split.empty:
        df = df.merge(
            df_split[["distance", "surface", "track_cat", "par_front_split_sec"]],
            on=["distance", "surface", "track_cat"],
            how="left",
        )
    else:
        df["par_front_split_sec"] = np.nan

    df["front_split_dev"] = df["front_split_sec"] - df["par_front_split_sec"]
    df["delta_pace_sec"] = beta_pace * df["front_split_dev"].fillna(0)
    df["delta_track_sec"] = -beta_track * df["tsi_normalized"].fillna(0)
    df["delta_weight_sec"] = beta_weight * df["weight_dev"].fillna(0) * df["dist_scale"].fillna(1)
    df["adjusted_no_level"] = (
        df["finish_time_sec"] - df["delta_pace_sec"] - df["delta_track_sec"] - df["delta_weight_sec"]
    )

    df_2nd = _second_place_rows(df, finish_col)
    if df_2nd.empty:
        return pd.DataFrame()

    records: list[pd.DataFrame] = []

    # L1: class-specific
    g1 = (
        df_2nd.groupby(["distance", "direction", "surface", "track_cat", "class_bucket"], as_index=False)
        .agg(
            par_time_sec=("adjusted_no_level", "mean"),
            par_front_split_sec=("front_split_sec", "mean"),
            sample_count=("race_id", "nunique"),
        )
    )
    g1 = g1[g1["sample_count"] >= min_samples].copy()
    records.append(g1)

    # L3 pool: no class
    g3 = (
        df_2nd.groupby(["distance", "direction", "surface", "track_cat"], as_index=False)
        .agg(
            par_time_sec=("adjusted_no_level", "mean"),
            par_front_split_sec=("front_split_sec", "mean"),
            sample_count=("race_id", "nunique"),
        )
    )
    g3 = g3[g3["sample_count"] >= min_samples].copy()
    g3["class_bucket"] = ""
    records.append(g3)

    out = pd.concat(records, ignore_index=True)
    out["par_time_sec"] = out["par_time_sec"].round(3)
    out["par_front_split_sec"] = out["par_front_split_sec"].round(3)
    return out.drop_duplicates(
        subset=["distance", "direction", "surface", "track_cat", "class_bucket"]
    )


def upsert_par_time(engine, df_par: pd.DataFrame, model_version: str) -> int:
    from src.db.models import MeguParTime  # noqa: F401

    if df_par.empty:
        return 0

    rows = []
    for _, r in df_par.iterrows():
        rows.append({
            "distance": int(r["distance"]),
            "course": str(r["direction"]),
            "surface": str(r["surface"]),
            "track_condition": str(r["track_cat"]),
            "class_bucket": str(r.get("class_bucket") or ""),
            "par_time_sec": float(r["par_time_sec"]),
            "par_front_split_sec": float(r["par_front_split_sec"]) if pd.notna(r.get("par_front_split_sec")) else None,
            "sample_count": int(r["sample_count"]),
            "model_version": model_version,
        })

    with engine.begin() as conn:
        stmt = insert(MeguParTime).values(rows)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_megu_par_time",
            set_={
                "par_time_sec": stmt.excluded.par_time_sec,
                "par_front_split_sec": stmt.excluded.par_front_split_sec,
                "sample_count": stmt.excluded.sample_count,
            },
        )
        conn.execute(stmt)
    return len(rows)


def run_build(*, model_version: str = "v2", years: list[int] | None = None, min_samples: int = MIN_SAMPLES) -> dict:
    from src.db.session import get_session, init_engine

    from src.db.session import init_engine

    init_engine()
    years = years or TRAIN_YEARS

    with get_session() as session:
        params = _load_regression_params(session, model_version)
        df_split = _load_v1_front_split(session)

    if not params:
        raise RuntimeError("megu_regression_params が見つかりません。NB-02 を先に実行してください。")

    df, beta_pace, beta_track, beta_weight, _tsi, finish_col = _prepare_training_frame(years, params)
    if df.empty:
        raise RuntimeError("学習データが空です")

    logger.info("学習行数: %d (%d–%d)", len(df), min(years), max(years))
    par_cells = build_par_time_cells(
        df,
        beta_pace=beta_pace,
        beta_track=beta_track,
        beta_weight=beta_weight,
        df_split=df_split,
        finish_col=finish_col,
        min_samples=min_samples,
    )
    logger.info("par_time セル: %d (class別=%d, プール=%d)",
                len(par_cells),
                int((par_cells["class_bucket"] != "").sum()),
                int((par_cells["class_bucket"] == "").sum()))

    from src.db.session import init_engine

    engine = init_engine()
    n = upsert_par_time(engine, par_cells, model_version)

    meta = {
        "model_version": model_version,
        "train_years": years,
        "n_cells": n,
        "min_samples": min_samples,
        "params_source": PARAM_FALLBACK_VERSION,
    }
    out_path = Path(__file__).resolve().parents[3] / "config" / "megu_par_time_v2_meta.json"
    out_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="クラス別 par_time 推定・DB投入")
    parser.add_argument("--model-version", default="v2")
    parser.add_argument("--year-start", type=int, default=2020)
    parser.add_argument("--year-end", type=int, default=2025)
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES,
                        help="par セル推定の最低レース数（既定 20。希少距離は 5 程度で再推定）")
    args = parser.parse_args()
    years = list(range(args.year_start, args.year_end + 1))
    meta = run_build(model_version=args.model_version, years=years, min_samples=args.min_samples)
    print("完了:", meta)


if __name__ == "__main__":
    main()
