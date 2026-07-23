#!/usr/bin/env python3
"""
短期データでめぐ指数品質を before/after 比較する。

Usage:
  KEIBA_ENV=stg python3 -m src.scripts.data.validate_megu_quality \\
      --dates 2026-07-18,2026-07-19

  # 既存 parquet を before、再計算を after として比較
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.megu_index.common import adjusted_time_to_megu
from src.pipeline.megu_index.compute import (
    MODEL_VERSION,
    TABLES_DIR,
    _load_cushion,
    _load_history_flat_for_level,
    _load_par_time,
    _load_regression_params,
    _load_result_flat,
    compute_for_dataframe,
)
from src.pipeline.megu_index.quality_check import (
    compare_quality,
    passes_quality_gates,
    summarize_megu_quality,
)


def _load_existing_megu(year: int) -> pd.DataFrame:
    p = TABLES_DIR / str(year) / "megu_index_flat.parquet"
    if not p.is_file():
        return pd.DataFrame()
    return pd.read_parquet(p)


def _load_flat_for_dates(dates: list[str]) -> pd.DataFrame:
    frames = []
    years = sorted({int(d[:4]) for d in dates})
    for y in years:
        df = _load_result_flat(y)
        if df.empty:
            continue
        df["date_str"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        iso_dates = {f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in dates}
        frames.append(df[df["date_str"].isin(iso_dates)])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _prepare_flat(df_flat: pd.DataFrame, params: dict) -> pd.DataFrame:
    if df_flat.empty:
        return df_flat
    year = int(str(df_flat["date_str"].iloc[0])[:4])
    df = df_flat.copy()
    df_cushion = _load_cushion(year)
    if not df_cushion.empty:
        df["venue_code"] = df["venue_code"].astype(str).str.strip()
        c_merge = df_cushion[["date_str", "venue_code", "cushion_value", "dirt_moisture_goal"]].drop_duplicates(
            subset=["date_str", "venue_code"]
        )
        df = df.merge(c_merge, on=["date_str", "venue_code"], how="left")
        tsi_mean = params.get("tsi_mean", 0.0)
        df["tsi_raw"] = np.where(
            df["surface"] == "芝",
            df["cushion_value"].fillna(tsi_mean),
            -df["dirt_moisture_goal"].fillna(-tsi_mean),
        )
    return df


def _merge_meta(computed: pd.DataFrame, flat: pd.DataFrame) -> pd.DataFrame:
    meta_cols = [
        "race_id", "horse_id", "grade", "race_class", "race_name",
        "finish_position", "distance", "surface", "date", "date_str",
    ]
    meta = flat[[c for c in meta_cols if c in flat.columns]].drop_duplicates(["race_id", "horse_id"])
    return computed.merge(meta, on=["race_id", "horse_id"], how="left")


def _baseline_from_existing(existing: pd.DataFrame, flat: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return pd.DataFrame()
    meta_cols = [
        "race_id", "horse_id", "grade", "race_class", "race_name",
        "finish_position", "distance", "surface",
    ]
    meta = flat[[c for c in meta_cols if c in flat.columns]].drop_duplicates(["race_id", "horse_id"])
    base = existing.merge(meta, on=["race_id", "horse_id"], how="left")
    base["megu_recalc"] = base.apply(
        lambda r: adjusted_time_to_megu(r.get("adjusted_time_sec"), r.get("par_time_final")),
        axis=1,
    )
    return base


def run_validation(dates: list[str], *, model_version: str = MODEL_VERSION) -> dict:
    from src.db.session import get_session, init_engine

    init_engine()
    with get_session() as session:
        params = _load_regression_params(session, model_version)
        df_par = _load_par_time(session, model_version)

    if not params:
        raise RuntimeError("megu_regression_params が未登録です")

    flat = _load_flat_for_dates(dates)
    if flat.empty:
        raise RuntimeError(f"race_result_flat に対象日がありません: {dates}")

    flat = _prepare_flat(flat, params)
    date_str = sorted(flat["date_str"].unique())[-1]
    df_hist = _load_history_flat_for_level(date_str)
    df_year_ref = _load_result_flat(int(date_str[:4]))

    computed = compute_for_dataframe(
        flat, params, df_par, df_hist=df_hist, df_ref=df_year_ref
    )
    after = _merge_meta(computed, flat)

    years = sorted({int(d[:4]) for d in dates})
    existing_parts = [_load_existing_megu(y) for y in years]
    existing = pd.concat([p for p in existing_parts if not p.empty], ignore_index=True) if existing_parts else pd.DataFrame()
    if not existing.empty:
        race_ids = set(flat["race_id"].astype(str))
        existing = existing[existing["race_id"].astype(str).isin(race_ids)]

    before = _baseline_from_existing(existing, flat)

    before_summary = summarize_megu_quality(before, label="before", megu_col="megu_recalc")
    after_summary = summarize_megu_quality(after, label="after", megu_col="megu_index")
    delta = compare_quality(before_summary, after_summary)
    ok, failures = passes_quality_gates(after_summary)

    return {
        "dates": dates,
        "before": before_summary,
        "after": after_summary,
        "delta": delta,
        "passed": ok,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="めぐ指数短期品質検証")
    parser.add_argument(
        "--dates",
        required=True,
        help="YYYYMMDD カンマ区切り (例: 2026-07-18,2026-07-19 または 20260718,20260719)",
    )
    parser.add_argument("--json", action="store_true", help="JSON 出力")
    args = parser.parse_args()

    raw_dates = []
    for part in args.dates.split(","):
        part = part.strip().replace("-", "")
        if len(part) == 8:
            raw_dates.append(part)

    if not raw_dates:
        print("ERROR: --dates に有効な日付を指定してください", file=sys.stderr)
        sys.exit(1)

    result = run_validation(raw_dates)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("=== BEFORE ===")
        for k, v in result["before"].items():
            print(f"  {k}: {v}")
        print("\n=== AFTER ===")
        for k, v in result["after"].items():
            print(f"  {k}: {v}")
        print("\n=== DELTA ===")
        for k, v in result["delta"].items():
            print(f"  {k}: {v}")
        print(f"\nPASSED: {result['passed']}")
        if result["failures"]:
            print("FAILURES:", result["failures"])

    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
