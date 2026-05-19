"""競走馬の適性プロファイル計算に必要なインデックスを構築する。

出力する主要アーティファクト:
    1. ``horse_name_index.parquet``
        - horse_id, horse_name, sex (牡/牝/セ), birth_year, n_races, last_date
    2. ``horse_perf_by_cond.parquet``
        - horse_id, cond_kind, cond_value, n_races, n_win, n_top3, win_rate, top3_rate
        - cond_kind ∈ {venue, surface, distance_bin, track_condition, pace}
        - **オッズ < 30 倍** のレコードのみを集約 (= 信頼性の高いサンプル)

距離ビン:
    1000-1300  → "短距離"
    1400-1600  → "マイル"
    1700-2000  → "中距離"
    2100-2400  → "中長距離"
    2500+      → "長距離"

集約対象 (年):
    data/local/tables/{year}/race_result_flat.parquet  for year in 2020-2026

Usage:
    python -m src.research.pedigree.build_horse_aptitude_index
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "data/research/horse_aptitude"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ODDS_NOISE_THRESHOLD = 30.0  # オッズ >= 30 倍 はノイズとして除外


def _bin_distance(d: int) -> str:
    if d < 1400:
        return "短距離"
    if d < 1700:
        return "マイル"
    if d < 2100:
        return "中距離"
    if d < 2500:
        return "中長距離"
    return "長距離"


def _parse_sex(sex_age: str) -> str:
    if not isinstance(sex_age, str):
        return "?"
    if sex_age.startswith("牡"):
        return "牡"
    if sex_age.startswith("牝"):
        return "牝"
    if sex_age.startswith("セ"):
        return "セ"
    return "?"


def main() -> int:
    print("[load] race_result_flat (全年) ...", flush=True)
    files = sorted(glob.glob(str(ROOT / "data/local/tables/*/race_result_flat.parquet")))
    print(f"  files: {len(files)}", flush=True)
    cols = [
        "race_id", "date", "venue", "surface", "distance", "track_condition",
        "horse_id", "horse_name", "horse_number", "sex_age", "odds", "popularity",
        "finish_position",
    ]
    dfs = []
    for f in files:
        d = pd.read_parquet(f, columns=cols)
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    print(f"  total: {len(df):,} 行", flush=True)

    # 整形
    df["horse_id"] = df["horse_id"].astype(str)
    df["finish_position"] = pd.to_numeric(df["finish_position"], errors="coerce")
    df["odds"] = pd.to_numeric(df["odds"], errors="coerce")
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df = df.dropna(subset=["horse_id", "finish_position", "distance"])
    df["sex"] = df["sex_age"].apply(_parse_sex)
    df["distance_bin"] = df["distance"].astype(int).apply(_bin_distance)

    # 1) 馬名インデックス (horse_id ごと最新エントリ)
    print("[build] horse_name_index ...", flush=True)
    df_sorted = df.sort_values("date", ascending=False)
    idx = (
        df_sorted.groupby("horse_id")
        .agg(
            horse_name=("horse_name", "first"),
            sex=("sex", "first"),
            sex_age_first=("sex_age", "first"),
            n_races=("race_id", "count"),
            last_date=("date", "max"),
            first_date=("date", "min"),
        )
        .reset_index()
    )
    print(f"  unique horses: {len(idx):,}", flush=True)
    idx.to_parquet(OUT_DIR / "horse_name_index.parquet")

    # 2) 馬ごとの条件別成績 (オッズ<30倍のみ)
    print(f"[build] horse_perf_by_cond (オッズ < {ODDS_NOISE_THRESHOLD} のみ) ...", flush=True)
    df_clean = df[df["odds"] < ODDS_NOISE_THRESHOLD].copy()
    print(f"  filtered: {len(df_clean):,} 行 ({len(df_clean)/max(len(df),1)*100:.1f}%)", flush=True)
    df_clean["is_win"] = (df_clean["finish_position"] == 1).astype(int)
    df_clean["is_top3"] = (df_clean["finish_position"] <= 3).astype(int)

    cond_kinds = {
        "venue": "venue",
        "surface": "surface",
        "distance_bin": "distance_bin",
        "track_condition": "track_condition",
    }

    rows = []
    for kind, col in cond_kinds.items():
        g = (
            df_clean.groupby(["horse_id", col])
            .agg(n_races=("race_id", "count"), n_win=("is_win", "sum"),
                 n_top3=("is_top3", "sum"))
            .reset_index()
        )
        g = g.rename(columns={col: "cond_value"})
        g["cond_kind"] = kind
        rows.append(g)

    perf = pd.concat(rows, ignore_index=True)
    perf["win_rate"] = (perf["n_win"] / perf["n_races"]).astype(float)
    perf["top3_rate"] = (perf["n_top3"] / perf["n_races"]).astype(float)
    perf.to_parquet(OUT_DIR / "horse_perf_by_cond.parquet")
    print(f"  rows: {len(perf):,}", flush=True)

    print("\n=== サマリ ===")
    print(f"  unique horses (全期間):  {len(idx):,}")
    print(f"  perf rows  (オッズ<30倍): {len(perf):,}")
    print(f"  sex 分布: {idx['sex'].value_counts().to_dict()}")
    print(f"  cond 分布: {perf['cond_kind'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
