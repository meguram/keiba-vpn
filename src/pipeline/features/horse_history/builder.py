"""馬の過去走履歴特徴量の組み立てとリーク検査。出力は (race_id, horse_id) をキーとする特徴量表。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from .common import apply_repair, build_repair_map, load_runs
from .common import FIT_END
from .elo import run_elo, select_delta
from .match import build_all, families
from .profile import KEY


def build_from_runs(runs: pl.DataFrame, repair_map: pl.DataFrame,
                    on_time: pl.DataFrame | None = None) -> tuple[pl.DataFrame, dict]:
    delta, lls = select_delta(runs, FIT_END)  # 斤量差のハンデ項（0 が最良なら従来どおり）
    F, P = build_all(apply_repair(runs, repair_map, on_time), run_elo(runs, delta=delta))
    P["elo_delta"], P["elo_delta_ll"] = delta, lls
    return F, P


def build_with_state(runs: pl.DataFrame, repair_map: pl.DataFrame, on_time: pl.DataFrame | None = None):
    """build_from_runs と同じ特徴量に加え、推論用の状態（fit した値・過去走の素材・Elo の最終状態）も返す。"""
    delta, lls = select_delta(runs, FIT_END)
    elo, rating, cnt = run_elo(runs, delta=delta, return_state=True)
    S: dict = {}
    F, P = build_all(apply_repair(runs, repair_map, on_time), elo, S)
    P["elo_delta"], P["elo_delta_ll"] = delta, lls
    return F, P, S, {"rating": rating, "cnt": cnt, "delta": delta}


def build_horse_history(
    layer_a_path: str | Path, tables_dir: str | Path | None = None, on_time: pl.DataFrame | None = None
) -> tuple[pl.DataFrame, dict]:
    """2020〜2025 年の全走から特徴量表と全体フィット値を作る。"""
    return build_from_runs(load_runs(tables_dir), build_repair_map(layer_a_path), on_time)


def feature_columns(columns: list[str]) -> dict[str, list[str]]:
    return families(columns)


def _compare(a_df: pl.DataFrame, b_df: pl.DataFrame, cols: list[str], expr: pl.Expr) -> tuple[int, dict]:
    a = a_df.filter(expr).select(KEY + cols).sort(KEY)
    b = b_df.filter(expr).select(KEY + cols).sort(KEY)
    if a.height != b.height or not (a.select(KEY).to_numpy() == b.select(KEY).to_numpy()).all():
        raise AssertionError(f"比較対象の行が一致しません: {a.height} vs {b.height}")
    bad = {}
    for c in cols:
        x = a[c].to_numpy().astype(float)
        y = b[c].to_numpy().astype(float)
        d = np.where(np.isnan(x) & np.isnan(y), 0.0, np.abs(x - y))
        d = np.where(np.isnan(d), np.inf, d)
        if (d > 1e-9).any():
            bad[c] = int((d > 1e-9).sum())
    return a.height, bad


def truncation_check(
    runs: pl.DataFrame, repair_map: pl.DataFrame, F: pl.DataFrame, cut: date, cols: list[str],
    on_time: pl.DataFrame | None = None,
) -> dict:
    """cut 以降のデータを消して再計算しても、cut より前のレースの特徴量が変わらないこと。"""
    Ft, _ = build_from_runs(runs.filter(pl.col("date") < cut), repair_map.filter(pl.col("src_date") < cut), on_time)
    rows, bad = _compare(F, Ft, cols, pl.col("date") < cut)
    return {"cut": str(cut), "rows": rows, "features": len(cols), "mismatch": bad}


def permutation_check(
    runs: pl.DataFrame, repair_map: pl.DataFrame, F: pl.DataFrame, cut: date, cols: list[str], seed: int = 1,
    on_time: pl.DataFrame | None = None,
) -> dict:
    """cut 以降の全レースで結果をレース内で入れ替えても、cut 以降の最初の開催日の特徴量が変わらないこと。"""
    outcome_cols = ["finish_position", "time_sec", "last_3f", "passing_order"]
    rng = np.random.default_rng(seed)
    d = runs.with_row_index("ri")
    post = d.filter(pl.col("date") >= cut)
    src = post.sort(["race_id", "ri"]).with_columns(pl.int_range(pl.len()).over("race_id").alias("k"))
    shuf = (
        post.select("race_id", "ri")
        .with_columns(pl.Series("u", rng.random(post.height)))
        .sort(["race_id", "u"])
        .with_columns(pl.int_range(pl.len()).over("race_id").alias("k"))
        .select("race_id", "k", pl.col("ri").alias("ri_src"))
    )
    mapping = src.select("race_id", "k", "ri").join(shuf, on=["race_id", "k"])
    vals = d.select("ri", *outcome_cols).rename({c: c + "_n" for c in outcome_cols} | {"ri": "ri_src"})
    new = mapping.join(vals, on="ri_src").select("ri", *[c + "_n" for c in outcome_cols])
    d2 = d.join(new, on="ri", how="left")
    d2 = d2.with_columns([pl.coalesce(pl.col(c + "_n"), pl.col(c)).alias(c) for c in outcome_cols])
    d2 = d2.drop([c + "_n" for c in outcome_cols] + ["ri"])
    Fp, _ = build_from_runs(d2, repair_map, on_time)
    first = runs.filter(pl.col("date") >= cut)["date"].min()
    rows, bad = _compare(F, Fp, cols, pl.col("date") == first)
    return {"day": str(first), "rows": rows, "features": len(cols), "mismatch": bad}
