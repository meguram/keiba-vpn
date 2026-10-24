"""基準前半スプリット（2着タイム連続モデル）の推定と付与。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _fit_front_split_ols(grp: pd.DataFrame) -> dict | None:
    g = grp.dropna(subset=["front_split_sec", "race_t2nd_sec"])
    n = len(g)
    if n < 10:
        return None
    t2nd_ref = float(g["race_t2nd_sec"].median())
    x = g["race_t2nd_sec"].to_numpy() - t2nd_ref
    y = g["front_split_sec"].to_numpy()
    varx = float(np.var(x))
    intercept = float(np.mean(y))
    if varx > 1e-9:
        slope = float(np.cov(x, y, bias=True)[0, 1] / varx)
    else:
        slope = 0.0
    return {
        "par_intercept": intercept,
        "par_slope": slope,
        "t2nd_ref": t2nd_ref,
        "n_fit": n,
    }


def fit_par_front_split_coefficients(
    df_2nd: pd.DataFrame,
    min_cell_n: int = 30,
) -> pd.DataFrame:
    """distance×surface の基準前半スプリット係数テーブルを返す。"""
    cell_rows = []
    for (distance, surface), grp in df_2nd.groupby(["distance", "surface"], sort=False):
        fit = _fit_front_split_ols(grp)
        if fit is None:
            continue
        cell_rows.append({"distance": distance, "surface": surface, "model": "cell", **fit})
    par_split_cell = pd.DataFrame(cell_rows)

    surf_rows = []
    for surface, grp in df_2nd.groupby("surface", sort=False):
        fit = _fit_front_split_ols(grp)
        if fit is None:
            continue
        surf_rows.append({"surface": surface, **fit})
    par_split_surface = pd.DataFrame(surf_rows)

    cells = (
        df_2nd[["distance", "surface"]]
        .drop_duplicates()
        .merge(par_split_cell, on=["distance", "surface"], how="left")
    )
    cells = cells.merge(
        par_split_surface.add_prefix("pool_"),
        left_on="surface",
        right_on="pool_surface",
        how="left",
    )
    use_cell = cells["n_fit"].fillna(0) >= min_cell_n
    cells["par_intercept"] = np.where(use_cell, cells["par_intercept"], cells["pool_par_intercept"])
    cells["par_slope"] = np.where(use_cell, cells["par_slope"], cells["pool_par_slope"])
    cells["t2nd_ref"] = np.where(use_cell, cells["t2nd_ref"], cells["pool_t2nd_ref"])
    cells["n_fit"] = (
        pd.Series(np.where(use_cell, cells["n_fit"], cells["pool_n_fit"]), index=cells.index)
        .round()
        .astype(int)
    )
    cells["model"] = np.where(use_cell, "cell", "pool_surface")
    cells = cells.dropna(subset=["par_intercept"])
    return cells[["distance", "surface", "par_intercept", "par_slope", "t2nd_ref", "n_fit", "model"]].copy()


def attach_par_front_split_sec(
    df: pd.DataFrame,
    par_split_full: pd.DataFrame,
    df_2nd: pd.DataFrame,
) -> pd.DataFrame:
    """係数テーブルから par_front_split_sec を各行へ付与。"""
    out = df.copy()
    coef_cols = ["par_intercept", "par_slope", "t2nd_ref"]
    drop_cols = [c for c in coef_cols + ["par_front_split_sec", "par_front_split_fb"] if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    out = out.merge(par_split_full, on=["distance", "surface"], how="left")
    out["par_front_split_sec"] = (
        out["par_intercept"] + out["par_slope"] * (out["race_t2nd_sec"] - out["t2nd_ref"])
    )
    fb = (
        df_2nd.groupby(["distance", "surface"])["front_split_sec"]
        .median()
        .reset_index()
        .rename(columns={"front_split_sec": "par_front_split_fb"})
    )
    out = out.merge(fb, on=["distance", "surface"], how="left")
    out["par_front_split_sec"] = out["par_front_split_sec"].fillna(out["par_front_split_fb"])
    return out
