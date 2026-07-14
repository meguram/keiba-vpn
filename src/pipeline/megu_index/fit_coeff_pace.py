"""venue×surface×distance 単位の Δpace 係数（coeff_pace）推定。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.pipeline.megu_index.shrinkage import shrink_scalar

MIN_SAMPLES = 10
MIN_POOL_SAMPLES = 30
SHRINK_STRENGTH = 30.0
RIDGE_ALPHA = 1.0
THEORY_COEFF_PACE = 0.8
COEFF_CLIP = (0.3, 1.5)


def _ridge_slope(x: np.ndarray, y: np.ndarray, alpha: float = RIDGE_ALPHA) -> float:
    if len(x) < 2 or np.nanvar(x) < 1e-9:
        return np.nan
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(x.reshape(-1, 1), y)
    return float(model.coef_[0])


def _prepare_pace_fit(df_train: pd.DataFrame) -> pd.DataFrame:
    needed = {"front_split_sec", "front_split_dev", "adjusted_time_sec", "distance_band"}
    missing = needed - set(df_train.columns)
    if missing:
        raise KeyError(f"fit_coeff_pace: missing columns {sorted(missing)}")
    out = df_train[
        df_train["front_split_sec"].notna()
        & df_train["distance_band"].notna()
        & df_train["adjusted_time_sec"].notna()
    ].copy()
    out["front_split_dev_dm"] = out["front_split_dev"] - out.groupby("race_id")["front_split_dev"].transform(
        "mean"
    )
    out["time_dm"] = out["adjusted_time_sec"] - out.groupby("race_id")["adjusted_time_sec"].transform("mean")
    return out


def _race_level_frame(df_pace_fit: pd.DataFrame) -> pd.DataFrame:
    return (
        df_pace_fit.groupby(["venue", "surface", "distance", "race_id"], as_index=False)
        .agg(
            front_split_dev=("front_split_dev", "mean"),
            adjusted_time_sec=("adjusted_time_sec", "mean"),
        )
    )


def _fit_group_slopes(
    df: pd.DataFrame,
    group_cols: list[str],
    x_col: str,
    y_col: str,
    min_samples: int,
) -> pd.DataFrame:
    rows = []
    for keys, grp in df.groupby(group_cols, sort=False, observed=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        x = grp[x_col].to_numpy(dtype=float)
        y = grp[y_col].to_numpy(dtype=float)
        n = len(x)
        slope = _ridge_slope(x, y) if n >= min_samples else np.nan
        row = dict(zip(group_cols, keys))
        row["slope"] = slope
        row["n_fit"] = n
        rows.append(row)
    return pd.DataFrame(rows)


def _prior_from_pools(
    row: pd.Series,
    theory_value: float,
) -> tuple[float, str]:
    for col, label in (
        ("coeff_pool_db", "pool_distband"),
        ("coeff_pool_surface", "pool_surface"),
    ):
        val = row.get(col)
        if pd.notna(val) and float(val) > 0:
            return float(val), label
    return theory_value, "theory"


def _expand_coeff_coverage(
    fitted: pd.DataFrame,
    df_ref: pd.DataFrame,
    dist_to_band: pd.DataFrame,
    pool_db: pd.DataFrame,
    pool_surf: pd.DataFrame,
    theory_value: float,
) -> pd.DataFrame:
    """スプリット欠損などで推定されなかった venue×surface×distance にプール係数を補完。"""
    needed = {"venue", "surface", "distance"}
    if not needed.issubset(df_ref.columns):
        return fitted

    all_keys = (
        df_ref[list(needed)]
        .dropna(subset=["distance"])
        .assign(distance=lambda d: pd.to_numeric(d["distance"], errors="coerce"))
        .dropna(subset=["distance"])
        .drop_duplicates()
        .astype({"distance": int})
    )
    if "distance_band" in df_ref.columns:
        band_map = (
            df_ref.assign(distance=lambda d: pd.to_numeric(d["distance"], errors="coerce"))
            .dropna(subset=["distance", "distance_band"])
            .astype({"distance": int})
            .drop_duplicates(subset=["distance"])[["distance", "distance_band"]]
        )
        dist_to_band = (
            pd.concat([dist_to_band, band_map], ignore_index=True)
            .drop_duplicates(subset=["distance"], keep="first")
        )
    merged = all_keys.merge(fitted, on=["venue", "surface", "distance"], how="left")
    missing = merged["coeff_pace"].isna()
    if not missing.any():
        return fitted

    impute = merged.loc[missing, ["venue", "surface", "distance"]].copy()
    impute = impute.merge(dist_to_band, on="distance", how="left")
    pool_db_cols = pool_db.rename(columns={"slope": "coeff_pool_db"}) if "slope" in pool_db.columns else pool_db
    pool_surf_cols = pool_surf.rename(columns={"slope": "coeff_pool_surface"}) if "slope" in pool_surf.columns else pool_surf
    impute = impute.merge(pool_db_cols, on=["surface", "distance_band"], how="left")
    impute = impute.merge(pool_surf_cols, on="surface", how="left")
    priors = impute.apply(lambda r: _prior_from_pools(r, theory_value), axis=1)
    impute["coeff_pace"] = [float(p[0]) for p in priors]
    impute["source"] = [p[1] + "_imputed" for p in priors]
    impute["n_fit"] = 0
    impute["coeff_pace"] = impute["coeff_pace"].clip(lower=COEFF_CLIP[0], upper=COEFF_CLIP[1])

    out = pd.concat(
        [
            fitted,
            impute[["venue", "surface", "distance", "coeff_pace", "n_fit", "source"]],
        ],
        ignore_index=True,
    )
    return out.sort_values(["venue", "surface", "distance"]).reset_index(drop=True)


def fit_coeff_pace(
    df_train: pd.DataFrame,
    min_samples: int = MIN_SAMPLES,
    shrink_strength: float = SHRINK_STRENGTH,
    theory_value: float = THEORY_COEFF_PACE,
    expand_coverage: bool = True,
) -> pd.DataFrame:
    """
    Δpace 係数テーブルを返す。

    各セルでレース平均の Ridge 傾きを推定し、
    surface×distance_band → surface → 理論値 へ経験的ベイズ縮約する。
    """
    df_pace_fit = _prepare_pace_fit(df_train)
    df_race = _race_level_frame(df_pace_fit)
    dist_to_band = (
        df_pace_fit[["distance", "distance_band"]].drop_duplicates().dropna(subset=["distance_band"])
    )
    df_race = df_race.merge(dist_to_band, on="distance", how="left")

    cell_race = _fit_group_slopes(
        df_race,
        ["venue", "surface", "distance"],
        "front_split_dev",
        "adjusted_time_sec",
        min_samples,
    ).rename(columns={"slope": "coeff_cell", "n_fit": "n_race"})

    # プール事前値はレース平均 Ridge（レース内偏差より正の傾きが出やすい）
    pool_db = _fit_group_slopes(
        df_race,
        ["surface", "distance_band"],
        "front_split_dev",
        "adjusted_time_sec",
        MIN_POOL_SAMPLES,
    ).rename(columns={"slope": "coeff_pool_db"})

    pool_surf = _fit_group_slopes(
        df_race,
        ["surface"],
        "front_split_dev",
        "adjusted_time_sec",
        MIN_POOL_SAMPLES,
    ).rename(columns={"slope": "coeff_pool_surface"})

    cells = cell_race.copy()
    cells = cells.merge(dist_to_band, on="distance", how="left")
    cells = cells.merge(pool_db, on=["surface", "distance_band"], how="left")
    cells = cells.merge(pool_surf, on="surface", how="left")

    def _prior_chain(row: pd.Series) -> tuple[float, str]:
        return _prior_from_pools(row, theory_value)

    def _fit_row(row: pd.Series) -> tuple[float, str]:
        prior, prior_label = _prior_chain(row)
        raw = row.get("coeff_cell")
        n = int(row.get("n_race") or 0)
        if pd.isna(raw) or float(raw) <= 0:
            return prior, prior_label
        shrunk, w = shrink_scalar(float(raw), prior, n, strength=shrink_strength)
        if shrunk <= 0:
            return prior, prior_label
        if w >= 0.5:
            return shrunk, "cell_shrink"
        if w > 0:
            return shrunk, "pool_shrink"
        return prior, prior_label

    picked = cells.apply(_fit_row, axis=1)
    cells["coeff_pace_raw"] = [p[0] for p in picked]
    cells["source"] = [p[1] for p in picked]
    cells["coeff_pace"] = cells["coeff_pace_raw"].clip(lower=COEFF_CLIP[0], upper=COEFF_CLIP[1])
    cells["n_fit"] = cells["n_race"].fillna(0).astype(int)
    cells["distance"] = cells["distance"].astype(int)

    out = cells[["venue", "surface", "distance", "coeff_pace", "n_fit", "source"]].copy()
    if expand_coverage:
        out = _expand_coeff_coverage(out, df_train, dist_to_band, pool_db, pool_surf, theory_value)
    return out
