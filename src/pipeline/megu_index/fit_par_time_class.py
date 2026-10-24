"""venue×surface×distance×class_rank 単位の par_time 推定。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.pipeline.megu_index.shrinkage import shrink_scalar

MIN_CELL_N = 10
MIN_SHRINK_N = 10  # これ未満はセル Ridge / 縮約を使わずプールのみ
MIN_CLASS_LEVELS = 2
MIN_RANK_SAMPLES = 5
SHRINK_STRENGTH = 30.0
POOL_SHRINK_STRENGTH = 30.0
RIDGE_ALPHA = 1.0


def _enforce_beta_floor(beta: float, global_beta: float) -> float:
    """beta がフラット過ぎる場合は global_beta（より急な負の傾き）を床とする。"""
    if beta >= 0:
        return global_beta
    return min(beta, global_beta)


def _beta_from_class_means(
    grp: pd.DataFrame,
    global_beta: float,
    min_rank_n: int = MIN_RANK_SAMPLES,
) -> float:
    """surface×distance_band 等のプールで、クラス別平均タイムから beta を推定。"""
    stats = grp.groupby("class_rank")["adjusted_time_sec"].agg(["mean", "count"])
    valid = stats[stats["count"] >= min_rank_n]
    if len(valid) < 2:
        return global_beta
    x = np.array(valid.index, dtype=float).reshape(-1, 1)
    y = valid["mean"].to_numpy(dtype=float)
    ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    ridge.fit(x, y)
    beta = float(ridge.coef_[0])
    if beta >= 0:
        return global_beta
    n_eff = int(valid["count"].sum())
    beta, _ = shrink_scalar(beta, global_beta, n_eff, strength=POOL_SHRINK_STRENGTH)
    return _enforce_beta_floor(beta, global_beta)


def fit_pool_betas(df_par_base: pd.DataFrame) -> tuple[float, dict[tuple[str, str], float]]:
    """グローバルβと surface×distance_band プールβを返す。"""
    work = df_par_base.copy()
    work["time_mean_cell"] = work.groupby(["venue", "surface", "distance"])["adjusted_time_sec"].transform(
        "mean"
    )
    work["time_center"] = work["adjusted_time_sec"] - work["time_mean_cell"]

    ridge_global = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    ridge_global.fit(work[["class_rank"]].values, work["time_center"].values)
    global_beta = float(ridge_global.coef_[0])

    pool_distband_beta: dict[tuple[str, str], float] = {}
    for (surface, distband), grp in work.groupby(["surface", "distance_band"], sort=False, observed=False):
        key = (str(surface), str(distband))
        pool_distband_beta[key] = _beta_from_class_means(grp, global_beta)

    if global_beta >= 0:
        global_beta = min(pool_distband_beta.values()) if pool_distband_beta else -0.05

    return global_beta, pool_distband_beta


def _prior_beta(
    surface: str,
    distband_key: str | None,
    pool_distband_beta: dict[tuple[str, str], float],
    global_beta: float,
) -> tuple[float, str]:
    if distband_key and (surface, distband_key) in pool_distband_beta:
        beta = pool_distband_beta[(surface, distband_key)]
        if beta < 0:
            return beta, "pool_distband"
    return global_beta, "pool_global"


def _cell_ridge_fit(grp: pd.DataFrame) -> tuple[float, float] | None:
    if len(grp) < MIN_CELL_N or grp["class_rank"].nunique() < MIN_CLASS_LEVELS:
        return None
    ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    ridge.fit(grp[["class_rank"]].values, grp["adjusted_time_sec"].values)
    alpha = float(ridge.intercept_)
    beta = float(ridge.coef_[0])
    return alpha, beta


def _is_monotone_decreasing(alpha: float, beta: float) -> bool:
    if beta >= 0:
        return False
    times = [alpha + beta * cr for cr in range(1, 8)]
    return all(times[i] >= times[i + 1] for i in range(len(times) - 1))


def _calibrate_alpha(grp: pd.DataFrame, beta: float) -> float:
    cell_time_mean = float(grp["adjusted_time_sec"].mean())
    cell_rank_mean = float(grp["class_rank"].mean())
    return cell_time_mean - beta * cell_rank_mean


def fit_par_time_class(
    df_par_base: pd.DataFrame,
    global_beta: float,
    pool_distband_beta: dict[tuple[str, str], float],
    min_cell_n: int = MIN_CELL_N,
    shrink_strength: float = SHRINK_STRENGTH,
) -> pd.DataFrame:
    """
    par_time_class テーブルを返す。

    セル内は 2着馬の全行で Ridge（class_rank → adjusted_time_sec）を当て、
    beta をプール値へ縮約。beta>=0 や単調性 NG は prior のみ採用。
    """
    rows: list[dict] = []

    for (venue, surface, distance), grp in df_par_base.groupby(["venue", "surface", "distance"], sort=False):
        n_fit = len(grp)
        distance_int = int(distance)
        distband_key = str(grp["distance_band"].iloc[0]) if "distance_band" in grp.columns else None
        beta_prior, prior_label = _prior_beta(surface, distband_key, pool_distband_beta, global_beta)

        if n_fit < min_cell_n:
            beta_val, source = beta_prior, prior_label
            alpha_val = _calibrate_alpha(grp, beta_val)
        else:
            ridge_fit = _cell_ridge_fit(grp)
            if ridge_fit is None:
                beta_val, source = beta_prior, prior_label
                alpha_val = _calibrate_alpha(grp, beta_val)
            else:
                _alpha_ridge, beta_cell = ridge_fit
                if n_fit < MIN_SHRINK_N:
                    beta_val, source = beta_prior, prior_label
                    alpha_val = _calibrate_alpha(grp, beta_val)
                else:
                    beta_val, w = shrink_scalar(beta_cell, beta_prior, n_fit, strength=shrink_strength)
                    alpha_val = _calibrate_alpha(grp, beta_val)
                    if not _is_monotone_decreasing(alpha_val, beta_val):
                        beta_val, source = beta_prior, prior_label
                        alpha_val = _calibrate_alpha(grp, beta_val)
                    elif w >= 0.5:
                        source = "cell_shrink"
                    elif w > 0:
                        source = "pool_shrink"
                    else:
                        source = prior_label

        beta_val = _enforce_beta_floor(beta_val, global_beta)
        alpha_val = _calibrate_alpha(grp, beta_val)

        for cr in range(1, 8):
            rows.append(
                {
                    "venue": venue,
                    "surface": surface,
                    "distance": distance_int,
                    "class_rank": cr,
                    "par_time_sec": alpha_val + beta_val * cr,
                    "alpha": alpha_val,
                    "beta": beta_val,
                    "n_fit": n_fit,
                    "source": source,
                }
            )

    out = pd.DataFrame(rows)
    out["distance"] = out["distance"].astype(int)
    out["class_rank"] = out["class_rank"].astype(int)
    out["n_fit"] = out["n_fit"].astype(int)
    return out
