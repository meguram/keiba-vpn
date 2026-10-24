"""オッズ予測モデルの評価指標（tp-20）。確定オッズを正規化した市場確率との誤差で測る。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def market_prob(race: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """レース内で 1/オッズ を正規化した市場確率（控除率を除いた相対的な人気）。"""
    inv = pd.Series(1.0 / np.asarray(odds, dtype=float))
    return (inv / inv.groupby(race).transform("sum")).to_numpy()


def per_race_sq_log_err(race: np.ndarray, p_true: np.ndarray, p_hat: np.ndarray) -> pd.Series:
    """レースごとの log 市場確率の二乗誤差の平均。"""
    e = (np.log(p_true) - np.log(np.clip(p_hat, 1e-9, None))) ** 2
    return pd.Series(e).groupby(race).mean()


def kl_by_race(race: np.ndarray, p_true: np.ndarray, p_hat: np.ndarray) -> pd.Series:
    """KL(市場 || 予測)。p_hat はレース内で再正規化してから測る。"""
    ph = pd.Series(np.clip(p_hat, 1e-9, None))
    ph = (ph / ph.groupby(race).transform("sum")).to_numpy()
    return pd.Series(p_true * (np.log(p_true) - np.log(ph))).groupby(race).sum()


def spearman_by_race(race: np.ndarray, odds: np.ndarray, pred_odds: np.ndarray) -> float:
    d = pd.DataFrame({"r": race, "a": odds, "b": pred_odds})
    ra = d.groupby("r")["a"].rank()
    rb = d.groupby("r")["b"].rank()
    c = pd.DataFrame({"r": race, "a": ra, "b": rb}).groupby("r").apply(lambda g: g["a"].corr(g["b"]) if len(g) > 2 else np.nan)
    return float(c.mean())


def band_table(odds: np.ndarray, pred: np.ndarray, bands=(1.0, 3.0, 5.0, 10.0, 20.0, 50.0, 1e9)) -> pd.DataFrame:
    """確定オッズの帯ごとの、対数誤差の RMSE・偏り（予測÷確定の中央値）。穴馬帯の精度が馬券に効く。"""
    d = pd.DataFrame({"odds": odds, "pred": pred})
    d["band"] = pd.cut(d["odds"], list(bands), right=False)
    d["e"] = np.log(d["pred"]) - np.log(d["odds"])
    g = d.groupby("band", observed=True)
    return pd.DataFrame({"n": g.size(), "rmse_log": g["e"].apply(lambda s: float(np.sqrt((s**2).mean()))), "予測÷確定の中央値": g.apply(lambda x: float((x["pred"] / x["odds"]).median()))})


def paired_bootstrap_mean_diff(a: pd.Series, b: pd.Series, n_boot: int = 1000, seed: int = 0) -> dict:
    """レース単位の誤差 a（新）と b（基準）の平均差（a − b）。負 = 新が良い。"""
    idx = a.index.intersection(b.index)
    d = (a.loc[idx] - b.loc[idx]).to_numpy()
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return {"diff": float(d.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}
