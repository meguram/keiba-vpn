"""tp-50 共通部品: チケット表の読み込みと特徴量、方策の評価（レース単位）、多重比較。

チケット表は tp50_data.py が作る（1 行 = レース × 買い目）。方策は『各行にいくら賭けるか（stake）』で表し、
レースごとの賭け金 S_r と払戻 R_r（= Σ stake × ret）に集計して比較する。ROI = ΣR / ΣS。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BET = ROOT / "data/local/modeling/betting"
TP50 = Path(os.environ.get("TP50_OUT", str(BET / "tp50")))
BT_NAMES = ["tansho", "fukusho", "umaren", "wide", "umatan", "sanrentan"]
FEATS = (
    [f"bt{i}" for i in range(6)]
    + ["n", "r_p1", "r_ratio12", "r_ent", "r_q1", "r_kl", "r_dom", "r_even", "r_upset"]
    + ["lodds", "lp_model", "lp_mix", "lq", "lev_model", "lev_mix", "pq_model", "pq_mix", "pop_min", "pop_max", "sum_pw", "sum_q", "rk_ev_mix", "rk_ev_model", "rk_p", "n_c"]
)
PRIMARY_YEARS = (2024, 2025)
RET_CAP = 30.0


def load_years(years) -> pd.DataFrame:
    parts = []
    for y in years:
        f = TP50 / f"tickets_{y}.parquet"
        if f.exists():
            parts.append(pd.read_parquet(f))
    d = pd.concat(parts, ignore_index=True)
    d["ev_model"] = (d["p_model"] * d["odds"]).astype(np.float32)
    d["ev_mix"] = (d["p_mix"] * d["odds"]).astype(np.float32)
    eps = 1e-9
    d["lodds"] = np.log(d["odds"].clip(lower=1.0)).astype(np.float32)
    d["lp_model"] = np.log(d["p_model"].clip(lower=eps)).astype(np.float32)
    d["lp_mix"] = np.log(d["p_mix"].clip(lower=eps)).astype(np.float32)
    d["lq"] = np.log(d["q"].clip(lower=eps)).astype(np.float32)
    d["lev_model"] = np.log1p(d["ev_model"].clip(0, 50)).astype(np.float32)
    d["lev_mix"] = np.log1p(d["ev_mix"].clip(0, 50)).astype(np.float32)
    d["pq_model"] = (d["lp_model"] - d["lq"]).astype(np.float32)
    d["pq_mix"] = (d["lp_mix"] - d["lq"]).astype(np.float32)
    d = d.sort_values(["date", "race_id"], kind="stable").reset_index(drop=True)
    d["ridx"] = pd.factorize(d["race_id"])[0].astype(np.int32)
    g = d.groupby(["ridx", "bt"], sort=False)
    d["rk_ev_mix"] = g["ev_mix"].rank(pct=True, ascending=False).astype(np.float32)
    d["rk_ev_model"] = g["ev_model"].rank(pct=True, ascending=False).astype(np.float32)
    d["rk_p"] = g["p_model"].rank(pct=True, ascending=False).astype(np.float32)
    d["n_c"] = d.groupby("ridx", sort=False)["odds"].transform("size").astype(np.float32)
    for i in range(6):
        d[f"bt{i}"] = (d["bt"] == i).astype(np.float32)
    # レース内は ev_mix の降順（先頭が最有力。パディングの切り詰めもここを基準にする）
    d = d.sort_values(["ridx", "ev_mix"], ascending=[True, False], kind="stable").reset_index(drop=True)
    return d


def race_starts(ridx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """行が race で連続・昇順であることが前提。各レースの開始位置と行数。"""
    chg = np.flatnonzero(np.diff(ridx, prepend=ridx[0] - 1)) if len(ridx) else np.array([], int)
    starts = chg
    counts = np.diff(np.append(starts, len(ridx)))
    return starts, counts


class Stats:
    """方策ごとのレース単位の集計（S: 賭け金, R: 払戻, nb: 買い点数, nh: 的中点数）。"""

    def __init__(self, d: pd.DataFrame):
        self.ret = d["ret"].to_numpy(np.float64)
        self.ridx = d["ridx"].to_numpy()
        self.starts, self.counts = race_starts(self.ridx)
        self.n_races = len(self.starts)
        self.race_ids = d["race_id"].to_numpy()[self.starts]
        self.year = d["year"].to_numpy()[self.starts]

    def summarize(self, stake: np.ndarray) -> dict:
        s = np.asarray(stake, np.float64)
        S = np.add.reduceat(s, self.starts)
        R = np.add.reduceat(s * self.ret, self.starts)
        on = (s > 0)
        nb = np.add.reduceat(on.astype(np.float64), self.starts)
        nh = np.add.reduceat((on & (self.ret > 0)).astype(np.float64), self.starts)
        return {"S": S.astype(np.float32), "R": R.astype(np.float32), "nb": nb.astype(np.float32), "nh": nh.astype(np.float32)}


def roi_ci(S: np.ndarray, R: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    if S.sum() <= 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(S)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        sb = S[idx].sum()
        vals[b] = R[idx].sum() / sb if sb > 0 else np.nan
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(R.sum() / S.sum()), float(lo), float(hi)


def reality_check(profit: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """White の Reality Check（スチューデント化した最大統計量のレース単位ブートストラップ）。profit: [レース数, 方策数]。"""
    R, K = profit.shape
    mu = profit.mean(0)
    sd = profit.std(0, ddof=1)
    sd = np.where(sd > 1e-12, sd, np.inf)
    t = np.sqrt(R) * mu / sd
    rng = np.random.default_rng(seed)
    tmax = np.empty(n_boot)
    for b in range(n_boot):
        pb = profit[rng.integers(0, R, R)]
        sdb = pb.std(0, ddof=1)
        sdb = np.where(sdb > 1e-12, sdb, np.inf)
        tmax[b] = (np.sqrt(R) * (pb.mean(0) - mu) / sdb).max()
    return {"t_max": float(t.max()), "best": int(t.argmax()), "p_value": float((tmax >= t.max()).mean()), "n_strategies": int(K)}


def pad_races(d: pd.DataFrame, X: np.ndarray, K: int = 320):
    """行列 X [行, 特徴] をレースごとに [R, K, F] に詰める。戻り値: (Xp, retp, maskp, rows_r, rows_pos, keep)"""
    starts, counts = race_starts(d["ridx"].to_numpy())
    R = len(starts)
    rid_local = np.repeat(np.arange(R), counts)
    pos = np.arange(len(d)) - np.repeat(starts, counts)
    keep = pos < K
    Xp = np.zeros((R, K, X.shape[1]), np.float32)
    retp = np.zeros((R, K), np.float32)
    maskp = np.zeros((R, K), bool)
    Xp[rid_local[keep], pos[keep]] = X[keep]
    retp[rid_local[keep], pos[keep]] = np.minimum(d["ret"].to_numpy(np.float32)[keep], RET_CAP)
    maskp[rid_local[keep], pos[keep]] = True
    return Xp, retp, maskp, rid_local, pos, keep


def standardize(Xtr: np.ndarray, *others: np.ndarray):
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    return [(x - mu) / sd for x in (Xtr, *others)]
