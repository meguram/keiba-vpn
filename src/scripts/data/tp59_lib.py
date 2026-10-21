"""tp-59 の共通部品（純粋関数）: レース単位ブートストラップ、as-of の関係者別成績、縮約、再現性の指標、Reality Check。

回収率は『1 出走に 100 円』の払い戻し（ret_* は 100 円あたりの額）。回収率 = Σret / (100 × 件数)。区間はレース単位（同じレースの馬をまとめて 1 単位）のブートストラップ。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ODDS_EDGES = [0, 2, 3, 5, 8, 12, 20, 35, 60, 100, 1e9]
ODDS_LABELS = ["o01_<2", "o02_2-3", "o03_3-5", "o04_5-8", "o05_8-12", "o06_12-20", "o07_20-35", "o08_35-60", "o09_60-100", "o10_100+"]


def odds_band(win_odds: pd.Series) -> pd.Series:
    return pd.cut(win_odds, ODDS_EDGES, labels=ODDS_LABELS, right=False).astype(str)


# ------------------------------------------------------------------ レース単位の集計・ブートストラップ
def race_sums(race: np.ndarray, *vals: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """レースごとの合計。戻り値: (レース ID の配列, 各値のレース別合計)。"""
    codes, uniq = pd.factorize(pd.Series(race), sort=False)
    out = [np.bincount(codes, weights=np.asarray(v, dtype=float), minlength=len(uniq)) for v in vals]
    return np.asarray(uniq), out


def boot_roi(race: np.ndarray, ret: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """1 出走 100 円の回収率（%）とレース単位ブートストラップの 95% 区間。n = 件数、races = レース数。"""
    n = len(ret)
    if n == 0:
        return {"n": 0, "races": 0, "roi": float("nan"), "lo": float("nan"), "hi": float("nan")}
    _, (cnt, tot) = race_sums(race, np.ones(n), ret)
    roi = tot.sum() / (100.0 * cnt.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(cnt), size=(n_boot, len(cnt)))
    r = tot[idx].sum(1) / (100.0 * np.maximum(cnt[idx].sum(1), 1))
    return {"n": int(n), "races": int(len(cnt)), "roi": float(100 * roi), "lo": float(100 * np.quantile(r, 0.025)), "hi": float(100 * np.quantile(r, 0.975))}


def race_equal_roi(race: np.ndarray, ret: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """レース均等の回収率: 買ったレースごとの（払戻 / 購入額）の平均。区間はレース単位のブートストラップ。"""
    n = len(ret)
    if n == 0:
        return {"races": 0, "roi": float("nan"), "lo": float("nan"), "hi": float("nan")}
    _, (cnt, tot) = race_sums(race, np.ones(n), ret)
    x = tot / (100.0 * cnt)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    b = x[idx].mean(1)
    return {"races": int(len(x)), "roi": float(100 * x.mean()), "lo": float(100 * np.quantile(b, 0.025)), "hi": float(100 * np.quantile(b, 0.975))}


def boot_diff(race_a: np.ndarray, ret_a: np.ndarray, race_b: np.ndarray, ret_b: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """回収率の差（A − B）。同じレースを共通の単位として再抽出する。p = 差 ≤ 0 の割合（片側）。"""
    ra, (ca, ta) = race_sums(race_a, np.ones(len(ret_a)), ret_a)
    rb, (cb, tb) = race_sums(race_b, np.ones(len(ret_b)), ret_b)
    allr = pd.Index(ra).union(pd.Index(rb))
    A = pd.DataFrame({"c": ca, "t": ta}, index=pd.Index(ra)).reindex(allr).fillna(0.0)
    B = pd.DataFrame({"c": cb, "t": tb}, index=pd.Index(rb)).reindex(allr).fillna(0.0)
    if A["c"].sum() == 0 or B["c"].sum() == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "p_le0": float("nan")}
    d0 = A["t"].sum() / (100.0 * A["c"].sum()) - B["t"].sum() / (100.0 * B["c"].sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(allr), size=(n_boot, len(allr)))
    ac, at, bc, bt = A["c"].to_numpy()[idx].sum(1), A["t"].to_numpy()[idx].sum(1), B["c"].to_numpy()[idx].sum(1), B["t"].to_numpy()[idx].sum(1)
    ok = (ac > 0) & (bc > 0)
    dd = at[ok] / (100.0 * ac[ok]) - bt[ok] / (100.0 * bc[ok])
    return {"diff": float(100 * d0), "lo": float(100 * np.quantile(dd, 0.025)), "hi": float(100 * np.quantile(dd, 0.975)), "p_le0": float((dd <= 0).mean())}


def reality_check(profit: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """White の Reality Check。profit: [レース数, 方策数] のレース別利益。帰無仮説は『どの方策も平均利益 ≤ 0』。"""
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
    return {"t": t, "t_max": float(t.max()), "best": int(t.argmax()), "p_value": float((tmax >= t.max()).mean()), "n_strategies": int(K)}


def max_drawdown(dates: np.ndarray, race: np.ndarray, profit: np.ndarray) -> dict:
    """日付順にレース単位で積んだ累積利益の最大ドローダウン（円、100 円/件）と、最長の連続マイナスのレース数。"""
    df = pd.DataFrame({"d": dates, "r": race, "p": profit}).groupby("r", as_index=False).agg(d=("d", "first"), p=("p", "sum")).sort_values(["d", "r"])
    c = df["p"].cumsum().to_numpy()
    peak = np.maximum.accumulate(np.concatenate([[0.0], c]))[1:]
    dd = float((peak - c).max()) if len(c) else 0.0
    neg = 0
    best = 0
    for x in df["p"].to_numpy():
        neg = neg + 1 if x < 0 else 0
        best = max(best, neg)
    return {"max_drawdown_yen": dd, "longest_losing_races": int(best), "final_profit_yen": float(c[-1]) if len(c) else 0.0}


# ------------------------------------------------------------------ 関係者別の as-of 成績
def entity_year_table(d: pd.DataFrame, key: list[str], extra: dict[str, str] | None = None) -> pd.DataFrame:
    """key × 年 の合計表。n, win, top2, top3, ret_win, ret_place, q（市場の勝率の合計）, ret2_win/ret2_place（二乗和。分散の推定用）。"""
    t = d.assign(ret2_win=d["ret_win"] ** 2, ret2_place=d["ret_place"] ** 2, n=1.0)
    agg = {"n": "sum", "win": "sum", "top2": "sum", "top3": "sum", "ret_win": "sum", "ret_place": "sum", "q_win": "sum", "ret2_win": "sum", "ret2_place": "sum"}
    if "ex_win" in t.columns:
        agg["ex_win"] = "sum"
        agg["ex_place"] = "sum"
    if "p_win" in t.columns:
        agg["p_win"] = "sum"
    if extra:
        agg.update(extra)
    return t.groupby(key + ["year"], observed=True).agg(agg).reset_index()


def asof_table(ty: pd.DataFrame, key: list[str], year: int) -> pd.DataFrame:
    """year より前の年（< year）の合計。リークなし（year 当年の結果は使わない）。"""
    h = ty[ty["year"] < year]
    cols = [c for c in h.columns if c not in key + ["year"]]
    return h.groupby(key, observed=True)[cols].sum().reset_index()


def band_expectation(d: pd.DataFrame, upto_year: int) -> pd.DataFrame:
    """単勝オッズの帯ごとの 1 出走あたりの平均払戻（年 < upto_year の全出走）。単勝・複勝。"""
    h = d[d["year"] < upto_year]
    return h.groupby("obs_band", observed=True)[["ret_win", "ret_place"]].mean().rename(columns={"ret_win": "e_win", "ret_place": "e_place"})


def add_excess(d: pd.DataFrame, upto_by_year: bool = True) -> pd.DataFrame:
    """各出走の『オッズ帯の期待払戻を引いた超過払戻』ex_win / ex_place を足す（期待値は当年より前の年だけで作る。2020 年は自身の年を使う）。"""
    d = d.copy()
    d["obs_band"] = odds_band(d["win_odds"])
    d["ex_win"] = np.nan
    d["ex_place"] = np.nan
    for y in sorted(d["year"].unique()):
        e = band_expectation(d, y if y > d["year"].min() else y + 1)
        m = d["year"] == y
        be = d.loc[m, "obs_band"].map(e["e_win"])
        bp = d.loc[m, "obs_band"].map(e["e_place"])
        d.loc[m, "ex_win"] = d.loc[m, "ret_win"] - be
        d.loc[m, "ex_place"] = d.loc[m, "ret_place"] - bp
    return d


def shrunk_roi(sum_ret: np.ndarray, n: np.ndarray, prior_mean: float, k: float) -> np.ndarray:
    """経験ベイズ風の縮約: (Σret + k × 全体平均) / (100 × (n + k))。"""
    return (np.asarray(sum_ret, dtype=float) + k * prior_mean) / (100.0 * (np.asarray(n, dtype=float) + k))


def reliability(t: pd.DataFrame, col: str, n_min: int = 100) -> dict:
    """関係者別の回収率のばらつきのうち、本物の差（標本の揺れを除いた分）の割合。col は win / place。"""
    x = t[t["n"] >= n_min]
    if len(x) < 5:
        return {"entities": int(len(x)), "reliability": float("nan")}
    r = x[f"ret_{col}"].to_numpy() / (100.0 * x["n"].to_numpy())
    m = x[f"ret_{col}"].to_numpy() / x["n"].to_numpy()
    v = x[f"ret2_{col}"].to_numpy() / x["n"].to_numpy() - m**2
    samp = float(np.mean(v / 1e4 / x["n"].to_numpy()))
    obs = float(np.var(r, ddof=1))
    return {"entities": int(len(x)), "obs_sd": float(np.sqrt(obs)), "sampling_sd": float(np.sqrt(samp)), "reliability": float(max(0.0, obs - samp) / obs) if obs > 0 else float("nan")}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 5:
        return float("nan")
    r = spearmanr(a, b, nan_policy="omit")
    return float(r.statistic if hasattr(r, "statistic") else r[0])


def sharpe_like(profit_by_race: np.ndarray) -> float:
    s = profit_by_race.std(ddof=1)
    return float(profit_by_race.mean() / s * np.sqrt(len(profit_by_race))) if s > 0 else float("nan")
