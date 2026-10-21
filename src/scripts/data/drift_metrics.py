"""ドリフト診断・監視の純関数（入出力なし）。診断（drift_diagnose）と週次監視（drift_monitor）が共有する。

指標はすべて『報告・警報のみ』に使う。モデルや閾値の選択・学習には使わない（config/drift_periods.json の usage を参照）。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

EPS = 1e-6


def bin_ll(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def calib_bins(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """予測確率の分位ごとの平均予測と実現率。"""
    s = pd.Series(np.asarray(p, dtype=float))
    q = pd.qcut(s.rank(method="first"), bins, labels=False)
    df = pd.DataFrame({"p": s, "y": np.asarray(y, dtype=float), "q": q})
    g = df.groupby("q").agg(p=("p", "mean"), y=("y", "mean"), n=("p", "size")).reset_index()
    g["diff"] = g["y"] - g["p"]
    return g


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    g = calib_bins(p, y, bins)
    return float((np.abs(g["p"] - g["y"]) * g["n"]).sum() / g["n"].sum())


def calib_slope(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """y ~ sigmoid(a + b*logit(p)) の (a, b)。ニュートン法（診断用）。"""
    pp = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    z = np.log(pp / (1 - pp))
    X = np.c_[np.ones_like(z), z]
    y = np.asarray(y, dtype=float)
    w = np.array([0.0, 1.0])
    for _ in range(50):
        m = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - m)
        H = (X * (m * (1 - m))[:, None]).T @ X + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        w = w + step
        if np.abs(step).max() < 1e-8:
            break
    return float(w[0]), float(w[1])


# ---------------------------------------------------------------- 分布のずれ

def _edges(ref: np.ndarray, bins: int) -> np.ndarray:
    r = ref[~np.isnan(ref)]
    if len(r) == 0:
        return np.array([])
    e = np.unique(np.quantile(r, np.linspace(0, 1, bins + 1)))
    if len(e) < 2:
        return np.array([])
    e[0], e[-1] = -np.inf, np.inf
    return e


def psi_numeric(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """PSI（基準の分位で区切る。欠損は別の区画）。0.1 未満は小さい、0.25 以上は大きいずれ、が慣例。"""
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    e = _edges(ref, bins)
    if len(e) == 0:
        return float("nan")
    rc = np.histogram(ref[~np.isnan(ref)], e)[0].astype(float)
    cc = np.histogram(cur[~np.isnan(cur)], e)[0].astype(float)
    rc = np.append(rc, float(np.isnan(ref).sum()))
    cc = np.append(cc, float(np.isnan(cur).sum()))
    rp = np.clip(rc / max(rc.sum(), 1), 1e-4, None)
    cp = np.clip(cc / max(cc.sum(), 1), 1e-4, None)
    return float(((cp - rp) * np.log(cp / rp)).sum())


def ks_stat(ref: np.ndarray, cur: np.ndarray) -> float:
    a = np.sort(np.asarray(ref, dtype=float)[~np.isnan(ref)])
    b = np.sort(np.asarray(cur, dtype=float)[~np.isnan(cur)])
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    allv = np.concatenate([a, b])
    return float(np.abs(np.searchsorted(a, allv, side="right") / len(a) - np.searchsorted(b, allv, side="right") / len(b)).max())


def feature_shift(ref: pd.DataFrame, cur: pd.DataFrame, cols: list[str], bins: int = 10) -> pd.DataFrame:
    """列ごとの PSI・KS・欠損率（基準/今回）・平均の変化（基準の標準偏差単位）。数値化できない列は除く。"""
    rows = []
    for c in cols:
        if c not in ref.columns or c not in cur.columns:
            continue
        if ref[c].dtype == object or str(ref[c].dtype) == "category":          # カテゴリ列は共通のコードに直して数値と同じ扱いにする
            codes, _ = pd.factorize(pd.concat([ref[c].astype(str), cur[c].astype(str)], ignore_index=True))
            a, b = codes[: len(ref)].astype(float), codes[len(ref):].astype(float)
        else:
            a = pd.to_numeric(ref[c], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(cur[c], errors="coerce").to_numpy(dtype=float)
        if np.isnan(a).all() and np.isnan(b).all():
            continue
        sd = np.nanstd(a) if (~np.isnan(a)).any() else float("nan")
        rows.append({
            "feature": c, "psi": psi_numeric(a, b, bins), "ks": ks_stat(a, b),
            "miss_ref": float(np.isnan(a).mean()), "miss_cur": float(np.isnan(b).mean()),
            "mean_shift_sd": float((np.nanmean(b) - np.nanmean(a)) / sd) if sd and sd > 0 and (~np.isnan(b)).any() else float("nan"),
        })
    return pd.DataFrame(rows)


def noise_psi(x: np.ndarray, n_a: int, n_b: int, bins: int = 10, reps: int = 50, seed: int = 0) -> float:
    """同じ分布から n_a / n_b 件を引いた PSI の中央値（標本が小さいと PSI が上振れする目安。0 のずれでもこの程度は出る）。"""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 10:
        return float("nan")
    vals = [psi_numeric(rng.choice(x, n_a), rng.choice(x, n_b), bins) for _ in range(reps)]
    return float(np.nanmedian(vals))


# ---------------------------------------------------------------- モデルと市場の差

def race_gap_table(df: pd.DataFrame, p_col: str = "p_win", m_col: str = "p_mkt", y_col: str = "y_win") -> pd.DataFrame:
    """レースごとの (モデル LL − 市場 LL) の合計と頭数。"""
    d = pd.DataFrame({"race_id": df["race_id"], "d": bin_ll(df[y_col], df[p_col]) - bin_ll(df[y_col], df[m_col])})
    g = d.groupby("race_id")["d"].agg(["sum", "size"])
    g.columns = ["sum", "n"]
    return g


def gap_with_ci(df: pd.DataFrame, n_boot: int = 500, seed: int = 0, **kw) -> dict:
    """馬単位の平均差とレース単位ブートストラップの 95% 区間。"""
    g = race_gap_table(df, **kw)
    if len(g) == 0:
        return {"races": 0, "rows": 0, "gap": float("nan"), "lo": float("nan"), "hi": float("nan"), "se": float("nan")}
    s, n = g["sum"].to_numpy(), g["n"].to_numpy()
    gap = float(s.sum() / n.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (n_boot, len(g)))
    est = s[idx].sum(1) / n[idx].sum(1)
    return {"races": int(len(g)), "rows": int(n.sum()), "gap": gap, "lo": float(np.quantile(est, 0.025)),
            "hi": float(np.quantile(est, 0.975)), "se": float(est.std(ddof=1))}


def decompose_gap(base: pd.DataFrame, cur: pd.DataFrame, col: str) -> dict:
    """今回の差 − 基準の差 を、層ごとの『構成比の変化（mix）』と『層内の差の変化（within）』に分ける。

    gap = Σ_s w_s · g_s（w は馬単位の構成比）。cur の差 − base の差 = Σ (w_cur − w_base)·g_base + Σ w_cur·(g_cur − g_base)。
    """
    def agg(d: pd.DataFrame) -> pd.DataFrame:
        x = d.assign(_d=bin_ll(d["y_win"], d["p_win"]) - bin_ll(d["y_win"], d["p_mkt"]))
        t = x.groupby(col, observed=True)["_d"].agg(["mean", "size"])
        t["w"] = t["size"] / t["size"].sum()
        return t

    b, c = agg(base), agg(cur)
    keys = sorted(set(b.index) | set(c.index), key=str)
    mix = within = 0.0
    rows = []
    for k in keys:
        wb, gb = (b.loc[k, "w"], b.loc[k, "mean"]) if k in b.index else (0.0, 0.0)
        wc, gc = (c.loc[k, "w"], c.loc[k, "mean"]) if k in c.index else (0.0, 0.0)
        if k not in b.index:
            gb = gc                                        # 基準に無い層は層内の変化を 0 として扱う（構成の変化に入れる）
        mix += (wc - wb) * gb
        within += wc * (gc - gb)
        rows.append({"stratum": str(k), "w_base": float(wb), "w_cur": float(wc), "gap_base": float(gb), "gap_cur": float(gc),
                     "mix": float((wc - wb) * gb), "within": float(wc * (gc - gb))})
    total = float((c["mean"] * c["w"]).sum() - (b["mean"] * b["w"]).sum())
    return {"total": total, "mix": float(mix), "within": float(within), "rows": rows}


# ---------------------------------------------------------------- 管理図・警報

def control_limits(values: np.ndarray, k: float = 3.0) -> dict:
    """基準期間の値から平均と ±k シグマ。"""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    mu, sd = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan")
    return {"mean": mu, "sd": sd, "ucl": mu + k * sd, "lcl": mu - k * sd, "n": int(len(v)), "k": k}


def moving_average(values: list[float], window: int) -> list[float]:
    s = pd.Series(values, dtype=float)
    return s.rolling(window, min_periods=max(1, window // 2)).mean().tolist()


def consecutive_over(flags: list[bool], k: int) -> bool:
    """末尾から数えて k 回続けて True か。"""
    if len(flags) < k:
        return False
    return all(bool(x) for x in flags[-k:])


def run_length(flags: list[bool]) -> int:
    n = 0
    for x in reversed(flags):
        if not x:
            break
        n += 1
    return n


def cusum(values: list[float], mu: float, sigma: float, k: float = 0.5, h: float = 5.0, direction: str = "up") -> dict:
    """片側 CUSUM（シグマ単位）。S_t = max(0, S_{t-1} + z_t − k)、S_t > h で警報。direction='down' は低下側。
    k=0.5, h=5 は『1.5σ の持続的なずれを数週で検出し、ずれが無ければ約 470 週に 1 回しか鳴らさない』標準的な設定。"""
    s_path, s, first = [], 0.0, None
    for i, v in enumerate(values):
        z = (v - mu) / sigma if sigma and sigma > 0 else 0.0
        z = z if direction == "up" else -z
        s = max(0.0, s + z - k) if not np.isnan(z) else s
        s_path.append(s)
        if first is None and s > h:
            first = i
    return {"path": s_path, "latest": s_path[-1] if s_path else 0.0, "first_alarm_index": first, "alarm_now": bool(s_path and s_path[-1] > h)}


def month_of(date: pd.Series) -> pd.Series:
    return pd.to_datetime(date).dt.strftime("%Y-%m")


def iso_week_of(date: pd.Series) -> pd.Series:
    """JRA の開催は土日なので、週の識別は『その週の月曜日』にする（YYYY-MM-DD）。"""
    d = pd.to_datetime(date)
    return (d - pd.to_timedelta(d.dt.weekday, unit="D")).dt.strftime("%Y-%m-%d")


def fmt(x: float, nd: int = 4) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


# ---------------------------------------------------------------- 週次監視の拡張（Brier・順位相関・市場の連対/複勝・分布・FDR・EWMA）

def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def market_top_probs(df: pd.DataFrame, lam: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """市場の勝率(p_mkt)から、Harville で連対・複勝の確率を作る（市場の連対/複勝の参考値）。レースごと。df は race_id, p_mkt を持つ。"""
    from src.utils.race_probabilities import harville_top2_prob, harville_top3_prob

    p2 = np.zeros(len(df))
    p3 = np.zeros(len(df))
    pos = pd.Series(np.arange(len(df)), index=df.index)
    for _, idx in df.groupby("race_id").indices.items():
        q = df["p_mkt"].to_numpy()[idx] ** lam
        q = q / q.sum()
        p2[idx] = harville_top2_prob(q)
        p3[idx] = harville_top3_prob(q)
    del pos
    return p2, p3


def race_rank_corr(df: pd.DataFrame, p_col: str, y_col: str = "finish_position") -> float:
    """予測が高い順と着順の、レースごとのスピアマン順位相関（符号を反転して、高いほど良い）の平均。"""
    d = df[["race_id", p_col, y_col]].copy()
    d["rp"] = d.groupby("race_id")[p_col].rank(ascending=False, method="average")
    d["rf"] = d.groupby("race_id")[y_col].rank(ascending=True, method="average")
    g = d.groupby("race_id")
    mp, mf = g["rp"].transform("mean"), g["rf"].transform("mean")
    cov = ((d["rp"] - mp) * (d["rf"] - mf)).groupby(d["race_id"]).sum()
    vp = ((d["rp"] - mp) ** 2).groupby(d["race_id"]).sum()
    vf = ((d["rf"] - mf) ** 2).groupby(d["race_id"]).sum()
    r = cov / np.sqrt(vp * vf)
    return float(r.replace([np.inf, -np.inf], np.nan).dropna().mean())


def entropy_rel(df: pd.DataFrame, p_col: str) -> float:
    """レースごとの勝率の分布のエントロピーを log(頭数) で割った値の平均（1 に近いほど平ら）。"""
    d = df[["race_id", p_col]].copy()
    d["h"] = -d[p_col].clip(1e-9, 1) * np.log(d[p_col].clip(1e-9, 1))
    g = d.groupby("race_id")
    return float((g["h"].sum() / np.log(g.size())).mean())


def cluster_gap_se(df: pd.DataFrame, d_col: str = "d") -> tuple[float, float]:
    """行ごとの差 d の平均と、レースでクラスタした標準誤差（比推定量）。"""
    g = df.groupby("race_id")[d_col].agg(["sum", "size"])
    n = g["size"].sum()
    r = g["sum"].sum() / n
    var = (((g["sum"] - r * g["size"]) ** 2).sum()) / (n ** 2) * len(g) / max(len(g) - 1, 1)
    return float(r), float(np.sqrt(var))


def norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


def bh_fdr(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg の q 値。"""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        prev = min(prev, p[i] * n / rank)
        q[i] = prev
    return q.tolist()


def ewma(values: list[float], lam: float, mu: float, sigma: float, L: float = 3.0) -> dict:
    """EWMA 管理図。z_t = λx_t + (1−λ)z_{t−1}、限界 μ ± Lσ√(λ/(2−λ)·(1−(1−λ)^{2t}))。"""
    z, path, ucl, flags = mu, [], [], []
    for t, x in enumerate(values, start=1):
        z = lam * x + (1 - lam) * z if not np.isnan(x) else z
        lim = mu + L * sigma * math.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * t)))
        path.append(z)
        ucl.append(lim)
        flags.append(bool(z > lim))
    return {"path": path, "ucl": ucl, "flags": flags}


def quantile_table(x: np.ndarray, qs: tuple[float, ...] = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)) -> dict:
    x = np.asarray(x, dtype=float)
    return {f"q{int(q * 100):02d}": float(np.quantile(x, q)) for q in qs}


def quantile_edges(x: np.ndarray, bins: int = 10) -> np.ndarray:
    """基準の分位で区切る端点（両端は ±inf）。psi_with_edges の基準。"""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.array([])
    e = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if len(e) < 2:
        return np.array([])
    e[0], e[-1] = -np.inf, np.inf
    return e


def psi_with_edges(edges: np.ndarray, cur: np.ndarray) -> float:
    """基準の分位（各区画の基準の割合は等分）に対する今回の PSI。"""
    c = np.asarray(cur, dtype=float)
    c = c[~np.isnan(c)]
    if len(edges) < 2 or len(c) == 0:
        return float("nan")
    cc = np.histogram(c, edges)[0].astype(float)
    rp = np.full(len(cc), 1.0 / len(cc))
    cp = np.clip(cc / cc.sum(), 1e-4, None)
    return float(((cp - rp) * np.log(cp / rp)).sum())


def race_rank_corr_by_race(df: pd.DataFrame, p_col: str, y_col: str = "finish_position") -> pd.Series:
    """レースごとのスピアマン順位相関（予測が高い順 vs 着順。高いほど予測が着順を当てている）。"""
    d = df[["race_id", p_col, y_col]].copy()
    d["rp"] = d.groupby("race_id")[p_col].rank(ascending=False, method="average")
    d["rf"] = d.groupby("race_id")[y_col].rank(ascending=True, method="average")
    g = d.groupby("race_id")
    mp, mf = g["rp"].transform("mean"), g["rf"].transform("mean")
    cov = ((d["rp"] - mp) * (d["rf"] - mf)).groupby(d["race_id"]).sum()
    vp = ((d["rp"] - mp) ** 2).groupby(d["race_id"]).sum()
    vf = ((d["rf"] - mf) ** 2).groupby(d["race_id"]).sum()
    return (cov / np.sqrt(vp * vf)).replace([np.inf, -np.inf], np.nan)
