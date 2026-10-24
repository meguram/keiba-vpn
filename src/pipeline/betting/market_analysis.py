"""馬券戦略の比較用: 優位の有無の検定、市場だけから作る着順分布の当てはめ、期待回収率の較正。

- ``fit_global_alphas``: 市場の単勝確率だけから作った割引 Harville（Benter 型）の指数 α2・α3 を、実際の 1〜3 着の並び（3連単）の対数尤度で最尤推定する。
- ``lr_test_model_weight``: Benter ブレンドの a（モデルの重み）が 0 かの尤度比検定と、レース単位 bootstrap の信頼区間。
- ``ev_slope``: 期待回収率の予測値に対する実現回収率の回帰の傾き（1 なら較正済み、0 以下なら優位なし）。レース単位 bootstrap。
- ``IsotonicEV``: 学習年で当てはめた「期待回収率 → 実現回収率」の単調回帰（券種別）。選別時の推定誤差（optimizer's curse）を縮める。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from src.pipeline.inference.exotic_probabilities import RankDistribution


def finishing_orders(payoff: pd.DataFrame) -> dict[str, tuple[int, int, int]]:
    """3連単の払戻 ``a>b>c`` から、1・2・3 着の馬番を取り出す（同着などで一意でないレースは除く）。"""
    d = payoff[(payoff["bet_type"] == "sanrentan") & payoff["combo"].notna() & ~payoff["ambiguous"].astype(bool)]
    cnt = d.groupby("race_id").size()
    d = d[d["race_id"].isin(cnt[cnt == 1].index)]
    out = {}
    for rid, c in zip(d["race_id"], d["combo"]):
        parts = str(c).split(">")
        if len(parts) == 3:
            try:
                out[rid] = (int(parts[0]), int(parts[1]), int(parts[2]))
            except ValueError:
                continue
    return out


def order_groups(win_probs: dict[str, tuple[np.ndarray, np.ndarray]], orders: dict[str, tuple[int, int, int]]) -> list[tuple[np.ndarray, int, int, int]]:
    """win_probs: race_id → (馬番の配列, 勝率の配列)。戻り値: [(勝率, 1 着の位置, 2 着の位置, 3 着の位置)]。"""
    out = []
    for rid, (hn, w) in win_probs.items():
        o = orders.get(rid)
        if o is None:
            continue
        pos = {int(h): i for i, h in enumerate(hn)}
        if all(x in pos for x in o) and len(hn) >= 3:
            out.append((np.asarray(w, dtype=float), pos[o[0]], pos[o[1]], pos[o[2]]))
    return out


def order_nll(groups, alpha2: float, alpha3: float) -> float:
    """1〜3 着の並びの平均負対数尤度（割引 Harville）。"""
    s = 0.0
    for w, i, j, k in groups:
        d = RankDistribution(w, alpha2, alpha3)
        s -= np.log(max(d.sanrentan(i, j, k), 1e-300))
    return s / len(groups)


def fit_global_alphas(groups, x0=(0.9, 0.9)) -> tuple[float, float, float]:
    """戻り値: (α2, α3, その時の平均負対数尤度)。"""
    r = minimize(lambda a: order_nll(groups, a[0], a[1]), np.array(x0), method="L-BFGS-B", bounds=[(0.3, 1.5), (0.3, 1.5)], options={"maxiter": 40})
    return float(r.x[0]), float(r.x[1]), float(r.fun)


def _pad(groups):
    """[(log p_model, log p_market, 勝ち馬の位置)] → (LM, LK, MASK, W)（レース × 頭数の配列）。"""
    n = max(len(g[0]) for g in groups)
    R = len(groups)
    LM, LK = np.zeros((R, n)), np.zeros((R, n))
    M = np.zeros((R, n), bool)
    W = np.zeros(R, int)
    for r, (lm, lk, w) in enumerate(groups):
        k = len(lm)
        LM[r, :k], LK[r, :k], M[r, :k], W[r] = lm, lk, True, w
    return LM, LK, M, W


def _nll_arrays(LM, LK, M, W, a: float, b: float) -> float:
    z = np.where(M, a * LM + b * LK, -np.inf)
    mx = z.max(1, keepdims=True)
    lse = mx[:, 0] + np.log(np.exp(z - mx).sum(1))
    return float(-(z[np.arange(len(W)), W] - lse).mean())


def _fit_arrays(LM, LK, M, W, free_a: bool = True) -> tuple[float, float]:
    if free_a:
        r = minimize(lambda x: _nll_arrays(LM, LK, M, W, x[0], x[1]), np.array([0.5, 0.5]), method="L-BFGS-B", bounds=[(0.0, 2.0), (0.0, 2.0)])
        return float(r.x[0]), float(r.x[1])
    r = minimize(lambda x: _nll_arrays(LM, LK, M, W, 0.0, x[0]), np.array([1.0]), method="L-BFGS-B", bounds=[(0.0, 2.0)])
    return 0.0, float(r.x[0])


def lr_test_model_weight(groups, n_boot: int = 200, seed: int = 0) -> dict:
    """a（モデル）= 0 の尤度比検定と、(a, b) のレース単位 bootstrap 区間。groups は fit_benter と同じ形式。"""
    LM, LK, M, W = _pad(groups)
    a, b = _fit_arrays(LM, LK, M, W)
    nll_full = _nll_arrays(LM, LK, M, W, a, b)
    _, b0 = _fit_arrays(LM, LK, M, W, free_a=False)
    nll_null = _nll_arrays(LM, LK, M, W, 0.0, b0)
    R = len(groups)
    lr = 2.0 * R * (nll_null - nll_full)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        ii = rng.integers(0, R, R)
        bs.append(_fit_arrays(LM[ii], LK[ii], M[ii], W[ii]))
    bs = np.array(bs)
    return {"a": a, "b": b, "b_null": b0, "nll_full": nll_full, "nll_null": float(nll_null), "gain_per_race": nll_null - nll_full, "lr": float(lr), "p_value": float(stats.chi2.sf(max(lr, 0.0), 1)),
            "a_lo": float(np.quantile(bs[:, 0], 0.025)), "a_hi": float(np.quantile(bs[:, 0], 0.975)), "b_lo": float(np.quantile(bs[:, 1], 0.025)), "b_hi": float(np.quantile(bs[:, 1], 0.975))}


def ev_slope(df: pd.DataFrame, x_col: str = "ev_hat", n_boot: int = 300, seed: int = 0) -> dict:
    """実現損益（mult − 1）を期待損益（x − 1）に回帰した傾き。df は race_id・mult と x_col（= p×odds）を持つ。"""
    x = df[x_col].to_numpy() - 1.0
    y = df["mult"].to_numpy() - 1.0
    def slope(ii):
        xx, yy = x[ii], y[ii]
        xm = xx - xx.mean()
        return float((xm * (yy - yy.mean())).sum() / max((xm ** 2).sum(), 1e-12))
    full = slope(np.arange(len(x)))
    codes, uniq = pd.factorize(df["race_id"])
    idx = [np.flatnonzero(codes == c) for c in range(len(uniq))]
    rng = np.random.default_rng(seed)
    bs = [slope(np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])) for _ in range(n_boot)]
    return {"slope": full, "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975)), "n": int(len(df))}


class IsotonicEV:
    """券種ごとの「期待回収率 x = p×odds → 実現回収率」の単調回帰。fit は学習年だけで行う。"""

    def __init__(self):
        self.models = {}

    def fit(self, df: pd.DataFrame, x: np.ndarray) -> "IsotonicEV":
        from sklearn.isotonic import IsotonicRegression

        t = df["type"].to_numpy()
        y = df["mult"].to_numpy()
        for ty in np.unique(t):
            m = t == ty
            self.models[ty] = IsotonicRegression(out_of_bounds="clip", increasing=True).fit(np.clip(x[m], 0, 5), y[m])
        return self

    def predict(self, df: pd.DataFrame, x: np.ndarray) -> np.ndarray:
        out = np.zeros(len(df))
        t = df["type"].to_numpy()
        for ty, m_ in self.models.items():
            m = t == ty
            if m.any():
                out[m] = m_.predict(np.clip(x[m], 0, 5))
        return out
