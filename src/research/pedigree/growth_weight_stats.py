"""馬体重の増減とめぐ指数の変化の結びつき（成長曲線 × 血統）の調査で使う、純粋な統計関数。

- クラスタ頑健な OLS（同じレースの馬を独立と数えすぎないよう race_id でクラスタ）
- 経験ベイズ（DerSimonian-Laird）の縮約
- 成長段階（2歳前半〜5歳以上）の割り当て、馬 ID の先頭 4 桁（生年）からの年齢
- 順位相関、半分に分けた再現性
入出力は numpy / pandas のみ。データの読み込みは growth_weight_dataset.py。
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

STAGES = ["2歳前半", "2歳後半", "3歳前半", "3歳後半", "4歳", "5歳以上"]
EARLY_STAGES = ("2歳前半", "2歳後半", "3歳前半", "3歳後半")


def age_from_id(horse_id: str, race_year: int) -> int | None:
    """netkeiba の馬 ID は先頭 4 桁が生年。JRA の馬齢（生年を 0 歳とし、翌年を 1 歳）で race_year − 生年。"""
    s = str(horse_id or "")
    if len(s) < 4 or not s[:4].isdigit():
        return None
    a = int(race_year) - int(s[:4])
    return a if 1 <= a <= 12 else None


def stage_of(age: int | None, month: int) -> str | None:
    """成長段階。2歳は 9 月までを前半、3歳は 6 月までを前半（春の世代戦まで）。"""
    if age is None or age < 2:
        return None
    if age == 2:
        return "2歳前半" if month <= 9 else "2歳後半"
    if age == 3:
        return "3歳前半" if month <= 6 else "3歳後半"
    return "4歳" if age == 4 else "5歳以上"


def design_with_dummies(cols: dict[str, np.ndarray], cats: dict[str, Sequence], ref: dict[str, object]) -> tuple[np.ndarray, list[str]]:
    """数値列 cols と、カテゴリ列 cats（基準水準 ref を除いてダミー化）から切片つきの計画行列を作る。"""
    names = ["const"]
    parts = [None]
    n = None
    for k, v in cols.items():
        v = np.asarray(v, dtype=float)
        n = len(v)
        parts.append(v)
        names.append(k)
    for k, v in cats.items():
        arr = np.asarray(v, dtype=object)
        n = len(arr)
        levels = sorted({x for x in arr if x == x and x is not None}, key=str)
        for lv in levels:
            if lv == ref.get(k):
                continue
            parts.append((arr == lv).astype(float))
            names.append(f"{k}={lv}")
    parts[0] = np.ones(n)
    return np.column_stack(parts), names


def ols_cluster(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    """OLS とクラスタ頑健な標準誤差（CR1）。groups は race_id など。係数・SE・95% 区間・クラスタ数を返す。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    g = inv.max() + 1
    # スコアをクラスタごとに合計
    S = np.zeros((g, k))
    np.add.at(S, inv, X * resid[:, None])
    meat = S.T @ S
    adj = (g / max(g - 1, 1)) * ((n - 1) / max(n - k, 1))
    cov = adj * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return {"beta": beta, "se": se, "lo": beta - 1.96 * se, "hi": beta + 1.96 * se, "n": int(n), "clusters": int(g), "resid": resid}


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """順位相関（同順位は平均順位）。2 点未満・分散 0 は nan。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return float("nan")

    def rank(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x))
        r[order] = np.arange(len(x), dtype=float)
        # 同順位は平均
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.zeros(len(cnt))
        np.add.at(sums, inv, r)
        return (sums / cnt)[inv]

    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def eb_shrink(est: Sequence[float], se: Sequence[float]) -> dict:
    """経験ベイズ（DerSimonian-Laird）で、各推定値を母集団平均へ縮約する。
    返り値: mu（逆分散加重の平均）, tau2（真の個体差の分散。負は 0）, shrunk（縮約後）, weight（個体の重み = tau2/(tau2+se²)）。
    tau2 が 0 に近いほど『個体差は検出できない』。"""
    est = np.asarray(est, dtype=float)
    se = np.asarray(se, dtype=float)
    ok = ~(np.isnan(est) | np.isnan(se)) & (se > 0)
    if ok.sum() < 3:
        return {"mu": float("nan"), "tau2": float("nan"), "shrunk": np.full(len(est), np.nan), "weight": np.full(len(est), np.nan), "k": int(ok.sum())}
    w = 1.0 / se[ok] ** 2
    mu_fe = float(np.sum(w * est[ok]) / np.sum(w))
    q = float(np.sum(w * (est[ok] - mu_fe) ** 2))
    c = float(np.sum(w) - np.sum(w**2) / np.sum(w))
    tau2 = max(0.0, (q - (ok.sum() - 1)) / c) if c > 0 else 0.0
    wr = 1.0 / (se[ok] ** 2 + tau2)
    mu = float(np.sum(wr * est[ok]) / np.sum(wr))
    shr = np.full(len(est), np.nan)
    wt = np.full(len(est), np.nan)
    wt_ok = tau2 / (tau2 + se[ok] ** 2)
    shr[ok] = mu + wt_ok * (est[ok] - mu)
    wt[ok] = wt_ok
    return {"mu": mu, "tau2": float(tau2), "shrunk": shr, "weight": wt, "k": int(ok.sum()), "q": q}


def split_half_corr(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    """半分ずつ別々に出した個体の値の順位相関（再現性）。"""
    return spearman(values_a, values_b)


def slope_se(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """1 変数の回帰（切片あり）の傾きと標準誤差。n<4 や x の分散 0 は nan。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 4:
        return float("nan"), float("nan"), n
    xc = x - x.mean()
    sxx = float(np.sum(xc**2))
    if sxx <= 0:
        return float("nan"), float("nan"), n
    b = float(np.sum(xc * (y - y.mean())) / sxx)
    resid = y - y.mean() - b * xc
    s2 = float(np.sum(resid**2) / (n - 2))
    return b, math.sqrt(s2 / sxx), n


def bin_labels(x: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """edges（昇順）で区切ったビン名。例 edges=[-10,-4,-1,2,5,10] → '<-10', '-10〜-5', ...。"""
    x = np.asarray(x, dtype=float)
    names = [f"≤{int(edges[0]) - 1}"]
    for lo, hi in zip(edges[:-1], edges[1:]):
        names.append(f"{int(lo)}〜{int(hi) - 1}")
    names.append(f"≥{int(edges[-1])}")
    idx = np.digitize(x, edges)
    out = np.array([names[i] for i in idx], dtype=object)
    out[np.isnan(x)] = None
    return out


def group_slopes(gid: np.ndarray, x: np.ndarray, y: np.ndarray, min_n: int = 4) -> dict:
    """グループ（gid は 0..G-1 の整数）ごとの 1 変数回帰の傾きと標準誤差を、ループなしでまとめて出す。
    n<min_n や x の分散 0 は nan。"""
    gid = np.asarray(gid, dtype=np.int64)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    G = int(gid.max()) + 1 if len(gid) else 0
    n = np.bincount(gid, minlength=G).astype(float)
    sx = np.bincount(gid, weights=x, minlength=G)
    sy = np.bincount(gid, weights=y, minlength=G)
    sxx = np.bincount(gid, weights=x * x, minlength=G)
    sxy = np.bincount(gid, weights=x * y, minlength=G)
    syy = np.bincount(gid, weights=y * y, minlength=G)
    with np.errstate(divide="ignore", invalid="ignore"):
        cxx = sxx - sx * sx / n
        cxy = sxy - sx * sy / n
        cyy = syy - sy * sy / n
        b = cxy / cxx
        sse = np.clip(cyy - b * cxy, 0, None)
        se = np.sqrt(sse / (n - 2) / cxx)
    bad = (n < min_n) | ~(cxx > 1e-12)
    b[bad] = np.nan
    se[bad] = np.nan
    return {"slope": b, "se": se, "n": n.astype(int)}
