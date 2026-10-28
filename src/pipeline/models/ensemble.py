"""tp-05e: OOF 予測だけでブレンド重み・スタッキングを学習する。

対象ごと（p_win・p_top2・p_top3）に別々に学習し、最後にレース内で合計 1・2・3 にそろえる。
過剰適合の見積もりは leave-one-year-out（1 年を外して他の年で重みを決め、外した年で評価）で行う。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from src.pipeline.models import win_models as wm

EPS = 1e-9


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def ll(p: np.ndarray, y: np.ndarray) -> float:
    return wm.binary_logloss(p, y)


def fit_simplex_weights(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """logit 空間のブレンド p = σ(Σ w_m z_m)、w ≥ 0・Σw = 1 で LogLoss を最小にする重み。"""
    M = Z.shape[1]
    w0 = np.full(M, 1.0 / M)

    def f(w):
        return ll(sigmoid(Z @ w), y)

    r = minimize(f, w0, method="SLSQP", bounds=[(0, 1)] * M, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"maxiter": 200, "ftol": 1e-10})
    w = np.clip(r.x, 0, None)
    return w / w.sum()


def greedy_selection(P: np.ndarray, y: np.ndarray, steps: int = 20) -> np.ndarray:
    """Caruana の前進選択（重複あり）。確率の算術平均の LogLoss が下がる限りモデルを足し、選択回数を重みにする。"""
    M = P.shape[1]
    counts = np.zeros(M)
    cur = np.zeros(len(y))
    best = 9.0
    for k in range(1, steps + 1):
        scores = [ll((cur * (k - 1) + P[:, m]) / k, y) for m in range(M)]
        m = int(np.argmin(scores))
        if scores[m] >= best - 1e-9 and k > 1:
            break
        best = scores[m]
        counts[m] += 1
        cur = (cur * (k - 1) + P[:, m]) / k
    return counts / counts.sum()


def fit_stack(Z: np.ndarray, y: np.ndarray, C: float = 1.0) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=500).fit(Z, y)


METHODS = ("mean", "logit_mean", "weighted_logit", "greedy", "stack_lr")


def blend(method: str, P: np.ndarray, Z: np.ndarray, state) -> np.ndarray:
    """state は fit_method の戻り値。"""
    if method == "mean":
        return P.mean(axis=1)
    if method == "logit_mean":
        return sigmoid(Z.mean(axis=1))
    if method == "weighted_logit":
        return sigmoid(Z @ state)
    if method == "greedy":
        return P @ state
    if method == "stack_lr":
        return state.predict_proba(Z)[:, 1]
    raise ValueError(method)


def fit_method(method: str, P: np.ndarray, Z: np.ndarray, y: np.ndarray):
    if method in ("mean", "logit_mean"):
        return None
    if method == "weighted_logit":
        return fit_simplex_weights(Z, y)
    if method == "greedy":
        return greedy_selection(P, y)
    if method == "stack_lr":
        return fit_stack(Z, y)
    raise ValueError(method)


def loyo_eval(method: str, P: np.ndarray, years: np.ndarray, race: np.ndarray, y: np.ndarray, total: float) -> tuple[float, np.ndarray]:
    """1 年を外して他の年で学習し、外した年を予測。戻り値: (正規化後の全体 LogLoss, 正規化後の予測)。"""
    Z = logit(P)
    out = np.zeros(len(y))
    for yr in np.unique(years):
        va = years == yr
        st = fit_method(method, P[~va], Z[~va], y[~va])
        out[va] = blend(method, P[va], Z[va], st)
    out = wm.normalize_to_total(race, out, total)
    return ll(out, y), out


def apply_spec(entry: dict, P: np.ndarray) -> np.ndarray:
    """ensemble_spec.json の 1 対象分（method と重み・係数）を、モデル別の確率 P（n × M）に適用する。正規化前の確率を返す。"""
    m = entry["method"]
    Z = logit(P)
    if m == "mean":
        return P.mean(axis=1)
    if m == "logit_mean":
        return sigmoid(Z.mean(axis=1))
    if m == "weighted_logit":
        return sigmoid((Z * np.asarray(entry["weights"])).sum(axis=1))      # 行列積は行数でブロック分けが変わり末位がずれるので、行ごとの和にする
    if m == "greedy":
        return (P * np.asarray(entry["weights"])).sum(axis=1)
    if m == "stack_lr":
        return sigmoid((Z * np.asarray(entry["coef"])).sum(axis=1) + entry["intercept"])
    raise ValueError(m)
