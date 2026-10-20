"""tp-51: クロス対象ロジスティック回帰（3 対象 × メンバーの logit を入力に、対象ごとに LR）と、現行の合成との L3（logit 平均）。

採用候補（S11e）: 勝率・連対は『現行の合成（weighted_logit / greedy）』と『クロス対象 LR』の logit 平均、複勝は現行のまま。
推論側の追加は、対象ごとの標準化の平均・尺度と LR の係数（3M 個 + 切片）だけで、メンバーの追加や再学習は要らない。

    spec = fit_cross_lr(P_oof, y_oof, race_ids)          # P_oof[t]: (n, M) のメンバー確率（OOF）、y_oof[t]: 0/1
    p = apply_l3(spec, P, base, race_ids)                       # base[t]: 現行の合成の確率（正規化前）。戻り値: 対象ごとの確率（正規化前）
"""

from __future__ import annotations

import numpy as np

TARGETS = ("win", "top2", "top3")
L3_TARGETS = ("win", "top2")
EPS = 1e-9


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def cross_features(P: dict[str, np.ndarray], race_ids: np.ndarray) -> np.ndarray:
    """(n, 3 × (M + 3)): 対象（勝率・連対・複勝）ごとに、各メンバーの logit（M）+ メンバー平均 + メンバー間の標準偏差 + 平均 logit のレース内 z。
    評価（tp51_stack.meta_features の文脈なし）と同じ列。"""
    import pandas as pd

    cols = []
    for t in TARGETS:
        Z = _logit(P[t])
        zm = Z.mean(1)
        g = pd.Series(zm).groupby(np.asarray(race_ids))
        z_race = (zm - g.transform("mean").to_numpy()) / (g.transform("std").to_numpy() + 1e-6)
        cols += [Z, zm[:, None], Z.std(1, keepdims=True), z_race[:, None]]
    return np.hstack(cols)


def fit_cross_lr(P: dict[str, np.ndarray], y: dict[str, np.ndarray], race_ids: np.ndarray, C: float = 0.1) -> dict:
    from sklearn.linear_model import LogisticRegression

    Z = cross_features(P, race_ids)
    mu, sd = Z.mean(0), Z.std(0) + 1e-12
    spec = {"C": C, "mean": mu.tolist(), "scale": sd.tolist(), "targets": {}}
    for t in TARGETS:
        m = LogisticRegression(C=C, max_iter=1000).fit((Z - mu) / sd, y[t])
        spec["targets"][t] = {"coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0])}
    return spec


def apply_cross_lr(spec: dict, P: dict[str, np.ndarray], race_ids: np.ndarray) -> dict[str, np.ndarray]:
    Z = (cross_features(P, race_ids) - np.asarray(spec["mean"])) / np.asarray(spec["scale"])
    return {t: _sigmoid(Z @ np.asarray(spec["targets"][t]["coef"]) + spec["targets"][t]["intercept"]) for t in TARGETS}


def apply_l3(spec: dict, P: dict[str, np.ndarray], base: dict[str, np.ndarray], race_ids: np.ndarray, w_lr: float = 0.5) -> dict[str, np.ndarray]:
    """勝率・連対は base と LR の logit 平均（重み w_lr が LR）、複勝は base のまま。"""
    lr = apply_cross_lr(spec, P, race_ids)
    out = {}
    for t in TARGETS:
        out[t] = _sigmoid((1 - w_lr) * _logit(base[t]) + w_lr * _logit(lr[t])) if t in L3_TARGETS else np.asarray(base[t])
    return out
