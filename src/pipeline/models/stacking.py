"""多層スタッキングの部品（L2 メタ学習器）。L1 = 各モデルの OOF 予測（valid 2022〜2024）。

メタ学習器はすべて leave-one-year-out（他の年の OOF だけで当てはめ、当該年に適用）で評価する。
入力は、メンバーの logit に加えて、メンバー間の平均・ばらつき・レース内の位置などの文脈特徴（レース内の相対構造をメタ層に渡す）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.models import ensemble as en
from src.pipeline.models import win_models as wm

TOT = {"win": 1.0, "top2": 2.0, "top3": 3.0}


def meta_features(Z: np.ndarray, race: np.ndarray, names: list[str], with_context: bool = True) -> pd.DataFrame:
    """Z: (行, メンバー数) の logit。戻り値: メタ特徴の表。"""
    f = pd.DataFrame(Z, columns=[f"z_{n}" for n in names])
    if not with_context:
        return f
    mean = Z.mean(1)
    f["z_mean"], f["z_std"] = mean, Z.std(1)
    g = pd.DataFrame({"r": race, "m": mean})
    f["m_rank"] = g.groupby("r")["m"].rank(ascending=False, pct=True).to_numpy()
    f["m_gap"] = mean - g.groupby("r")["m"].transform("max").to_numpy()
    f["m_z"] = (mean - g.groupby("r")["m"].transform("mean").to_numpy()) / g.groupby("r")["m"].transform("std").replace(0, np.nan).fillna(1.0).to_numpy()
    f["log_field"] = np.log(g.groupby("r")["m"].transform("size").to_numpy())
    return f


def loyo_lgb_meta(F: dict[str, pd.DataFrame], Y: dict, years: np.ndarray, race: np.ndarray, threads: int = 4, seed: int = 0, params: dict | None = None,
                  n_rounds: int = 300) -> dict[str, np.ndarray]:
    """LightGBM（浅く強く正則化）をメタ学習器にする。target ごとに別モデル。戻り値は合計をそろえた確率。"""
    import lightgbm as lgb

    p = dict({"objective": "binary", "learning_rate": 0.03, "num_leaves": 6, "min_data_in_leaf": 400, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
              "lambda_l2": 20.0, "verbosity": -1, "seed": seed, "num_threads": threads, "deterministic": True, "force_row_wise": True}, **(params or {}))
    out = {t: np.zeros(len(years)) for t in TOT}
    for t in TOT:
        X = F[t]
        for y in np.unique(years):
            tr, te = years != y, years == y
            b = lgb.train(p, lgb.Dataset(X[tr], label=Y[t][tr]), num_boost_round=n_rounds)
            out[t][te] = b.predict(X[te])
    return {t: wm.normalize_to_total(race, np.clip(out[t], 1e-6, 1 - 1e-6), TOT[t]) for t in TOT}


def loyo_nn_meta(Fall: pd.DataFrame, Y: dict, years: np.ndarray, race: np.ndarray, hidden=(32, 16), epochs: int = 12, seeds=(0, 1, 2), lr: float = 2e-3, wd: float = 1e-3,
                 dropout: float = 0.1) -> dict[str, np.ndarray]:
    """小さな NN（レース内の文脈つき listwise softmax + BCE）をメタ学習器にする。"""
    from src.pipeline.models import zoo

    sums = {t: np.zeros(len(years)) for t in TOT}
    for y in np.unique(years):
        tr, te = years != y, years == y
        ytr = {"win": Y["win"][tr], "top2": Y["top2"][tr], "top3": Y["top3"][tr]}
        acc = {t: np.zeros(int(te.sum())) for t in TOT}
        for s in seeds:
            o = zoo.nn_fit_predict(Fall[tr].reset_index(drop=True), ytr, race[tr], Fall[te].reset_index(drop=True), race[te], seed=s, hidden=hidden, epochs=epochs, lr=lr, wd=wd, dropout=dropout)
            fp = zoo.finalize(race[te], {"win": wm.softmax_by_race(race[te], o["win_score"]), "top2": zoo.sigmoid(o["top2_logit"]), "top3": zoo.sigmoid(o["top3_logit"])})
            for t in TOT:
                acc[t] += fp[t] / len(seeds)
        for t in TOT:
            sums[t][te] = acc[t]
    return {t: wm.normalize_to_total(race, np.clip(sums[t], 1e-6, 1 - 1e-6), TOT[t]) for t in TOT}


def logit_mean(parts: list[np.ndarray], race: np.ndarray, total: float) -> np.ndarray:
    z = np.mean([en.logit(np.clip(p, 1e-6, 1 - 1e-6)) for p in parts], axis=0)
    return wm.normalize_to_total(race, en.sigmoid(z), total)


def _pad_features(F: np.ndarray, race: np.ndarray):
    codes, uniq = pd.factorize(race)
    n_max = int(np.bincount(codes).max())
    pos = pd.Series(codes).groupby(codes).cumcount().to_numpy()
    A = np.zeros((len(uniq), n_max, F.shape[1]))
    M = np.zeros((len(uniq), n_max), bool)
    A[codes, pos] = F
    M[codes, pos] = True
    return A, M, codes, pos


def fit_clogit(F: np.ndarray, race: np.ndarray, y_win: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    """条件付きロジット（レース内 softmax）の重みを学習する。重みの符号・合計に制約なし（温度は重みの大きさに吸収される）。勝ち馬がちょうど 1 頭のレースだけ使う。"""
    from scipy.optimize import minimize

    A, M, codes, pos = _pad_features(F, race)
    Wn = np.zeros(M.shape, bool)
    Wn[codes, pos] = y_win > 0.5
    ok = Wn.sum(1) == 1
    A, M, Wn = A[ok], M[ok], Wn[ok]
    K = A.shape[2]
    Aw = (A * Wn[..., None]).sum(1)

    def f(w):
        s = np.where(M, A @ w, -np.inf)
        mx = s.max(1, keepdims=True)
        e = np.exp(s - mx)
        z = e.sum(1, keepdims=True)
        p = e / z
        nll = -((np.where(Wn, s, 0.0).sum(1)) - (mx[:, 0] + np.log(z[:, 0]))).mean() + 0.5 * l2 * (w ** 2).sum()
        g = -(Aw - (p[..., None] * A).sum(1)).mean(0) + l2 * w
        return nll, g

    r = minimize(f, np.full(K, 1.0 / K), jac=True, method="L-BFGS-B")
    return r.x


def loyo_clogit_meta(Zt: dict[str, np.ndarray], Y: dict, years: np.ndarray, race: np.ndarray, cross: bool = False, C: float = 1.0) -> dict[str, np.ndarray]:
    """L2 = 勝率は条件付きロジット（レース内 softmax、重み自由）、連対率・複勝率は切片つきロジスティック回帰（cross=True なら 3 つの目的の logit を全部入力にする）。"""
    from sklearn.linear_model import LogisticRegression

    out = {t: np.zeros(len(years)) for t in TOT}
    allZ = np.concatenate([Zt[t] for t in TOT], axis=1)
    for y in np.unique(years):
        tr, te = years != y, years == y
        Fw = allZ if cross else Zt["win"]
        w = fit_clogit(Fw[tr], race[tr], Y["win"][tr])
        pw = wm.softmax_by_race(race[te], Fw[te] @ w)
        out["win"][te] = pw
        for t in ("top2", "top3"):
            Ft = allZ if cross else Zt[t]
            lr = LogisticRegression(C=C, max_iter=500).fit(Ft[tr], Y[t][tr])
            out[t][te] = lr.predict_proba(Ft[te])[:, 1]
    return {t: wm.normalize_to_total(race, np.clip(out[t], 1e-6, 1 - 1e-6), TOT[t]) for t in TOT}
