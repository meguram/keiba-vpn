"""段 2: 血統だけの勝率・複勝率のモデル（LightGBM）と評価。

- 学習・評価は 2017〜2024 年（2016 年は as-of の助走）。
- 分け方: random（開催の無作為。split 0 学習・1 検証・2 保留）/ forward（2017〜21 学習・22 検証・23〜24 保留）/ dam（母の単位）。
- 勝率: 二値で当て、レースの中で和 1 に揃え、検証で温度 1 つを当てる。複勝: 3 着内で当て、和を min(3,頭数) に揃える。
- 物差し: 一様（1/頭数）と市場（確定オッズの 1/odds を和 1 に。基準線としてだけ使う）の対数損失。
"""
from __future__ import annotations

import json
import re

import lightgbm as lgb
import numpy as np
import pandas as pd

CTX = {"surf", "dband", "going", "cls", "ageb", "sexb", "by", "career", "prev_date", "prev_dist", "n_surf_before",
       "debut", "layoff", "first_surf", "dchg"}
FEAT_RE = re.compile(r"^(sire|bms|ss|sss|bms2|dam|dd|nick|nickl|full|anc)_|^ib_|^rel_")
SUB_RE = r"_(deb|lay|fs|dchg)_"      # 副の版（馬の状態の鍵）の列


def feature_cols(X: pd.DataFrame, drop: tuple[str, ...] = (), sub: bool = False) -> list[str]:
    """許可の正規表現に当たる血統の列。副の版（馬の状態の鍵）は sub=True のときだけ。"""
    cols = [c for c in X.columns if FEAT_RE.match(c) and c not in CTX]
    if not sub:
        cols = [c for c in cols if not re.search(SUB_RE, c)]
    return [c for c in cols if not any(re.search(d, c) for d in drop)]


def split_labels(X: pd.DataFrame, how: str) -> np.ndarray:
    y = X["date"].dt.year.to_numpy()
    if how == "random":
        return X["split"].to_numpy()
    if how == "forward":
        return np.where(y <= 2021, 0, np.where(y == 2022, 1, 2))
    if how == "dam":     # 2 代母の単位（いとこまで同じ側。1 号の R2）
        h = pd.util.hash_pandas_object(X["dd"].astype(str), index=False).to_numpy() % 100
        return np.where(h < 70, 0, np.where(h < 85, 1, 2))
    raise ValueError(how)


PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=63, min_data_in_leaf=400, feature_fraction=0.5,
              bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=6, seed=7)


def race_norm(p: np.ndarray, rid: np.ndarray, total: np.ndarray | float = 1.0, temp: float = 1.0) -> np.ndarray:
    """ロジットに温度を掛け、レースの中で和を total に揃える（複勝は min(3,n) に、上限 0.999 で）。"""
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6)) * temp
    w = np.exp(lp - pd.Series(lp).groupby(rid).transform("max").to_numpy())
    q = w / pd.Series(w).groupby(rid).transform("sum").to_numpy() * total
    return np.clip(q, 1e-6, 0.999)


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def race_ll(y, p, rid):
    """勝ち: レースの勝ち馬の -log p の平均（条件付きロジットの対数損失）。"""
    m = y == 1
    return float(-np.mean(np.log(np.clip(p[m], 1e-9, 1))))


def fit_eval(X: pd.DataFrame, target: str = "win", how: str = "random", drop=(), rounds: int = 3000,
             cols: list[str] | None = None, verbose: bool = True) -> dict:
    X = X[(X["date"].dt.year >= 2017) & (X["date"].dt.year <= 2024)].reset_index(drop=True)   # 2025 年以降は開発に入れない
    cols = cols or feature_cols(X, drop)
    sp = split_labels(X, how)
    y = X[target].to_numpy()
    rid = X["race_id"].to_numpy()
    n = X["n_run"].to_numpy()
    total = 1.0 if target == "win" else np.minimum(3, n)
    tr, va, te = sp == 0, sp == 1, sp == 2
    dtr = lgb.Dataset(X.loc[tr, cols], y[tr], free_raw_data=False)
    dva = lgb.Dataset(X.loc[va, cols], y[va], reference=dtr)
    m = lgb.train(PARAMS, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    p = m.predict(X[cols], num_iteration=m.best_iteration)
    # 温度を検証で当てる
    best = (1e9, 1.0)
    for t in np.arange(0.6, 1.81, 0.05):
        q = race_norm(p[va], rid[va], total if np.isscalar(total) else total[va], t)
        s = race_ll(y[va], q, rid[va]) if target == "win" else ll(y[va], q)
        best = min(best, (s, t))
    temp = best[1]
    q = race_norm(p, rid, total, temp)
    uni = (1.0 / n) if target == "win" else np.minimum(3, n) / n
    o = X["odds"].where(X["odds"] > 0).fillna(999.0).to_numpy()      # 0 以下・欠けは極端な人気薄として扱う
    inv = 1.0 / o
    mk = inv / pd.Series(inv).groupby(rid).transform("sum").to_numpy()
    if target != "win":     # 市場の複勝の物差し: log(勝率)・log(頭数) のロジスティックを学習で当て、和を min(3,n) に
        from sklearn.linear_model import LogisticRegression
        Z = np.c_[np.log(mk), np.log(n)]
        lr = LogisticRegression(C=100.0, max_iter=500).fit(Z[tr], y[tr])
        mk = race_norm(lr.predict_proba(Z)[:, 1], rid, total, 1.0)
    res = {"target": target, "split": how, "n_feat": len(cols), "best_iter": m.best_iteration, "temp": round(temp, 2)}
    for nm, msk in (("va", va), ("te", te)):
        if target == "win":
            f = lambda pp: race_ll(y[msk], pp[msk], rid[msk])
        else:
            f = lambda pp: ll(y[msk], pp[msk])
        res[f"{nm}_model"], res[f"{nm}_uniform"], res[f"{nm}_market"] = f(q), f(uni if np.ndim(uni) else uni), f(mk)
        res[f"{nm}_gain_vs_uni"] = res[f"{nm}_uniform"] - res[f"{nm}_model"]
        res[f"{nm}_rows"] = int(msk.sum())
    if verbose:
        print(json.dumps(res, ensure_ascii=False))
    imp = pd.Series(m.feature_importance("gain"), index=cols).sort_values(ascending=False)
    return {"res": res, "pred": q, "raw": p, "split": sp, "imp": imp, "model": m, "X": X, "cols": cols}


def _harville_top3(w: np.ndarray, rid: np.ndarray) -> np.ndarray:
    """勝率から 3 着内の確率（Harville）。市場の物差しの複勝版。"""
    out = np.zeros_like(w)
    s = pd.Series(np.arange(len(w))).groupby(rid)
    for _, idx in s:
        ii = idx.to_numpy()
        p = w[ii]
        k = len(p)
        if k <= 3:
            out[ii] = 1.0
            continue
        P1 = p
        P2 = np.zeros(k)
        P3 = np.zeros(k)
        for a in range(k):
            ra = 1 - p[a]
            for b in range(k):
                if b == a:
                    continue
                pab = p[a] * p[b] / ra
                P2[b] += pab
                rb = ra - p[b]
                rem = p / rb
                rem[[a, b]] = 0
                P3 += pab * rem
        out[ii] = P1 + P2 + P3
    return np.clip(out, 0, 1)
