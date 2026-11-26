"""段 3 の部品: 払戻（単勝・複勝。100 円あたり）を付け、選んだ馬の回収率・区間・同じ人気の基準線を出す。

- 払戻は保存された組ごとの値を使う（src/pipeline/betting/winning_combos.py。同着・7 頭以下の複勝 2 着までに対応済み）。
- 区間は開催の単位のブートストラップ。基準線は「同じ人気の順 × 同じ年」の全馬の回収率を、選んだ馬の構成で重み付けした値。
- 買い方の帰無（2 号）: 年 × 人気の帯の中で点を並べ替え、同じ規則で選んだときの回収率の分布。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.pipeline.betting.winning_combos import hit_payout, normalize_payoff


def attach_pay(X: pd.DataFrame) -> pd.DataFrame:
    """レースごとに払戻を 1 回だけ読み、各馬の単勝・複勝の 100 円あたり払戻を付ける。"""
    pays = {}
    for rid, s in X.drop_duplicates("race_id")[["race_id", "payoff"]].itertuples(index=False):
        try:
            pays[rid] = normalize_payoff(json.loads(s)) if isinstance(s, str) and s else {}
        except Exception:
            pays[rid] = {}
    w, p = np.zeros(len(X)), np.zeros(len(X))
    for i, (rid, hn) in enumerate(zip(X["race_id"].to_numpy(), X["horse_number"].to_numpy())):
        d = pays.get(rid, {})
        w[i] = hit_payout("tansho", (int(hn),), d.get("tansho", {}))
        p[i] = hit_payout("fukusho", (int(hn),), d.get("fukusho", {}))
    X = X.copy()
    X["pay_win"], X["pay_place"] = w, p
    X["has_place"] = X["race_id"].map({r: bool(d.get("fukusho")) for r, d in pays.items()}).fillna(False).to_numpy()
    return X


def roi(X: pd.DataFrame, sel: np.ndarray, kind: str = "win", n_boot: int = 500, seed: int = 0) -> dict:
    col = "pay_win" if kind == "win" else "pay_place"
    S = X[sel]
    if kind == "place":
        S = S[S["has_place"]]
    n = len(S)
    if n == 0:
        return {"n": 0}
    r = S[col].sum() / (100 * n)
    mt = pd.factorize(S["meeting"])[0]
    nm = mt.max() + 1
    pay_m = np.bincount(mt, S[col].to_numpy()) / 100
    cnt_m = np.bincount(mt)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        c = np.bincount(rng.integers(0, nm, nm), minlength=nm)
        bs.append((c * pay_m).sum() / max((c * cnt_m).sum(), 1))
    # 同じ人気の順 × 年 × 頭数の帯の基準線（全馬の回収率を、選んだ馬の構成で重み付け）。複勝は発売のあるレースだけ（2 号）
    U = X if kind == "win" else X[X["has_place"]]
    def keyf(D):
        fb = pd.cut(D["n_run"], [0, 10, 14, 99], labels=["s", "m", "l"]).astype(str)
        return D["year"].astype(str) + "_" + D["popularity"].clip(upper=18).astype(str) + "_" + fb
    allr = U.groupby(keyf(U))[col].sum() / (100 * U.groupby(keyf(U)).size())
    base = float(allr.reindex(keyf(S)).mean())
    return {"n": int(n), "hits": int((S[col] > 0).sum()), "roi": round(float(r), 4),
            "ci": [round(float(v), 4) for v in np.percentile(bs, [2.5, 97.5])], "same_pop": round(base, 4),
            "vs_same_pop": round(float(r) - base, 4), "mean_odds": round(float(S["odds"].mean()), 2)}


def perm_null(X: pd.DataFrame, score: str, rule, kind: str = "win", n_perm: int = 200, seed: int = 0) -> np.ndarray:
    """買い方の帰無（2 号の段 1 の R1）。点を年ごとに log q と頭数へ回帰し、残差だけを「年 × オッズの 20 分位」の中で並べ替えて
    点を作り直す（= 市場と直交する部分に情報が無い、という帰無）。X は場面の行だけを渡す。rule(Y) は選ぶ行の真偽の配列を返す。"""
    col = "pay_win" if kind == "win" else "pay_place"
    v = X[score].to_numpy(float)
    lq = np.log(X["q"].to_numpy(float))
    ln = np.log(X["n_run"].to_numpy(float))
    yr = X["year"].to_numpy()
    fit = np.empty_like(v)
    for t in np.unique(yr):
        m = yr == t
        A = np.c_[np.ones(m.sum()), lq[m], ln[m]]
        coef, *_ = np.linalg.lstsq(A, v[m], rcond=None)
        fit[m] = A @ coef
    res = v - fit
    ob = pd.qcut(pd.Series(lq).rank(method="first"), 20, labels=False).to_numpy()
    g = pd.Series(yr).astype(str).to_numpy() + "_" + ob.astype(str)
    idx = pd.Series(np.arange(len(X))).groupby(g).indices
    rng = np.random.default_rng(seed)
    out = []
    has = X["has_place"].to_numpy() if kind == "place" else np.ones(len(X), bool)
    pay = X[col].to_numpy()
    for _ in range(n_perm):
        r2 = res.copy()
        for ii in idx.values():
            r2[ii] = res[rng.permutation(ii)]
        Y = X.assign(**{score: fit + r2})
        sel = np.asarray(rule(Y)) & has
        out.append(pay[sel].sum() / max(100 * sel.sum(), 1))
    return np.array(out)
