"""危ない人気馬の深掘り（log.md「調べ N2」。記述・買い方の設計ではない）。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.fav_doubt2

- 調べ N と同じ行・印・回帰。(a) 年ごと・2 期、(b) T-60 の再現の人気、(c) 芝ダ・クラス・頭数の群、(d) logit(p_A) の 2 次を足した版。
- T-60 の再現は pre_odds.simulate（2026 年の T-60 と確定の比を借りる。読むのはオッズだけで着順は読まない）。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data, pre_odds
from src.research.pedigree_lab.sire_market import irls


def prep(X: pd.DataFrame, qcol: str, popcol: str) -> pd.DataFrame:
    D = X[X[popcol] <= 3].copy()
    D["z"] = np.log(D["pA"].clip(1e-6)) - np.log(D[qcol])
    D["pop_"] = D[popcol].astype(int)
    D["low"] = (D["z"] <= D.groupby(["y", "pop_"])["z"].transform(lambda s: s.quantile(0.2))).astype(float)
    qq = D[qcol].clip(1e-4, 1 - 1e-4)
    D["lq"] = np.log(qq / (1 - qq))
    pp = D["pA"].clip(1e-4, 1 - 1e-4)
    D["lp"] = np.log(pp / (1 - pp))
    return D.reset_index(drop=True)


def fit(D: pd.DataFrame, rng, with_p: str = "", nb: int = 1000) -> dict:
    lq = D["lq"].to_numpy()
    cols = [np.ones(len(D)), lq, lq ** 2, D["pop_"].eq(2).to_numpy(float), D["pop_"].eq(3).to_numpy(float)]
    extra = []
    if with_p == "lp2":
        lp = D["lp"].to_numpy(); cols += [lp, lp ** 2]; extra = ["lp", "lp2"]
    elif with_p == "lp":
        lp = D["lp"].to_numpy(); cols += [lp]; extra = ["lp"]              # 1 次だけ（情報が足されているか。2 号）
    elif with_p == "z":
        cols += [D["z"].to_numpy()]; extra = ["z"]                         # z の 1 次を入れた上での印（裾に集まっているか。2 号）
    cols.append(D["low"].to_numpy())
    M = np.column_stack(cols)
    y = D["win"].to_numpy(float)
    b = irls(M, y)
    day = D["date"].dt.strftime("%Y%m%d").to_numpy()
    du, di = np.unique(day, return_inverse=True)
    idx = [np.flatnonzero(di == k) for k in range(len(du))]
    bs, be = [], []
    for _ in range(nb):
        ii = np.concatenate([idx[k] for k in rng.integers(0, len(du), len(du))])
        bb = irls(M[ii], y[ii])
        bs.append(bb[-1]); be.append(bb[5:5 + len(extra)])
    low = D["low"].to_numpy(bool)
    q = D["q_use"].to_numpy()
    return {"走り": int(len(D)), "下位 2 割の勝ち": int(y[low].sum()), "A/E 下位 2 割": round(float(y[low].sum() / q[low].sum()), 3),
            "A/E それ以外": round(float(y[~low].sum() / q[~low].sum()), 3),
            "係数": round(float(b[-1]), 3), "95%": np.percentile(bs, [2.5, 97.5]).round(3).tolist(),
            "足した項": {nm: [round(float(b[5 + i]), 3), np.percentile(np.array(be)[:, i], [2.5, 97.5]).round(3).tolist()] for i, nm in enumerate(extra)}}


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "fav_doubt2", "危ない人気馬の深掘り（記述）"))
    X = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "date", "split", "win", "odds", "popularity",
                                                              "surf", "cls", "field_size"])
    X = X[X["date"] <= data.DEV_END].copy()
    P = pd.read_parquet(data.OUT / "rulebased_wf.parquet")[["race_id", "horse_id", "p_A"]]
    X = X.merge(P, on=["race_id", "horse_id"])
    X["odds"] = pd.to_numeric(X["odds"], errors="coerce")
    X = X[X["odds"] > 0].sort_values(["race_id", "horse_id"]).reset_index(drop=True)
    X = X[X.groupby("race_id")["race_id"].transform("size") >= 2].reset_index(drop=True)
    rid = X["race_id"].to_numpy()
    inv = 1 / X["odds"].to_numpy(float)
    X["q"] = inv / pd.Series(inv).groupby(rid).transform("sum").to_numpy()
    X["pA"] = X["p_A"] / X.groupby("race_id")["p_A"].transform("sum")
    X["pop_fin"] = pd.to_numeric(X["popularity"], errors="coerce")
    # T-60 の再現（5 回）。人気は 1 回目の再現オッズの順、q は 5 回の平均
    rk = X.groupby("race_id")["odds"].rank(method="first").to_numpy()
    sims = pre_odds.simulate(rid, X["odds"].to_numpy(float), rk, pre_odds.load_drift("t60"), n=5)
    qs = [(1 / o) / pd.Series(1 / o).groupby(rid).transform("sum").to_numpy() for o in sims]
    X["q60"] = np.mean(qs, axis=0)
    X["pop60"] = pd.Series(sims[0]).groupby(rid).rank(method="first").to_numpy()
    X = X[X["split"].isin([0, 1])].reset_index(drop=True)
    X["y"] = X["date"].dt.year
    rng = np.random.default_rng(3)
    res = {}
    Df = prep(X, "q", "pop_fin"); Df["q_use"] = Df["q"]
    # (a) 年ごと・2 期
    res["(a) 年ごと"] = {int(y): fit(Df[Df["y"] == y].reset_index(drop=True), rng, nb=300)["係数"] for y in range(2019, 2025)}
    res["(a) 2019〜21 年"] = fit(Df[Df["y"].between(2019, 2021)].reset_index(drop=True), rng)
    res["(a) 2022〜24 年"] = fit(Df[Df["y"].between(2022, 2024)].reset_index(drop=True), rng)
    # (b) T-60 の再現の人気
    Dt = prep(X.assign(q=X["q60"]), "q60", "pop60"); Dt["q_use"] = Dt["q60"]
    res["(b) T-60 の再現"] = fit(Dt, rng)
    # (c) 分けた記述
    fs = pd.to_numeric(Df["field_size"], errors="coerce")
    groups = {"芝": Df["surf"].eq("T"), "ダート": Df["surf"].eq("D"), "12 頭以下": fs <= 12, "13〜15 頭": fs.between(13, 15), "16 頭以上": fs >= 16}
    for c in sorted(Df["cls"].astype(str).unique()):
        groups[f"クラス {c}"] = Df["cls"].astype(str).eq(c)
    res["(c) 分けた記述"] = {k: fit(Df[m.to_numpy()].reset_index(drop=True), rng, nb=300) for k, m in groups.items() if m.sum() > 500}
    # (d) 血統のモデルの較正の曲がりを除く
    res["(d) logit(p_A) の 2 次を足す"] = fit(Df, rng, with_p="lp2")
    res["(d2) logit(p_A) の 1 次を足す"] = fit(Df, rng, with_p="lp")
    res["(d3) z の 1 次を足す"] = fit(Df, rng, with_p="z")
    print(json.dumps(res, ensure_ascii=False), flush=True)
    path = data.OUT / "fav_doubt2.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
