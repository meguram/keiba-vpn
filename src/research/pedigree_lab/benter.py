"""段 2: Benter の 2 段（血統の点 × 市場の確率）。血統が市場に足す情報があるかを測る。

p ∝ exp(α·log p_血統 + β·log q_市場)。q は確定オッズの 1/odds を和 1 にしたもの（ここは診断。買い方の判定には T-60 の再現を使う）。
- 当てる側 2019〜2022・測る側 2023〜2024（walk-forward の OOF なので、血統の点はどの年も未来を見ていない）。
- α の区間は開催の単位のブートストラップ。測る側の対数損失（市場だけ → 2 段）の改善も出す。
- 場面別（事前に数えた 6 場面と全体）の α も出す。
    nice -n 19 python -m src.research.pedigree_lab.benter --oof oof_v2_main
出力: OUT/benter_<oof>.json。費用 0 円。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data
from src.research.pedigree_lab.hyp_tests import clogit

SCENES = {   # 事前に数えた場面（log.md。1 号の R1 の 6 つ）
    "全体": lambda X: np.ones(len(X), bool),
    "新馬": lambda X: (X["debut"] == "Y").to_numpy(),
    "初ダート": lambda X: ((X["first_surf"] == "Y") & (X["surf"] == "D")).to_numpy(),
    "初芝": lambda X: ((X["first_surf"] == "Y") & (X["surf"] == "T")).to_numpy(),
    "距離の延長": lambda X: (X["dchg"] == "U").to_numpy(),
    "距離の短縮": lambda X: (X["dchg"] == "D").to_numpy(),
    "道悪": lambda X: (X["going"] == "W").to_numpy(),
}


def load(oof: str) -> pd.DataFrame:
    O = pd.read_parquet(data.OUT / f"{oof}.parquet")
    F = pd.read_parquet(data.OUT / "feat_v2.parquet",
                        columns=["race_id", "horse_id", "meeting", "date", "win", "top3", "odds", "popularity", "n_run",
                                 "debut", "first_surf", "surf", "dchg", "going", "cls", "payoff", "horse_number"])
    X = O.merge(F, on=["race_id", "horse_id"], how="left")
    X = X[X["odds"] > 0].copy()
    inv = 1.0 / X["odds"].to_numpy()
    X["q"] = inv / pd.Series(inv).groupby(X["race_id"].to_numpy()).transform("sum").to_numpy()
    X["lp"] = np.log(np.clip(X["p_win"], 1e-5, 1))
    X["lq"] = np.log(np.clip(X["q"], 1e-5, 1))
    return X.sort_values(["race_id", "horse_number"]).reset_index(drop=True)


def fit(X: pd.DataFrame) -> np.ndarray:
    rc = pd.factorize(X["race_id"])[0]
    return clogit(X[["lp", "lq"]].to_numpy(), X["win"].to_numpy(float), rc)


def combined(X: pd.DataFrame, b: np.ndarray) -> np.ndarray:
    eta = X[["lp", "lq"]].to_numpy() @ b
    rid = X["race_id"].to_numpy()
    e = np.exp(eta - pd.Series(eta).groupby(rid).transform("max").to_numpy())
    return e / pd.Series(e).groupby(rid).transform("sum").to_numpy()


def win_ll(X: pd.DataFrame, p: np.ndarray) -> float:
    m = X["win"].to_numpy() == 1
    return float(-np.mean(np.log(np.clip(p[m], 1e-9, 1))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="oof_v2_main")
    ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args()
    X = load(a.oof)
    yr = X["year"].to_numpy()
    A, B = X[yr <= 2022].reset_index(drop=True), X[yr >= 2023].reset_index(drop=True)
    res = {"oof": a.oof}
    b = fit(A)
    res["coef_fit"] = b.round(4).tolist()
    res["test_ll_market"] = win_ll(B, B["q"].to_numpy())
    res["test_ll_combined"] = win_ll(B, combined(B, b))
    res["test_ll_ped"] = win_ll(B, B["p_win"].to_numpy())
    res["test_gain"] = res["test_ll_market"] - res["test_ll_combined"]
    # 測る側での α（当て直し）とブートストラップ
    mt = pd.factorize(B["meeting"])[0]
    nm = mt.max() + 1
    rc = pd.factorize(B["race_id"])[0]
    M, y = B[["lp", "lq"]].to_numpy(), B["win"].to_numpy(float)
    bt = clogit(M, y, rc)
    rng = np.random.default_rng(0)
    al = [clogit(M, y, rc, n_iter=8, w=np.bincount(rng.integers(0, nm, nm), minlength=nm)[mt].astype(float))[0] for _ in range(a.boot)]
    res["test_alpha"] = [round(float(bt[0]), 4), *np.percentile(al, [2.5, 97.5]).round(4).tolist()]
    res["scenes"] = {}
    for nm_s, f in SCENES.items():
        out = {}
        for side, D in (("fit", A), ("test", B)):
            msk = f(D)
            S = D[msk].reset_index(drop=True)
            if S["race_id"].nunique() < 200:
                continue
            # 場面の馬だけを見るため、レースの全頭で当てて、場面の馬に α の上乗せ（交互作用）を置く
            Z = np.c_[D["lp"], D["lq"], D["lp"] * msk]
            bb = clogit(Z, D["win"].to_numpy(float), pd.factorize(D["race_id"])[0])
            out[side] = {"alpha": round(float(bb[0]), 4), "alpha_extra": round(float(bb[2]), 4), "rows": int(msk.sum())}
        res["scenes"][nm_s] = out
    print(json.dumps(res, ensure_ascii=False, indent=1))
    (data.OUT / f"benter_{a.oof}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
