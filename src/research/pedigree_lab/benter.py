"""段 2: Benter の 2 段（血統の点 × 市場の確率）。血統が市場に足す情報があるかを、開発の行（split 0・1）の中だけで測る。

p ∝ exp(α·log p_血統 + β·log q_市場)。q は確定オッズの 1/odds を和 1 にしたもの（診断。買い方の判定には T-60 の再現を使う）。
- 血統の点は kfold の開発の OOF（その行の折を学習に使っていない）。
- α・β は折ごとに「ほかの 4 折」で当て、自分の折で測る（交差。対数損失の改善は市場だけ〔β だけ当てた温度付き〕と比べる）。
- α の区間は開発の全行での開催の単位のブートストラップ。
- 場面別（事前に数えた 6 場面）は、場面の馬に α の上乗せ（交互作用）を置いて当てる。
保留（split 2）は読まない（段 3 の最後に 1 回だけ）。
    nice -n 19 python -m src.research.pedigree_lab.benter --kf kf_v3_pure_meeting
出力: OUT/benter_<kf>.json。費用 0 円。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data
from src.research.pedigree_lab.hyp_tests import clogit

SCENES = {   # 事前に数えた場面（log.md。1 号の R1 の 6 つ）
    "新馬": lambda X: (X["debut"] == "Y").to_numpy(),
    "初ダート": lambda X: ((X["first_surf"] == "Y") & (X["surf"] == "D")).to_numpy(),
    "初芝": lambda X: ((X["first_surf"] == "Y") & (X["surf"] == "T")).to_numpy(),
    "距離の延長": lambda X: (X["dchg"] == "U").to_numpy(),
    "距離の短縮": lambda X: (X["dchg"] == "D").to_numpy(),
    "道悪": lambda X: (X["going"] == "W").to_numpy(),
}
FCOLS = ["race_id", "horse_id", "meeting", "date", "win", "top3", "odds", "popularity", "n_run", "debut", "first_surf",
         "surf", "dchg", "going", "cls", "horse_number", "split"]


def load(kf: str, tag: str = "v3", part: str = "dev") -> pd.DataFrame:
    O = pd.read_parquet(data.OUT / f"{kf}.parquet")
    O = O[O["part"] == part]
    F = pd.read_parquet(data.OUT / f"feat_{tag}.parquet", columns=FCOLS)
    X = O.merge(F, on=["race_id", "horse_id"], how="left")
    X = X[X["odds"] > 0].copy()
    inv = 1.0 / X["odds"].to_numpy()
    X["q"] = inv / pd.Series(inv).groupby(X["race_id"].to_numpy()).transform("sum").to_numpy()
    X["lp"] = np.log(np.clip(X["p_win"], 1e-5, 1))
    X["lq"] = np.log(np.clip(X["q"], 1e-5, 1))
    return X.sort_values(["race_id", "horse_number"]).reset_index(drop=True)


def softmax_eta(X: pd.DataFrame, eta: np.ndarray) -> np.ndarray:
    rid = X["race_id"].to_numpy()
    e = np.exp(eta - pd.Series(eta).groupby(rid).transform("max").to_numpy())
    return e / pd.Series(e).groupby(rid).transform("sum").to_numpy()


def win_ll(X: pd.DataFrame, p: np.ndarray) -> float:
    m = X["win"].to_numpy() == 1
    return float(-np.mean(np.log(np.clip(p[m], 1e-9, 1))))


def rc(X):
    return pd.factorize(X["race_id"])[0]


HORSE_LEVEL = {"初ダート", "初芝", "距離の延長", "距離の短縮"}   # レースの中で値が変わる場面（水準の項を置ける）


def centered(X: pd.DataFrame, c: str) -> np.ndarray:
    v = X[c].to_numpy()
    return v - pd.Series(v).groupby(X["race_id"].to_numpy()).transform("mean").to_numpy()


def boot_w(mt: np.ndarray, rng) -> np.ndarray:
    nm = mt.max() + 1
    return np.bincount(rng.integers(0, nm, nm), minlength=nm)[mt].astype(float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kf", default="kf_v4_pure_meeting")
    ap.add_argument("--tag", default="v4")
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--scene-boot", type=int, default=50)
    a = ap.parse_args()
    X = load(a.kf, a.tag)
    y = X["win"].to_numpy(float)
    res = {"kf": a.kf, "rows": len(X), "races": int(X["race_id"].nunique())}
    rcode = rc(X)
    mt = pd.factorize(X["meeting"])[0]
    # 交差: レースの単位で 5 つに分け、外で当てて中で測る。組み合わせた確率（交差の当てはめ）を保存する（段 3 で使う）
    rfold = np.random.default_rng(7).permutation(rcode.max() + 1) % 5
    fo = rfold[rcode]
    p_mk, p_cb = np.zeros(len(X)), np.zeros(len(X))
    coefs = []
    for k in range(5):
        A, ib = X[fo != k], fo == k
        B = X[ib]
        bm = clogit(A[["lq"]].to_numpy(), A["win"].to_numpy(float), rc(A))
        bc = clogit(A[["lp", "lq"]].to_numpy(), A["win"].to_numpy(float), rc(A))
        coefs.append(bc.round(4).tolist())
        p_mk[ib] = softmax_eta(B, B[["lq"]].to_numpy() @ bm)
        p_cb[ib] = softmax_eta(B, B[["lp", "lq"]].to_numpy() @ bc)
    res["cv_coefs"] = coefs
    w = y == 1
    d_race = np.log(p_cb[w]) - np.log(p_mk[w])          # 勝ち馬ごと（= レースごと）の対数損失の改善
    mt_w = mt[w]
    res["cv_gain_over_market"] = float(d_race.mean())
    rng = np.random.default_rng(1)
    nm = mt.max() + 1
    sums, cnts = np.bincount(mt_w, d_race, minlength=nm), np.bincount(mt_w, minlength=nm)
    bs = []
    for _ in range(1000):
        c = np.bincount(rng.integers(0, nm, nm), minlength=nm)
        bs.append((c * sums).sum() / max((c * cnts).sum(), 1))
    res["cv_gain_ci"] = np.percentile(bs, [2.5, 97.5]).tolist()
    res["cv_ll_market_temp"] = float(-np.mean(np.log(p_mk[w])))
    res["cv_ll_combined"] = float(-np.mean(np.log(p_cb[w])))
    pd.DataFrame({"race_id": X["race_id"], "horse_id": X["horse_id"], "p_mk": p_mk, "p_cb": p_cb}).to_parquet(
        data.OUT / f"comb_{a.kf}.parquet", index=False)
    M = X[["lp", "lq"]].to_numpy()
    b = clogit(M, y, rcode)
    res["coef_all"] = b.round(4).tolist()
    rng = np.random.default_rng(0)
    al = [clogit(M, y, rcode, n_iter=8, w=boot_w(mt, rng))[0] for _ in range(a.boot)]
    res["alpha_ci"] = np.percentile(al, [2.5, 97.5]).round(4).tolist()
    lpc, lqc = centered(X, "lp"), centered(X, "lq")
    res["scenes"] = {}
    for nm_s, f in SCENES.items():
        msk = f(X).astype(float)
        cols = [X["lp"].to_numpy(), X["lq"].to_numpy(), lpc * msk, lqc * msk]
        if nm_s in HORSE_LEVEL:
            cols.append(msk)
        Z = np.c_[tuple(cols)]
        bb = clogit(Z, y, rcode)
        rng = np.random.default_rng(2)
        bx = [clogit(Z, y, rcode, n_iter=8, w=boot_w(mt, rng))[2] for _ in range(a.scene_boot)]
        res["scenes"][nm_s] = {"alpha_extra": round(float(bb[2]), 4), "alpha_extra_ci": np.percentile(bx, [2.5, 97.5]).round(4).tolist(),
                               "beta_extra": round(float(bb[3]), 4), "level": round(float(bb[4]), 4) if len(bb) > 4 else None,
                               "rows": int(msk.sum())}
        print(nm_s, res["scenes"][nm_s], flush=True)
    print(json.dumps({k: v for k, v in res.items() if k != "scenes"}, ensure_ascii=False))
    (data.OUT / f"benter_{a.kf}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
