"""探す側の記述の 3 巡目（log.md「3 巡目」。未確認・買い方に使わない）。

    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=4 nice -n 19 python -m src.research.pedigree_lab.explore3 <recal|wintop3>

- recal: 父の産駒の数の帯 × 初仔の 8 帯ごとに、lp の傾きを当て直す（レースの単位の 5 分割の交差）。勝ちの改善の差（開催の束ね）。
- wintop3: 採った仮説 9 本の列の、勝ちと 3 着内の傾き（レースの中で中心化した列のロジスティック）の比。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data
from src.research.pedigree_lab.hyp_tests import clogit

H9 = {"H1 父×芝ダ": "sire_surf_top3", "H2 父×距離帯": "sire_sd_top3", "H3 母父": "bms_all_top3", "H4 きょうだい": "dam_v_win",
      "H6' 父×レースの年齢条件": "sire_rasurf_top3", "H7 牝系": "dd_v_win", "H9 ニックス": "nick_all_top3",
      "H12 母の勝ち上がりの速さ": "dam_v_quick", "H14 父の距離の幅のずれ": "sire_dz"}


def load() -> pd.DataFrame:
    O = pd.read_parquet(data.OUT / "kf_v4_pure_dd.parquet")
    O = O[O["part"] == "dev"][["race_id", "horse_id", "p_win"]]
    F = pd.read_parquet(data.OUT / "feat_v4.parquet",
                        columns=["race_id", "horse_id", "meeting", "win", "top3", "horse_number", "sire_all_n", "dam_v_nrun", *H9.values()])
    X = O.merge(F, on=["race_id", "horse_id"]).sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    X["lp"] = np.log(np.clip(X["p_win"], 1e-5, 1))
    return X


def softmax(eta, rid):
    e = np.exp(eta - pd.Series(eta).groupby(rid).transform("max").to_numpy())
    return e / pd.Series(e).groupby(rid).transform("sum").to_numpy()


def run_recal(X: pd.DataFrame) -> dict:
    sn = np.nan_to_num(np.expm1(X["sire_all_n"].to_numpy()), nan=0.0)      # 欠けは「〜300」の帯（critic 4 回目の 10）
    sb = np.digitize(sn, [300, 1000, 3000])
    first = (np.expm1(X["dam_v_nrun"].to_numpy()) < 0.5).astype(int)
    band = sb * 2 + first
    Z = np.c_[tuple(X["lp"].to_numpy() * (band == b) for b in range(8))]
    y = X["win"].to_numpy(float)
    rid = X["race_id"].to_numpy()
    rc = pd.factorize(rid)[0]
    fo = np.random.default_rng(7).permutation(rc.max() + 1)[rc] % 5
    p_new = np.zeros(len(X))
    p_old = softmax(X["lp"].to_numpy(), rid)        # 傾き 1（今の確率そのもの）
    coefs = []
    for k in range(5):
        A = fo != k
        b = clogit(Z[A], y[A], pd.factorize(rid[A])[0])
        coefs.append(b.round(3).tolist())
        p_new[~A] = softmax(Z[~A] @ b, rid[~A])
    w = y == 1
    d = np.log(p_new[w]) - np.log(p_old[w])
    mt = pd.factorize(X.loc[w, "meeting"])[0]
    nu = mt.max() + 1
    s, c = np.bincount(mt, d, minlength=nu), np.bincount(mt, minlength=nu)
    rng = np.random.default_rng(3)
    bs = []
    for _ in range(1000):
        wb = np.bincount(rng.integers(0, nu, nu), minlength=nu)
        bs.append((wb * s).sum() / (wb * c).sum())
    names = [f"父{['〜300', '300〜1000', '1000〜3000', '3000〜'][i // 2]}・{'初仔' if i % 2 else '初仔でない'}" for i in range(8)]
    return {"gain": round(float(d.mean()), 5), "ci": np.percentile(bs, [2.5, 97.5]).round(5).tolist(),
            "slopes_last_fold": dict(zip(names, coefs[-1])), "share_rows": {n: round(float((band == i).mean()), 3) for i, n in enumerate(names)}}


def run_wintop3(X: pd.DataFrame) -> dict:
    from sklearn.linear_model import LogisticRegression
    rid = X["race_id"].to_numpy()
    logn = np.log(X.groupby("race_id")["race_id"].transform("size").to_numpy())
    out = {}
    for nm, col in H9.items():
        v = X[col].astype(float).fillna(X[col].astype(float).mean()).to_numpy()
        v = v - pd.Series(v).groupby(rid).transform("mean").to_numpy()
        v = v / v.std()
        r = {}
        for tg in ("win", "top3"):
            # レースの中で中心化した列なので、レースの固定効果の代わりに頭数の対数を足した単純なロジスティック
            m = LogisticRegression(C=1e6, max_iter=300).fit(np.c_[v, logn], X[tg].to_numpy())
            r[tg] = round(float(m.coef_[0][0]), 4)
        r["top3_over_win"] = round(r["top3"] / r["win"], 3) if r["win"] != 0 else None
        out[nm] = r
    return out


def run_seedfloor() -> dict:
    """第 4 版 pure を LightGBM の種だけ 3 通りで学び直した 2 代母の単位の OOF の改善の散らばり。"""
    from src.research.pedigree_lab import ablate, model, v5
    X = v5.base_frame("v4")
    dev = X["split"].isin([0, 1]).to_numpy()
    cols = ablate.col_sets(X)["pure"]
    gains = []
    for sd in (11, 23, 37):
        o = v5.oof_for(X, cols, "dd", dict(seed=sd, bagging_seed=sd, feature_fraction_seed=sd), ("win",))
        gains.append(v5.gain(X, o["win"], dev))
        print("seed", sd, gains[-1], flush=True)
    return {"gains": gains, "mean": round(float(np.mean(gains)), 5), "sd": round(float(np.std(gains, ddof=1)), 5)}


def run_harville(X: pd.DataFrame) -> dict:
    """凍結の勝ちの確率から着順を 1 回模擬し（Gumbel の技法・Plackett-Luce）、模擬の勝ち・3 着内で傾きの比を出す。"""
    # 模擬を 10 回（critic 4 回目の 9: 1 回では模擬の散らばりが分からない）
    runs = []
    for k in range(10):
        rng = np.random.default_rng(17 + k)
        g = np.log(np.clip(X["p_win"].to_numpy(), 1e-9, 1)) + rng.gumbel(size=len(X))
        rk = pd.Series(g).groupby(X["race_id"].to_numpy()).rank(ascending=False, method="first").to_numpy()
        runs.append(run_wintop3(X.assign(win=(rk == 1).astype(int), top3=(rk <= 3).astype(int))))
    out = {}
    for nm in runs[0]:
        r = [x[nm]["top3_over_win"] for x in runs]
        out[nm] = {"mean": round(float(np.mean(r)), 3), "min": round(float(np.min(r)), 3), "max": round(float(np.max(r)), 3)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["recal", "wintop3", "seedfloor", "harville"])
    a = ap.parse_args()
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", f"explore3_{a.what}", f"探す側の記述の 3 巡目: {a.what}（尤度だけ）"))
    if a.what == "seedfloor":
        res = run_seedfloor()
    else:
        X = load()
        res = {"recal": run_recal, "wintop3": run_wintop3, "harville": run_harville}[a.what](X)
    path = data.OUT / f"explore3_{a.what}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
