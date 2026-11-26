"""版 − 帰無 と 版 − (A) を、同じ開催日の重みのブートストラップで比べる（2 号の直し。開発の行・父が空でない行・行ごとの二値の改善）。

    KEIBA_SESSION=pedigree-lab python -m src.research.pedigree_lab.compare_null <版の parquet> <帰無の parquet ...> --col p_TB --out <名前>

- 行は rulebased.score と同じ（開発の行・父が空でない行・2019〜24 年）。p_A は rulebased_wf の (A)。
- 群: 全体・学習の期間に父の産駒が 1 走も無い行・それ以外。Bonferroni の本数は --ntests（既定 12。公開の手法の分からは 24）。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.research.pedigree_lab import data

N_TESTS = 12


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("version")
    ap.add_argument("nulls", nargs="+")
    ap.add_argument("--col", default="p_TB")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ntests", type=int, default=N_TESTS)    # 試した数（2 号: 公開の手法の分から 24）
    ap.add_argument("--base", default=None)                    # 比べ先の (A) の parquet（列は --col と同じ名前）
    a = ap.parse_args()
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", f"compare_null_{a.out}", "版と帰無の比べ（尤度だけ・記述）"))
    X = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "date", "sire", "win", "n_run", "split"])
    X = X[(X["date"].dt.year >= 2017) & (X["date"].dt.year <= 2024)]
    fr = X.groupby("sire")["date"].min()
    if a.base:
        # 比べ先の (A) を、同じ学習の窓で当てた予測にする（critic: 2018 年からの版と 2017 年からの (A) を比べない）
        T = pd.read_parquet(data.OUT / a.base)[["race_id", "horse_id", a.col]].rename(columns={a.col: "p_A"})
    else:
        T = pd.read_parquet(data.OUT / "rulebased_wf.parquet")[["race_id", "horse_id", "p_A"]]
    V = pd.read_parquet(data.OUT / a.version)[["race_id", "horse_id", a.col]].rename(columns={a.col: "p_V"})
    T = T.merge(V, on=["race_id", "horse_id"])
    for i, f in enumerate(a.nulls):
        T = T.merge(pd.read_parquet(data.OUT / f)[["race_id", "horse_id", a.col]].rename(columns={a.col: f"p_N{i}"}), on=["race_id", "horse_id"])
    T = T.merge(X, on=["race_id", "horse_id"])
    T = T[T["split"].isin([0, 1]) & T["sire"].fillna("").ne("")].reset_index(drop=True)
    unseen = (T["sire"].map(fr) >= pd.to_datetime(T["date"].dt.year.astype(str) + "-01-01")).to_numpy()
    w, u = T["win"].to_numpy(float), 1 / T["n_run"].to_numpy(float)

    def bll(yv, pv):
        pv = np.clip(pv, 1e-6, 1 - 1e-6)
        return -(yv * np.log(pv) + (1 - yv) * np.log(1 - pv))
    gA = bll(w, u) - bll(w, T["p_A"].to_numpy())
    gV = bll(w, u) - bll(w, T["p_V"].to_numpy())
    gN = np.mean([bll(w, u) - bll(w, T[f"p_N{i}"].to_numpy()) for i in range(len(a.nulls))], axis=0)
    day = T["date"].dt.strftime("%Y%m%d").to_numpy()
    du, dinv = np.unique(day, return_inverse=True)
    rng = np.random.default_rng(3)
    Wb = np.stack([np.bincount(rng.integers(0, len(du), len(du)), minlength=len(du)) for _ in range(1000)]).astype(float)
    zb = norm.ppf(1 - 0.025 / a.ntests)
    res = {"version": a.version, "nulls": a.nulls, "z_bonf": round(float(zb), 3), "score": {}}
    for nm, mk in {"全体": np.ones(len(T), bool), "学習の期間に父の産駒が 1 走も無い": unseen, "それ以外": ~unseen}.items():
        n_d = np.bincount(dinv[mk], minlength=len(du)).astype(float)
        r = {"rows": int(mk.sum()), "A": round(float(gA[mk].mean()), 5), "V": round(float(gV[mk].mean()), 5), "N": round(float(gN[mk].mean()), 5)}
        for k, v in (("V_minus_A", gV - gA), ("V_minus_N", gV - gN), ("N_minus_A", gN - gA)):
            d_d = np.bincount(dinv[mk], weights=v[mk], minlength=len(du))
            bs = (Wb @ d_d) / (Wb @ n_d)
            pt, se = float(d_d.sum() / n_d.sum()), float(bs.std())
            r[k] = [round(pt, 5), np.percentile(bs, [2.5, 97.5]).round(5).tolist(), [round(pt - zb * se, 5), round(pt + zb * se, 5)]]
            if k == "V_minus_N" and len(a.nulls) >= 2:
                # 当て直しの揺れを足した区間（2 号）: 帰無の種の間の分散を (1 + 1/種の数) 倍して足す
                sd = float(np.std([(gV - (bll(w, u) - bll(w, T[f"p_N{i}"].to_numpy())))[mk].mean() for i in range(len(a.nulls))], ddof=1))
                se2 = float(np.sqrt(se ** 2 + sd ** 2 * (1 + 1 / len(a.nulls))))
                # 種の数が少ないので、自由度 m − 1 の t で幅を出す（critic）。V の当て直しの揺れも帰無と同じ大きさと置いた式
                from scipy.stats import t as tdist
                tb = float(tdist.ppf(1 - 0.025 / a.ntests, len(a.nulls) - 1))
                r["V_minus_N_refit"] = [round(pt - tb * se2, 5), round(pt + tb * se2, 5), round(se2, 6), round(tb, 2)]
        # 帰無の種の間の揺れ（critic）: 種ごとの V − N_i の点を並べる
        r["V_minus_N_each_seed"] = [round(float((gV - (bll(w, u) - bll(w, T[f"p_N{i}"].to_numpy())))[mk].mean()), 5) for i in range(len(a.nulls))]
        res["score"][nm] = r
        print(nm, r, flush=True)
    path = data.OUT / f"compare_null_{a.out}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
