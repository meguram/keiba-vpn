"""系統の単位のニックス（父の父 × 母父の父）の階層の target encoding（延長の調べ 4b。開発の行・時間の順・記述）。

    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=6 nice -n 19 python -m src.research.pedigree_lab.linenick [--null 1]

- 組の効き = 系統の組 × 芝ダの −分位・3 着内の as-of の平均を、「父の父の値 + 母父の父の値 − 全体」の足し算の見込みへ m = 200 で縮め、見込みを引いた値。
  自分の行・同じ日を除く。個体のニックス（nick_*）は第 4 版にすでにある。母の質は第 4 版の dam_* が制御として入る。
- 帰無（--null）: 母父の父の札を、母の単位で（同じ母の子は同じ札）、同じ生年 × 父の父の中で入れ替える（1 号・2 号）。
- 時間の順（walkforward.fit_year）で (A) ＋ この列を当て、行ごとの予測を parquet に残す（compare_null で比べる）。
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.research.pedigree_lab import ablate, data
from src.research.pedigree_lab.data import asof
from src.research.pedigree_lab.walkforward import fit_year

M = 200.0


def bms_sire() -> pd.DataFrame:
    ped = data.load_ped()[["horse_id", "dam", "anc5"]]
    ped["ms"] = ped["anc5"].fillna("").str.extract(r"(?:^|;)3:4:([^;]+)")[0]
    return ped[["horse_id", "dam", "ms"]]


def main() -> None:
    null_seed = int(sys.argv[sys.argv.index("--null") + 1]) if "--null" in sys.argv else None
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "linenick" + ("_null" if null_seed is not None else ""),
                                   "系統の単位のニックスの TE の時間の順の比べ（尤度だけ・記述）"))
    X = pd.read_parquet(data.OUT / "feat_v4.parquet")
    X = X[X["date"] <= pd.Timestamp(data.DEV_END)].reset_index(drop=True)
    B = bms_sire()
    X = X.merge(B[["horse_id", "ms"]], on="horse_id", how="left")
    if null_seed is not None:
        rng = np.random.default_rng(null_seed)
        H = X.drop_duplicates("horse_id")[["horse_id", "ss"]].merge(B, on="horse_id", how="left")
        H = H[H["dam"].notna()]                      # 母の欠ける馬は入れ替えから外す（critic）
        H["by"] = H["horse_id"].str[:4]
        # 層は母の単位で作る: 母の一番多い父の父 × 母の最初の子の生年（母の単位と、子ごとの層は両立しないため。critic）
        U = H.groupby("dam").agg(ss=("ss", lambda x: x.mode().iat[0] if len(x.mode()) else ""), by=("by", "min"), ms=("ms", "first")).reset_index()
        for ii in U.groupby(["by", "ss"]).indices.values():
            U.iloc[ii, U.columns.get_loc("ms")] = U["ms"].to_numpy()[ii][rng.permutation(len(ii))]
        m = H["dam"].map(U.set_index("dam")["ms"])
        X["ms"] = X["horse_id"].map(dict(zip(H["horse_id"], m)))
    E = X.assign(_g=-X["fpct"].to_numpy(), _t3=X["top3"].astype(float), _k="1")
    cols = []
    for j, v in enumerate(("fpct", "top3")):
        g0, c0 = asof(E, ["_k", "surf"], ["_g", "_t3"])
        base = g0[:, j] / np.maximum(c0, 1)
        s1, c1 = asof(E, ["ss", "surf"], ["_g", "_t3"])
        a1 = (s1[:, j] + M * base) / (c1 + M)
        s2, c2 = asof(E, ["ms", "surf"], ["_g", "_t3"])
        a2 = (s2[:, j] + M * base) / (c2 + M)
        prior = a1 + a2 - base
        s3, c3 = asof(E, ["ss", "ms", "surf"], ["_g", "_t3"])
        X[f"ln_nick_{v}"] = (s3[:, j] + M * prior) / (c3 + M) - prior
        cols.append(f"ln_nick_{v}")
    X = X[X["date"].dt.year >= 2017].reset_index(drop=True)
    use = [c for c in ablate.col_sets(X)["pure"] if not c.startswith("ib_")] + cols
    yr = X["date"].dt.year.to_numpy()
    p = np.full(len(X), np.nan)
    for t in range(2019, 2025):
        p[yr == t] = fit_year(X, use, "win", t, 2000)
        print(t, flush=True)
    te = yr >= 2019
    out = X.loc[te, ["race_id", "horse_id"]].copy()
    out["p_TB"] = p[te]
    tag = "" if null_seed is None else f"_null{null_seed}"
    path = data.OUT / f"linenick_wf{tag}.parquet"
    out.to_parquet(path, index=False)
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
