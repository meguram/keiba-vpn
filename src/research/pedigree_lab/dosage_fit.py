"""本家の Dosage（chef-de-race 2019-10 の一覧）の列を (A) に足した時間の順の比べ（log.md「延長の続き」の調べ C。開発の行・記述）。

    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=6 nice -n 19 python -m src.research.pedigree_lab.dosage_fit [--null 1] [--check]

- 列（dosage_chef.parquet・馬ごとに一定）: dc_lDI = log(DI + 0.25)・dc_CD・dc_CDd = CD × (距離 − 1600) / 400・dc_total、
  母の側だけの dc_lDI_m・dc_CD_m・dc_CDd_m（1 号）。
- 帰無（--null）: Dosage の値を、生年 × 父の中で馬の単位で入れ替える（父の分を保ち、母の側の情報を試す。2 号）。
- 判定は 2020〜24 年の行だけ（2019 年の行は学習に使い、予測は残さない。一覧の先読みを避ける）。
- --check: 当てずに、列の SD と父の距離の列（sire_dz）との相関だけを出す（流す前の確かめ・1 号）。
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.research.pedigree_lab import ablate, data

NEW = ["dc_lDI", "dc_CD", "dc_CDd", "dc_total", "dc_lDI_m", "dc_CD_m", "dc_CDd_m", "dc_total_m"]


def cols(X: pd.DataFrame, D: pd.DataFrame) -> pd.DataFrame:
    M = X[["horse_id", "distance"]].merge(D, on="horse_id", how="left")
    dz = (pd.to_numeric(M["distance"], errors="coerce") - 1600) / 400
    out = pd.DataFrame(index=X.index)
    out["dc_lDI"] = np.log(M["DI"] + 0.25).to_numpy()
    out["dc_CD"] = M["CD"].to_numpy()
    out["dc_CDd"] = (M["CD"] * dz).to_numpy()
    out["dc_total"] = M["dp_total"].to_numpy()
    out["dc_lDI_m"] = np.log(M["DI_m"] + 0.25).to_numpy()
    out["dc_CD_m"] = M["CD_m"].to_numpy()
    out["dc_CDd_m"] = (M["CD_m"] * dz).to_numpy()
    out["dc_total_m"] = M["dm_total"].to_numpy()
    return out


def main() -> None:
    null_seed = int(sys.argv[sys.argv.index("--null") + 1]) if "--null" in sys.argv else None
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "dosage_fit" + ("_null" if null_seed is not None else ""),
                                   "本家の Dosage の列の時間の順の比べ（尤度だけ・記述）"))
    X = pd.read_parquet(data.OUT / "feat_v4.parquet")
    X = X[(X["date"].dt.year >= 2017) & (X["date"] <= pd.Timestamp(data.DEV_END))].reset_index(drop=True)
    D = pd.read_parquet(data.OUT / "dosage_chef.parquet")
    if null_seed is not None:
        rng = np.random.default_rng(null_seed)
        H = X.drop_duplicates("horse_id")[["horse_id", "sire"]].merge(D, on="horse_id", how="inner").reset_index(drop=True)
        g = (H["horse_id"].str[:4] + "|" + H["sire"].fillna("")).to_numpy()
        gi = pd.factorize(g)[0]
        a = np.lexsort((np.arange(len(gi)), gi)); b = np.lexsort((rng.random(len(gi)), gi))
        vals = H.drop(columns=["horse_id", "sire"]).to_numpy()
        newv = vals.copy(); newv[a] = vals[b]
        D = pd.DataFrame(newv, columns=H.columns.drop(["horse_id", "sire"])).assign(horse_id=H["horse_id"].to_numpy())
    C = cols(X, D)
    if "--check" in sys.argv:
        print("SD", C.std().round(3).to_dict(), flush=True)
        print("付く割合", C.notna().mean().round(3).to_dict(), flush=True)
        for c in ("sire_dz", "bms_dz"):
            if c in X.columns:
                print(c, "との相関", {k: round(float(pd.Series(C[k]).corr(X[c])), 3) for k in NEW}, flush=True)
        data.finish_all("dosage_fit_check")
        return
    for c in NEW:
        X[c] = C[c].to_numpy()
    from src.research.pedigree_lab.walkforward import fit_year
    use = [c for c in ablate.col_sets(X)["pure"] if not c.startswith("ib_")] + NEW
    yr = X["date"].dt.year.to_numpy()
    p = np.full(len(X), np.nan)
    for t in range(2020, 2025):
        p[yr == t] = fit_year(X, use, "win", t, 2000)
        print(t, flush=True)
    te = yr >= 2020
    o = X.loc[te, ["race_id", "horse_id"]].copy()
    o["p_TB"] = p[te]
    path = data.OUT / f"dosage_wf{'' if null_seed is None else f'_null{null_seed}'}.parquet"
    o.to_parquet(path, index=False)
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
