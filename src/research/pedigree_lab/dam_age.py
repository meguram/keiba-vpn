"""母の年齢・種付けの時の父の年齢（log.md「母の年齢・何番目の仔・種付けの時の父の年齢」。開発の行・時間の順・記述）。

    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=6 nice -n 19 python -m src.research.pedigree_lab.dam_age [--null 1] [--only bx_dam_age]

- 何番目の仔は、血統表の打ち切りが生年で変わるので列に入れない（作るだけ）。

- 列は血統表の id と生年だけから作る（生まれの事実で、レースの結果を使わない）。外国の id（数字でない）は欠け。
- (A) ＋ 2 列を時間の順（walkforward.fit_year）で当て、行ごとの予測を parquet に残す（compare_null で比べる）。
- 帰無（--null）: 2 列を、同じ生年 × 欠けの形の馬の中で、組のまま入れ替える。
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.research.pedigree_lab import ablate, data
from src.research.pedigree_lab.walkforward import fit_year


def birth_year(ids: pd.Series) -> pd.Series:
    s = ids.astype(str)
    y = pd.to_numeric(s.str[:4], errors="coerce")
    return y.where(s.str.fullmatch(r"\d{10}") & (y >= 1960) & (y <= 2025))


def cols_table() -> pd.DataFrame:
    ped = data.load_ped()[["horse_id", "sire", "dam"]]
    by = birth_year(ped["horse_id"])
    ped["bx_dam_age"] = by - birth_year(ped["dam"])
    ped["bx_sire_age"] = by - 1 - birth_year(ped["sire"])
    ped["_by"] = by
    ped = ped.sort_values(["dam", "_by", "horse_id"])
    ped["foal_order"] = ped.groupby("dam").cumcount() + 1
    ped.loc[ped["dam"].fillna("").eq(""), "foal_order"] = np.nan
    return ped[["horse_id", "bx_dam_age", "foal_order", "bx_sire_age", "_by"]]


def main() -> None:
    null_seed = int(sys.argv[sys.argv.index("--null") + 1]) if "--null" in sys.argv else None
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "dam_age" + ("_null" if null_seed is not None else ""),
                                   "母の年齢・何番目の仔・父の年齢の時間の順の比べ（尤度だけ・記述）"))
    X = pd.read_parquet(data.OUT / "feat_v4.parquet")
    X = X[(X["date"].dt.year >= 2017) & (X["date"] <= pd.Timestamp(data.DEV_END))].reset_index(drop=True)
    C = cols_table()
    if null_seed is not None:
        rng = np.random.default_rng(null_seed)
        C = C.reset_index(drop=True)
        perm = np.arange(len(C))
        # 欠けの形（外国の id の印）が同じ馬の中だけで入れ替える（critic）
        pat = C["bx_dam_age"].isna().astype(int).astype(str) + C["bx_sire_age"].isna().astype(int).astype(str)
        for ii in C.groupby([C["_by"].fillna(-1), pat]).indices.values():
            perm[ii] = rng.permutation(ii)
        for c in ("bx_dam_age", "bx_sire_age"):
            C[c] = C[c].to_numpy()[perm]
    X = X.merge(C.drop(columns=["_by", "foal_order"]), on="horse_id", how="left")
    # 何番目の仔は、血統表が 2016 年以降に走った馬だけなので、生年によって打ち切りの量が変わる（critic）。列に入れない
    new = ["bx_dam_age", "bx_sire_age"]
    if "--only" in sys.argv:                       # 切り分け: 1 列だけ足す版
        new = [sys.argv[sys.argv.index("--only") + 1]]
        assert new[0] in ("bx_dam_age", "bx_sire_age"), new
    use = [c for c in ablate.col_sets(X)["pure"] if not c.startswith("ib_")] + new
    yr = X["date"].dt.year.to_numpy()
    p = np.full(len(X), np.nan)
    for t in range(2019, 2025):
        p[yr == t] = fit_year(X, use, "win", t, 2000)
        print(t, flush=True)
    te = yr >= 2019
    out = X.loc[te, ["race_id", "horse_id"]].copy()
    out["p_TB"] = p[te]
    only = ("_" + sys.argv[sys.argv.index("--only") + 1]) if "--only" in sys.argv else ""
    path = data.OUT / f"dam_age_wf{only}{'' if null_seed is None else f'_null{null_seed}'}.parquet"
    out.to_parquet(path, index=False)
    print("coverage", {c: round(float(X[c].notna().mean()), 3) for c in new}, flush=True)
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
