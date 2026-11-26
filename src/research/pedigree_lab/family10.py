"""牝系の家族の賞金の価値（Bruce Lowe の牝系番号の代わり。log.md「調べ D」。開発の行・記述）。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.family10 --table      # 家族の札の表だけ作る（着順を使わない）
    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=6 nice -n 19 python -m src.research.pedigree_lab.family10 [--null 1]

- 家族の札: 手元の 10 代の表で、牝系の筋（g 代の 2^g − 1 番）の 10 代母の id（表には 6 代母も残す）。
- 列（as-of・aei_value の b と縮め先 c・10 代母を家族の祖にする 1 本）: fam_v = 家族の馬の b の平均から、自分と同じ 2 代母を持つ馬
  （兄弟・いとこ。母・叔母自身の走りは家族の側に残る）の分を除き、擬似件数 100 走で c へ縮めた値。fam_dirt = その家族の賞金のうち
  ダートの割合。fam_heads = その日より前に走った家族の頭数の log1p。fam_missing = 10 代までたどれない印（1 号・2 号）。
- 帰無（--null）: 母の単位で、(家族の札, 除く 2 代母) の組を母の生年の中で入れ替える（同じ母の子は同じ札・欠けの馬は外す。2 号）。
- (A) ＋ 4 列を時間の順（学習は 2018 年から・試す年は 2020〜24 年）で当て、行ごとの予測を family_wf{tag}.parquet に残す。
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src.research.pedigree_lab import ablate, data
from src.research.pedigree_lab.data import asof
from src.research.pedigree_lab.dosage_chef import PED10

NEW = ["fam_v", "fam_dirt", "fam_heads", "fam_missing"]


def table() -> pd.DataFrame:
    X = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["horse_id", "date"])
    rows = []
    for h in X.loc[X["date"] <= data.DEV_END, "horse_id"].unique():
        f = PED10 / h[:4] / f"{h}.json"
        if not f.exists():
            continue
        tf = {}
        for a in json.loads(f.read_text()).get("ancestors", []):
            g, p = int(a.get("generation", 0)), int(a.get("position", 0))
            if p == 2 ** g - 1 and a.get("horse_id"):
                tf[g] = a["horse_id"]
        if not tf:
            continue
        deep = tf[max(tf)]
        rows.append((h, tf.get(6, deep), tf.get(10, deep), max(tf)))
    T = pd.DataFrame(rows, columns=["horse_id", "f6", "f10", "depth"])
    T.to_parquet(data.OUT / "family10.parquet", index=False)
    return T


def main() -> None:
    if "--table" in sys.argv:
        data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "family10_table", "牝系の家族の札の表（着順を使わない）"))
        T = table()
        print("馬", len(T), "10 代までたどれる割合", round(float((T["depth"] >= 10).mean()), 3),
              "家族の数 f6", T["f6"].nunique(), "f10", T["f10"].nunique(), flush=True)
        print("家族の大きさ（馬）f10", T.groupby("f10").size().describe().round(1).to_dict(), flush=True)
        data.finish_all(str(data.OUT / "family10.parquet"))
        return
    null_seed = int(sys.argv[sys.argv.index("--null") + 1]) if "--null" in sys.argv else None
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "family10" + ("_null" if null_seed is not None else ""),
                                   "牝系の家族の賞金の価値の時間の順の比べ（尤度だけ・記述）"))
    X = pd.read_parquet(data.OUT / "feat_v4.parquet")
    X = X[X["date"] <= pd.Timestamp(data.DEV_END)].reset_index(drop=True)
    C = pd.read_parquet(data.OUT / "aei_cols.parquet", columns=["race_id", "horse_id", "b"])
    X = X.merge(C, on=["race_id", "horse_id"], how="left")
    F = pd.read_parquet(data.OUT / "family10.parquet")
    X = X.merge(F[["horse_id", "f10", "depth"]], on="horse_id", how="left")
    X["fam_missing"] = (X["depth"].fillna(0) < 10).astype(float)
    # 家族の札と、除く 2 代母（ex）の組。10 代までたどれない馬・2 代母の欠ける馬は家族の列を欠けにする
    ok = (X["depth"].fillna(0) >= 10) & X["dd"].fillna("").ne("")
    X["fk"] = np.where(ok, X["f10"].fillna("none"), "none")
    X["ex"] = np.where(ok, X["dd"].fillna("none"), "none")
    if null_seed is not None:
        # 帰無（2 号）: 母の単位で、(家族の札, 除く 2 代母) の組を、母の生年の中で入れ替える（同じ母の子は同じ札）。欠けの馬は外す
        rng = np.random.default_rng(null_seed)
        M = X.loc[ok, ["dam", "fk", "ex"]].drop_duplicates("dam").reset_index(drop=True)
        gi = pd.factorize(M["dam"].str[:4])[0]
        a_ = np.lexsort((np.arange(len(gi)), gi)); b_ = np.lexsort((rng.random(len(gi)), gi))
        v = M[["fk", "ex"]].to_numpy(); nv = v.copy(); nv[a_] = v[b_]
        mp = dict(zip(M["dam"], map(tuple, nv)))
        pair = X.loc[ok, "dam"].map(mp)
        X.loc[ok, "fk"] = [t[0] for t in pair]
        X.loc[ok, "ex"] = [t[1] for t in pair]
        print("札が替わった母の割合", round(float((nv[:, 0] != v[:, 0]).mean()), 3), flush=True)
    use_b = (X["date"] >= "2018-01-01") & X["b"].notna()
    first = (X.sort_values(["date", "race_id"]).groupby("horse_id").cumcount() == 0).reindex(X.index)
    E = X.assign(_b=np.where(use_b, X["b"], 0.0), _one=use_b.astype(float), _bd=np.where(use_b & X["surf"].eq("D"), X["b"], 0.0),
                 _h=first.astype(float), _k="1")                   # 頭数は 2016 年からの初めての走りで数える（時間の傾きを小さく。critic）
    cs, _ = asof(E, ["_k"], ["_b", "_one", "_bd"])
    c = np.where(cs[:, 1] > 100, cs[:, 0] / np.maximum(cs[:, 1], 1), 1.0)
    cd = np.where(cs[:, 0] > 0, cs[:, 2] / np.maximum(cs[:, 0], 1e-9), 0.5)
    # 家族の走り − 2 代母の子孫（自分の母の子・叔母の子孫を含む）の走り。自分の過去の走りは両方で引かれて打ち消す（as-of は日付で厳密に前）
    s, _ = asof(E, ["fk"], ["_b", "_one", "_bd", "_h"])
    sx, _ = asof(E, ["fk", "ex"], ["_b", "_one", "_bd", "_h"])
    d = s - sx
    none = (E["fk"] == "none").to_numpy()
    d[none] = 0.0
    X["fam_v"] = np.where(none, np.nan, (d[:, 0] + 100 * c) / (d[:, 1] + 100))
    X["fam_dirt"] = np.where(none, np.nan, (d[:, 2] + 100 * c * cd) / (d[:, 0] + 100 * c))   # 家族の賞金のうちダートの割合。擬似件数は 100 走ぶんの賞金（critic）
    X["fam_heads"] = np.where(none, np.nan, np.log1p(d[:, 3]))                       # その日より前に走った家族の頭数（1 号・2 号）
    print("mean", X[NEW].mean().round(4).to_dict(), "sd", X[NEW].std().round(4).to_dict(), "付く割合", X[NEW].notna().mean().round(3).to_dict(), flush=True)
    from src.research.pedigree_lab.walkforward import fit_year
    X = X[X["date"].dt.year >= 2018].reset_index(drop=True)
    base = "--base" in sys.argv                                    # 同じ窓（2018 年から）で (A) だけを当てた比べ先（critic）
    use = [c_ for c_ in ablate.col_sets(X)["pure"] if not c_.startswith("ib_")] + ([] if base else NEW)
    yr = X["date"].dt.year.to_numpy()
    p = np.full(len(X), np.nan)
    for t in range(2020, 2025):
        p[yr == t] = fit_year(X, use, "win", t, 2000)
        print(t, flush=True)
    te = yr >= 2020
    o = X.loc[te, ["race_id", "horse_id"]].copy()
    o["p_TB"] = p[te]
    path = data.OUT / ("base2018_wf.parquet" if base else f"family_wf{'' if null_seed is None else f'_null{null_seed}'}.parquet")
    o.to_parquet(path, index=False)
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
