"""母自身の競走成績の列（log.md「調べ J」。開発の行・時間の順）。

    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=6 nice -n 19 python -m src.research.pedigree_lab.dam_own [--group]

- 列: 母自身の走りの、その日より前の b（aei_cols の本賞金の物差し。b は 2018 年からの母の走りだけで平均する）の平均・
  ex_top3（top3 − min(3, n)/n）の平均・走り数の log1p・3 着内の走りの平均の距離（do_*）。
  記録がそろう母（2014 年以降の生まれ・2 歳の 2016 年から手元にいる）だけに付け、ほかは欠け（打ち切りの偏りを避ける。1 号）。
- 比べ先は (A) ＋ 母の年齢（dam_age_wf_bx_dam_age.parquet）。版も母の年齢の列を入れ、母の若さの分を除く（critic）。
  母の走りは子の走りより必ず前とは限らないので、子のその日より厳密に前の母の走りだけを数える（merge_asof）。
- (A) ＋ 4 列を時間の順（学習は 2017 年から・試す年は 2019〜24 年）で当て、行ごとの予測と「母が手元で走った」印を dam_own_wf.parquet に残す。
- --group: 2 本目の線（母が手元で走った行・2022〜24 年）の版 − (A) を出す（33 本の割り引き）。
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.research.pedigree_lab import ablate, data

NEW = ["do_b", "do_t3", "do_n", "do_dist3"]


def dam_cols(X: pd.DataFrame) -> pd.DataFrame:
    C = pd.read_parquet(data.OUT / "aei_cols.parquet", columns=["race_id", "horse_id", "b"])
    R = X[["race_id", "horse_id", "date", "ex_top3", "top3", "distance"]].merge(C, on=["race_id", "horse_id"], how="left")
    R = R.sort_values(["horse_id", "date"]).reset_index(drop=True)
    g = R.groupby("horse_id")
    R["_n"] = g.cumcount() + 1
    bb = R["b"].where(R["date"] >= "2018-01-01")                          # b は 2018 年からの走りだけ（critic）
    R["_bs"] = bb.fillna(0).groupby(R["horse_id"]).cumsum()
    R["_bn"] = bb.notna().astype(int).groupby(R["horse_id"]).cumsum()
    R["_t3"] = g["ex_top3"].cumsum()
    d3 = np.where(R["top3"] == 1, pd.to_numeric(R["distance"], errors="coerce"), 0.0)
    R["_d3s"] = pd.Series(d3).groupby(R["horse_id"]).cumsum()
    R["_d3n"] = R["top3"].eq(1).astype(float).groupby(R["horse_id"]).cumsum()
    # 同じ日に 2 走は無い前提で、日ごとの最後の累計を母の表にする
    M = R.groupby(["horse_id", "date"]).last().reset_index()[["horse_id", "date", "_n", "_bs", "_bn", "_t3", "_d3s", "_d3n"]]
    M = M.rename(columns={"horse_id": "dam", "date": "ddate"}).sort_values("ddate")
    Q = X[["dam", "date"]].reset_index().rename(columns={"index": "row"}).sort_values("date")
    J = pd.merge_asof(Q, M, left_on="date", right_on="ddate", by="dam", allow_exact_matches=False).set_index("row").sort_index()
    out = pd.DataFrame(index=X.index)
    out["do_b"] = J["_bs"] / J["_bn"].where(J["_bn"] > 0)
    out["do_t3"] = J["_t3"] / J["_n"]
    out["do_n"] = np.log1p(J["_n"])
    out["do_dist3"] = J["_d3s"] / J["_d3n"].where(J["_d3n"] > 0)
    young = (pd.to_numeric(X["dam"].astype(str).str[:4], errors="coerce") >= 2014).to_numpy()
    out.loc[~young, NEW] = np.nan                                         # 記録がそろう母だけ（1 号）
    return out


def main() -> None:
    if "--group" in sys.argv:
        data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "dam_own_group", "母が走った行の版 − (A)（尤度だけ・記述）"))
        X = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "date", "sire", "win", "split"])
        T = pd.read_parquet(data.OUT / "dam_age_wf_bx_dam_age.parquet").rename(columns={"p_TB": "p_A"})[["race_id", "horse_id", "p_A"]].merge(
            pd.read_parquet(data.OUT / "dam_own_wf.parquet"), on=["race_id", "horse_id"]).merge(X, on=["race_id", "horse_id"])
        T = T[T["split"].isin([0, 1]) & T["sire"].fillna("").ne("")]
        w = T["win"].to_numpy(float)

        def ll(p):
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return -(w * np.log(p) + (1 - w) * np.log(1 - p))
        g = ll(T["p_A"].to_numpy()) - ll(T["p_TB"].to_numpy())
        day = T["date"].dt.strftime("%Y%m%d").to_numpy()
        du, di = np.unique(day, return_inverse=True)
        rng = np.random.default_rng(3)
        Wb = np.stack([np.bincount(rng.integers(0, len(du), len(du)), minlength=len(du)) for _ in range(1000)]).astype(float)
        zb = float(norm.ppf(1 - 0.025 / 33))
        y = T["date"].dt.year.to_numpy()
        ran = T["_ran"].to_numpy(bool)

        def stat(mk):
            d = np.bincount(di[mk], weights=g[mk], minlength=len(du)); n = np.bincount(di[mk], minlength=len(du)).astype(float)
            bs = (Wb @ d) / np.maximum(Wb @ n, 1e-9)
            pt, se = float(d.sum() / n.sum()), float(bs.std())
            return {"行": int(mk.sum()), "版 − 比べ先": round(pt, 5), "95%": np.percentile(bs, [2.5, 97.5]).round(5).tolist(),
                    "割り引き（33 本）": [round(pt - zb * se, 5), round(pt + zb * se, 5)]}
        res = {"比べ先": "(A) ＋ 母の年齢（dam_age_wf_bx_dam_age）・物差しは行ごとの二値の対数損失（compare_null と同じ）",
               "母が走った行・2022〜24 年（2 本目の線）": stat(ran & (y >= 2022)),
               "母が走っていない行・2022〜24 年（負の対照・2 号）": stat(~ran & (y >= 2022)),
               "年ごと（母が走った行）": {int(k): stat(ran & (y == k))["版 − 比べ先"] for k in (2022, 2023, 2024)}}
        print(res, flush=True)
        path = data.OUT / "dam_own_group.json"
        path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        data.finish_all(str(path))
        return
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "dam_own", "母自身の競走成績の列の時間の順の比べ（尤度だけ・記述）"))
    X = pd.read_parquet(data.OUT / "feat_v4.parquet")
    X = X[X["date"] <= pd.Timestamp(data.DEV_END)].reset_index(drop=True)
    C = dam_cols(X)
    for c in NEW:
        X[c] = C[c].to_numpy()
    X["_ran"] = X["do_n"].notna()
    print("付く割合（年ごと）", X.groupby(X["date"].dt.year)["_ran"].mean().round(3).to_dict(), flush=True)
    X = X[X["date"].dt.year >= 2017].reset_index(drop=True)
    from src.research.pedigree_lab.walkforward import fit_year
    from src.research.pedigree_lab.dam_age import cols_table
    X = X.merge(cols_table()[["horse_id", "bx_dam_age"]], on="horse_id", how="left")
    use = [c for c in ablate.col_sets(X)["pure"] if not c.startswith("ib_")] + ["bx_dam_age"] + NEW
    yr = X["date"].dt.year.to_numpy()
    p = np.full(len(X), np.nan)
    for t in range(2019, 2025):
        p[yr == t] = fit_year(X, use, "win", t, 2000)
        print(t, flush=True)
    te = yr >= 2019
    o = X.loc[te, ["race_id", "horse_id", "_ran"]].copy()
    o["p_TB"] = p[te]
    path = data.OUT / "dam_own_wf.parquet"
    o.to_parquet(path, index=False)
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
