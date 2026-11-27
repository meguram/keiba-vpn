"""危ない人気馬（log.md「調べ N」。記述・買い方の設計ではない）。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.fav_doubt

- 行: 2019〜24 年の開発の行（split 0・1）の 1〜3 番人気（確定オッズの人気）。p_A は rulebased_wf の (A) の時間の順の予測。
- z = log p_A − log q（p_A と q はどちらもレースの中で和 1 に直した値）。下位 2 割の線は年 × 人気の中の 20% 点。
- 勝ち ~ logit(q) + logit(q)² + 人気の印 2 本 + 下位 2 割の印 のロジスティック回帰の印の係数。区間は開催日の単位のブートストラップ（1,000 回）。
- 群: 全部の 1〜3 番人気・条件が変わった 1〜3 番人気（初ダート・初芝〔first_surf == "Y"〕・前走との距離の差 400m 以上）が主の 2 本。
  初ダート・初芝・距離の延長は記述だけ。p_A はレースの中で和 1 に直す。下位 2 割の線は年 × 人気の中で決める。回帰に人気の印 2 本を足す。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data
from src.research.pedigree_lab.sire_market import irls


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "fav_doubt", "血統が低く見る人気馬と市場の見込み（記述）"))
    X = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "date", "split", "win", "odds", "popularity",
                                                              "first_surf", "distance", "prev_dist", "surf"])
    X = X[X["date"] <= data.DEV_END].copy()
    od = pd.to_numeric(X["odds"], errors="coerce").where(lambda v: v > 0)
    X["q"] = (1 / od) / (1 / od).groupby(X["race_id"]).transform("sum")
    P = pd.read_parquet(data.OUT / "rulebased_wf.parquet")[["race_id", "horse_id", "p_A"]]
    X = X.merge(P, on=["race_id", "horse_id"])
    assert X["p_A"].notna().all()
    X["p_A"] = X["p_A"] / X.groupby("race_id")["p_A"].transform("sum")       # レースの中で和 1 に直す（q とそろえる。critic）
    X = X[X["split"].isin([0, 1]) & (pd.to_numeric(X["popularity"], errors="coerce") <= 3) & X["q"].notna()].reset_index(drop=True)
    X["z"] = np.log(X["p_A"].clip(1e-6)) - np.log(X["q"])
    X["y"] = X["date"].dt.year
    X["pop"] = pd.to_numeric(X["popularity"], errors="coerce").astype(int)
    # 下位 2 割の線は「年 × 人気（1・2・3）」の中で決める（z の低さが人気の高さで決まらないように。1 号・2 号）
    X["low"] = (X["z"] <= X.groupby(["y", "pop"])["z"].transform(lambda s: s.quantile(0.2))).astype(float)
    dd = pd.to_numeric(X["distance"], errors="coerce") - pd.to_numeric(X["prev_dist"], errors="coerce")
    X["first_d"] = X["first_surf"].eq("Y") & X["surf"].eq("D")                 # 初ダート（"Y"/"N" の文字列。critic）
    X["first_t"] = X["first_surf"].eq("Y") & X["surf"].eq("T")
    X["chg"] = X["first_surf"].eq("Y") | (dd.abs() >= 400)
    X["ext"] = dd >= 400                                                       # 距離の延長
    qq = X["q"].clip(1e-4, 1 - 1e-4)
    X["lq"] = np.log(qq / (1 - qq))
    rng = np.random.default_rng(3)
    groups = (("全部の 1〜3 番人気（主）", np.ones(len(X), bool)), ("条件が変わった 1〜3 番人気（主）", X["chg"].to_numpy(bool)),
              ("初ダート（記述）", X["first_d"].to_numpy(bool)), ("初芝（記述）", X["first_t"].to_numpy(bool)), ("距離の延長 400m 以上（記述）", X["ext"].to_numpy(bool)))
    res = {"人気ごとの下位 2 割の件数": X.groupby("pop")["low"].sum().astype(int).to_dict()}
    for nm, mk in groups:
        D = X[mk].reset_index(drop=True)
        lq = D["lq"].to_numpy()
        M = np.column_stack([np.ones(len(D)), lq, lq ** 2, D["pop"].eq(2).to_numpy(float), D["pop"].eq(3).to_numpy(float), D["low"].to_numpy()])
        yv = D["win"].to_numpy(float)
        b = irls(M, yv)
        day = D["date"].dt.strftime("%Y%m%d").to_numpy()
        du, di = np.unique(day, return_inverse=True)
        idx = [np.flatnonzero(di == k) for k in range(len(du))]
        bs = []
        for _ in range(1000):
            ii = np.concatenate([idx[k] for k in rng.integers(0, len(du), len(du))])
            bs.append(irls(M[ii], yv[ii])[5])
        low = D["low"].to_numpy(bool)
        res[nm] = {"走り": int(len(D)), "下位 2 割の走り": int(low.sum()), "下位 2 割の勝ち": int(yv[low].sum()),
                   "下位 2 割の A/E（q に対して）": round(float(yv[low].sum() / D.loc[low, "q"].sum()), 3),
                   "それ以外の A/E": round(float(yv[~low].sum() / D.loc[~low, "q"].sum()), 3),
                   "印の係数": round(float(b[5]), 3), "95%": np.percentile(bs, [2.5, 97.5]).round(3).tolist(),
                   "97.5%（2 本の Bonferroni）": np.percentile(bs, [1.25, 98.75]).round(3).tolist()}
        print(nm, res[nm], flush=True)
    path = data.OUT / "fav_doubt.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
