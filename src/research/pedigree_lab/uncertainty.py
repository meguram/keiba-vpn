"""血統の不確かさと市場の確率の傾き（log.md「調べ O」と「流す前の直し」。記述）。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.uncertainty

- 行: 2019〜24 年の開発の行（split 0・1）。父・母父が分かる行だけ。q は確定オッズの逆数をレースの中で和 1 に直した値。
- u = |父のその芝ダの上乗せ − 母父のその芝ダの上乗せ|（sire_surf_dfp・bms_surf_dfp。芝ダ − 全体の着順の率。適性の食い違い）を、
  年 × 芝ダの中で標準化した値（1 号・2 号・critic）。父の父は父とほぼ同じ向きなので入れない（1 号）。
- 全部の行で、勝ち ~ lq + lq² + u + u·lq + m + m·lq + n + n·lq + f + f·lq + f·u + f·u·lq（lq = logit(q)・m = 2 つの上乗せの平均・
  n = log1p(2 つの件数の小さい方)・f = 初めての芝ダの印）。判定は f·u·lq（初めての芝ダの馬で、食い違いが大きいほど市場の確率の傾きが寝るか）。
  区間は開催日の単位のブートストラップ（500 回）。判定は 95% で行い、97.5% は控えめな参考として並べる。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data
from src.research.pedigree_lab.sire_market import irls

NAMES = ["1", "lq", "lq2", "u", "u_lq", "m", "m_lq", "n", "n_lq", "f", "f_lq", "f_u", "f_u_lq"]


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "uncertainty", "血統の不確かさと市場の確率の傾き（記述）"))
    X = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "date", "split", "win", "odds", "first_surf", "surf",
                                                              "sire", "ds", "sire_surf_dfp", "bms_surf_dfp", "sire_surf_n", "bms_surf_n"])
    X = X[X["date"] <= data.DEV_END].copy()
    od = pd.to_numeric(X["odds"], errors="coerce").where(lambda v: v > 0)
    X["q"] = (1 / od) / (1 / od).groupby(X["race_id"]).transform("sum")
    X = X[X["split"].isin([0, 1]) & X["date"].dt.year.between(2019, 2024) & X["q"].notna()
          & X["sire"].fillna("").ne("") & X["ds"].fillna("").ne("")].reset_index(drop=True)   # 父・母父が分からない行を除く（critic）
    X["y"] = X["date"].dt.year
    raw = (X["sire_surf_dfp"] - X["bms_surf_dfp"]).abs()
    g = [X["y"], X["surf"]]
    X["u"] = (raw - raw.groupby(g).transform("mean")) / raw.groupby(g).transform("std")
    mm = (X["sire_surf_dfp"] + X["bms_surf_dfp"]) / 2
    X["m"] = (mm - mm.groupby(g).transform("mean")) / mm.groupby(g).transform("std")
    nn = np.log1p(np.minimum(X["sire_surf_n"], X["bms_surf_n"]))
    X["n"] = (nn - nn.mean()) / nn.std()
    X = X[X[["u", "m", "n"]].notna().all(axis=1)].reset_index(drop=True)
    qq = X["q"].clip(1e-4, 1 - 1e-4)
    lq = np.log(qq / (1 - qq)).to_numpy()
    lq = lq - lq.mean()
    u, m, n = X["u"].to_numpy(), X["m"].to_numpy(), X["n"].to_numpy()
    f = X["first_surf"].eq("Y").to_numpy(float)
    M = np.column_stack([np.ones(len(X)), lq, lq ** 2, u, u * lq, m, m * lq, n, n * lq, f, f * lq, f * u, f * u * lq])
    yv = X["win"].to_numpy(float)
    b = irls(M, yv)
    day = X["date"].dt.strftime("%Y%m%d").to_numpy()
    du, di = np.unique(day, return_inverse=True)
    idx = [np.flatnonzero(di == k) for k in range(len(du))]
    rng = np.random.default_rng(3)
    bs = []
    for _ in range(500):
        ii = np.concatenate([idx[k] for k in rng.integers(0, len(du), len(du))])
        bs.append(irls(M[ii], yv[ii]))
    bs = np.array(bs)
    res = {"走り": int(len(X)), "初めての芝ダの走り": int(f.sum()), "初めての芝ダの勝ち": int(yv[f > 0].sum()), "係数": {}}
    for i, nm in enumerate(NAMES):
        res["係数"][nm] = [round(float(b[i]), 4), np.percentile(bs[:, i], [2.5, 97.5]).round(4).tolist(),
                         np.percentile(bs[:, i], [1.25, 98.75]).round(4).tolist()]
    print(json.dumps(res, ensure_ascii=False), flush=True)
    path = data.OUT / "uncertainty.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
