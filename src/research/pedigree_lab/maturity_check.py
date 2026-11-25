"""確かめ 2（1 号）: 休み明け × 成長の型で使った「早熟の度合い」の向きを、父の名前で確かめる（記述だけ・着順は読まない）。

早熟の度合い = (父 × レースの年齢条件 × 芝ダの 3 着内率 − 父 × 芝ダの率) × 向き（2・3 歳戦は +、3 歳以上・4 歳以上戦は −）。
休み明けの行（2014 年以降の生まれ・中央の新馬デビュー・90 日以上）で父ごとに平均し、産駒の休み明けが 200 行以上の父の上位・下位 8 頭。
    nice -n 19 python -m src.research.pedigree_lab.maturity_check
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data
from src.research.pedigree_lab.sire_profile import sire_names


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "maturity_check", "早熟の度合いの向きの確かめ（記述だけ）"))
    X = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["horse_id", "date", "sire", "rage", "layoff", "by", "cls",
                                                                 "sire_rasurf_top3", "sire_surf_top3"])
    X = X.sort_values(["horse_id", "date"], kind="stable")
    ok = (pd.to_numeric(X["by"], errors="coerce").fillna(0) >= 2014) & (X.groupby("horse_id")["cls"].transform("first") == "NEW")
    X = X[ok & (X["layoff"] == "Y")]
    sign = np.where(X["rage"].isin(["A2", "A3"]), 1.0, -1.0)
    X = X.assign(early=(X["sire_rasurf_top3"] - X["sire_surf_top3"]) * sign)
    g = X.groupby("sire")["early"].agg(["mean", "size"])
    g = g[g["size"] >= 200].sort_values("mean", ascending=False)
    ids = set(g.head(8).index) | set(g.tail(8).index)
    names = sire_names(ids)
    out = {"early_top": [(names.get(i, i), round(float(g.loc[i, "mean"]), 4), int(g.loc[i, "size"])) for i in g.head(8).index],
           "late_top": [(names.get(i, i), round(float(g.loc[i, "mean"]), 4), int(g.loc[i, "size"])) for i in g.tail(8).index]}
    print(json.dumps(out, ensure_ascii=False))
    (data.OUT / "maturity_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    data.finish_all(str(data.OUT / "maturity_check.json"))


if __name__ == "__main__":
    main()
