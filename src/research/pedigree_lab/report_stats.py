"""まとめの材料: 予測器（2 代母の単位の 5 折）の列の重要度（束ね別）と、勝率の較正（開発の OOF の 10 分位）。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.report_stats
出力: OUT/report_stats.json。費用 0 円。
"""
from __future__ import annotations

import json
import re

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.research.pedigree_lab import data

GROUPS = [  # 列の名前 → 束ね（まとめの表の行）
    ("父 × レースの年齢条件・芝ダ（早熟・晩成）", r"^(rel_)?sire_(ra|rasurf)_"),
    ("父 × 芝ダ・距離帯", r"^(rel_)?sire_(surf|sd)_"),
    ("父の全体・クラス・場・馬場", r"^(rel_)?sire_(all|cls|vs|sg)_"),
    ("父の 1 頭 1 票（産駒の勝ち上がり）", r"^(rel_)?sire_v_"),
    ("父の距離の幅", r"^(rel_)?sire_(dz|dsd)"),
    ("父の同じ産駒の中の差（道悪・芝ダ）", r"^(rel_)?sire_(wet_e|surf_e|res_n)"),
    ("母父", r"^(rel_)?bms_"),
    ("父系（父父・父父父）", r"^(rel_)?(ss|sss)_"),
    ("母・きょうだい", r"^(rel_)?dam_"),
    ("全きょうだい", r"^(rel_)?full_"),
    ("牝系（2 代母）", r"^(rel_)?dd_"),
    ("ニックス（父 × 母父）", r"^(rel_)?nick_"),
    ("インブリード", r"^ib_"),
]


def main() -> None:
    d = data.OUT / "models_v4_pure_dd"
    meta = json.loads((d / "win_meta.json").read_text())
    imp = pd.Series(0.0, index=meta["cols"])
    for k in range(5):
        m = lgb.Booster(model_file=str(d / f"win_fold{k}.txt"))
        imp += pd.Series(m.feature_importance("gain"), index=m.feature_name())
    imp = imp / imp.sum()
    grp = {}
    for nm, pat in GROUPS:
        cols = [c for c in imp.index if re.search(pat, c)]
        grp[nm] = {"share": round(float(imp[cols].sum()), 4), "n_cols": len(cols),
                   "top": [(c, round(float(imp[c]), 4)) for c in imp[cols].sort_values(ascending=False).head(3).index]}
    left = [c for c in imp.index if not any(re.search(p, c) for _, p in GROUPS)]
    grp["その他"] = {"share": round(float(imp[left].sum()), 4), "n_cols": len(left), "top": left[:5]}
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "report_calib", "まとめの較正の表（開発の OOF・勝ち）"))
    O = pd.read_parquet(data.OUT / "kf_v4_pure_dd.parquet")
    O = O[O["part"] == "dev"]
    F = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "win", "top3"])
    X = O.merge(F, on=["race_id", "horse_id"])
    X["dec"] = pd.qcut(X["p_win"].rank(method="first"), 10, labels=False)
    cal = X.groupby("dec").agg(pred=("p_win", "mean"), actual=("win", "mean"), n=("win", "size"),
                               pred3=("p_top3", "mean"), actual3=("top3", "mean")).round(4)
    out = {"importance_groups": grp, "top20": [(c, round(float(v), 4)) for c, v in imp.sort_values(ascending=False).head(20).items()],
           "calibration": cal.reset_index().to_dict("records")}
    (data.OUT / "report_stats.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))
    data.finish_all(str(data.OUT / "report_stats.json"))


if __name__ == "__main__":
    main()
