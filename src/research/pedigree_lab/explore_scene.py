"""段 3 の探り（開発の行 split 0・1 だけ）: 場面の馬を血統の点・血統と市場の差で 5 分位に分けた回収率。

探す側の 1 回として台帳（select）に残し、試した数に数える。保留（split 2）と 2025・26 年は読まない。
    nice -n 19 python -m src.research.pedigree_lab.explore_scene --kf kf_v3_pure_meeting --scene 初ダート
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import bets, data
from src.research.pedigree_lab.benter import SCENES


def load_dev_bets(kf: str, tag: str = "v3", reason: str = "") -> pd.DataFrame:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", f"{kf}_explore", reason))
    O = pd.read_parquet(data.OUT / f"{kf}.parquet")
    O = O[O["part"] == "dev"]
    F = pd.read_parquet(data.OUT / f"feat_{tag}.parquet")
    keep = ["race_id", "horse_id", "meeting", "date", "win", "top3", "odds", "popularity", "n_run", "debut", "first_surf",
            "surf", "dchg", "going", "cls", "horse_number", "payoff", "split", "sire_surf_top3", "sire_all_top3",
            "bms_surf_top3", "rel_sire_sd_top3", "dam_v_win"]
    X = O.merge(F[[c for c in keep if c in F.columns]], on=["race_id", "horse_id"], how="left")
    assert X["split"].isin([0, 1]).all() and X["date"].max() <= pd.Timestamp("2024-12-31")
    X["year"] = X["date"].dt.year
    X = X[X["odds"] > 0].copy()
    inv = 1.0 / X["odds"].to_numpy()
    X["q"] = inv / pd.Series(inv).groupby(X["race_id"].to_numpy()).transform("sum").to_numpy()
    X["edge"] = np.log(X["p_win"]) - np.log(X["q"])
    X["prank"] = X.groupby("race_id")["p_win"].rank(ascending=False, method="first")
    return bets.attach_pay(X)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kf", default="kf_v3_pure_meeting")
    ap.add_argument("--scene", default="初ダート")
    a = ap.parse_args()
    X = load_dev_bets(a.kf, reason=f"段 3 の探り: 場面 {a.scene} の 5 分位の回収率（試した数に数える）")
    msk = SCENES[a.scene](X)
    S = X[msk].copy()
    out = {"scene": a.scene, "rows": int(msk.sum()), "all": {k: bets.roi(S, np.ones(len(S), bool), k) for k in ("win", "place")}}
    for score in ("p_win", "edge", "sire_surf_top3"):
        S["qt"] = pd.qcut(S[score].rank(method="first"), 5, labels=False)
        out[score] = {}
        for q in range(5):
            sel = (S["qt"] == q).to_numpy()
            out[score][q] = {k: bets.roi(S, sel, k, n_boot=200) for k in ("win", "place")}
    # レースの中で血統 1 位かつ場面の馬
    sel = (S["prank"] == 1).to_numpy()
    out["race_top1"] = {k: bets.roi(S, sel, k) for k in ("win", "place")}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    (data.OUT / f"explore_{a.kf}_{a.scene}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    data.finish_all(str(data.OUT / f"explore_{a.kf}_{a.scene}.json"))


if __name__ == "__main__":
    main()
