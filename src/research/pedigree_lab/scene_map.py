"""探す側の記述: 血統が効く場面の地図（尤度の α の上乗せ・第 4 版 pure の 2 代母の単位の OOF・開発の行・未確認・買い方に使わない）。

場面ごとに 1 本ずつ、条件付きロジット [lp, lq, lpc·m, lqc·m, (m)] を当てる（gates.scene_alpha。区間は開催の束ね 100 回）。
    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.scene_map
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data, gates, scene_dist

RACE_LEVEL = {"2 歳戦", "牝馬だけのレース", "重・不良の日"}


def masks(F: pd.DataFrame) -> dict[str, np.ndarray]:
    F = F.sort_values(["horse_id", "date"], kind="stable")
    g = F.groupby("horse_id")
    ok = (pd.to_numeric(F["by"], errors="coerce").fillna(0) >= 2014) & (g["cls"].transform("first") == "NEW")
    isT = (F["surf"] == "T").astype(int)
    nT = isT.groupby(F["horse_id"]).cumsum() - isT
    long_ = (F["distance"] >= 2400).astype(int)
    nL = long_.groupby(F["horse_id"]).cumsum() - long_
    gap = (F["date"] - g["date"].shift()).dt.days
    allF = F.groupby("race_id")["sexb"].transform(lambda s: (s == "F").all())
    m = {
        "初ダート": F["FD"],
        "ダート 2 戦目": F["SD"],
        "初芝": ok & (F["surf"] == "T") & (F["career"] > 0) & (nT == 0),
        "初の重・不良": F["W0"],
        "初の長距離（2400m 以上）": ok & (F["career"] > 0) & (F["distance"] >= 2400) & (nL == 0),
        "初の距離帯": F["DB0"],
        "休み明け（90 日以上）": ok & (gap >= 90),
        "距離の延長（200m 以上）": ok & (F["dchg"] == "U"),
        "距離の短縮（200m 以上）": ok & (F["dchg"] == "D"),
        "2 歳戦": F["rage"] == "A2",
        "牝馬だけのレース": allF,
        "重・不良の日": F["going"] == "W",
    }
    return {k: v.sort_index().to_numpy().astype(bool) for k, v in m.items()}


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "scene_map", "探す側の記述: 場面の地図（尤度だけ）"))
    cols = list(dict.fromkeys(gates.FCOLS + ["dband", "going", "distance", "dchg", "rage", "sexb"]))
    F = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=cols)
    F = scene_dist.marks(gates.scenes(F))
    M = masks(F)
    for k, v in M.items():
        F[f"m_{k}"] = v
    O = pd.read_parquet(data.OUT / "kf_v4_pure_dd.parquet")
    O = O[O["part"] == "dev"][["race_id", "horse_id", "p_win"]]
    X = O.merge(F, on=["race_id", "horse_id"], how="left")
    X = X[X["odds"] > 0].sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    rid = X["race_id"].to_numpy()
    inv = 1 / X["odds"].to_numpy()
    X["q"] = inv / pd.Series(inv).groupby(rid).transform("sum").to_numpy()
    X["lp"], X["lq"] = np.log(np.clip(X["p_win"], 1e-5, 1)), np.log(np.clip(X["q"], 1e-5, 1))
    res = {}
    for k in M:
        r = gates.scene_alpha(X, {k: X[f"m_{k}"].to_numpy()}, n_boot=100)[k]
        res[k] = r
        print(k, r, flush=True)
    path = data.OUT / "scene_map.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
