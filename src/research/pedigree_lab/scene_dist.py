"""1 号の新しい場面「初の距離帯」の記述（log.md「新しい場面の記述」。買わない・尤度の α の上乗せだけ・試した数 1）。

定義: 2014 年以降の生まれ・中央の新馬デビュー・キャリア 2 走以上・今回の芝ダで初ダート／初芝ではない・今回の距離帯の経験の数で分ける。
予測: (1) 0 走 > 1 走 > 2 走以上（≈0） (2) 2 帯以上飛ばした行 > 1 帯 (3) 延長でも短縮でも同じ向き (4) 大きさは +0.1〜+0.2。
区別: 初めての重・不良（その芝ダで重・不良が 0 走）の上乗せ。条件に特有の情報が要る（M2）なら ≈0、血統の全体の質（M1）なら正。
点は第 4 版の pure の 2 代母の単位の OOF（開発の行だけ）。gates.py の場面の定義は変えない（rules-council の依頼）。
    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.scene_dist
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data, gates

ORDER = {"S": 0, "M": 1, "I": 2, "L": 3, "E": 4}


def marks(F: pd.DataFrame) -> pd.DataFrame:
    F = F.sort_values(["horse_id", "date"], kind="stable").copy()
    g = F.groupby("horse_id")
    ok = (pd.to_numeric(F["by"], errors="coerce").fillna(0) >= 2014) & (g["cls"].transform("first") == "NEW")
    key = F["horse_id"] + "|" + F["surf"] + "|" + F["dband"]
    one = pd.Series(1, index=F.index)
    n_band = one.groupby(key).cumsum() - 1                      # 同じ芝ダ × 距離帯の、前までの走りの数
    n_surf = one.groupby(F["horse_id"] + "|" + F["surf"]).cumsum() - 1
    bi = F["dband"].map(ORDER).astype(float)
    prev_bi = g["dband"].shift().map(ORDER).astype(float)
    base = ok & (F["career"] >= 2) & (n_surf > 0)               # 初ダート・初芝の行は除く
    F["DB0"] = base & (n_band == 0)
    F["DB1"] = base & (n_band == 1)
    F["DB2"] = base & (n_band >= 2)
    jump = (bi - prev_bi).abs()
    F["DB0_jump2"] = F["DB0"] & (jump >= 2)
    F["DB0_jump1"] = F["DB0"] & (jump <= 1)
    F["DB0_up"] = F["DB0"] & (bi > prev_bi)
    F["DB0_down"] = F["DB0"] & (bi < prev_bi)
    wet = F["going"] == "W"
    n_wet = wet.astype(int).groupby(F["horse_id"] + "|" + F["surf"]).cumsum() - wet.astype(int)
    F["W0"] = ok & (F["career"] >= 2) & wet & (n_wet == 0)
    return F.sort_index()


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "scene_dist", "初の距離帯の記述（尤度の α だけ・買わない）"))
    cols = gates.FCOLS + ["dband", "going"]
    F = marks(pd.read_parquet(data.OUT / "feat_v4.parquet", columns=list(dict.fromkeys(cols))))
    O = pd.read_parquet(data.OUT / "kf_v4_pure_dd.parquet")
    O = O[O["part"] == "dev"][["race_id", "horse_id", "p_win"]]
    X = O.merge(F, on=["race_id", "horse_id"], how="left")
    X = X[X["odds"] > 0].sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    inv = 1 / X["odds"].to_numpy()
    X["q"] = inv / pd.Series(inv).groupby(X["race_id"].to_numpy()).transform("sum").to_numpy()
    X["lp"], X["lq"] = np.log(np.clip(X["p_win"], 1e-5, 1)), np.log(np.clip(X["q"], 1e-5, 1))
    res = {}
    res["steps"] = gates.scene_alpha(X, {k: X[k].to_numpy() for k in ("DB0", "DB1", "DB2")}, n_boot=100)
    res["jump"] = gates.scene_alpha(X, {k: X[k].to_numpy() for k in ("DB0_jump1", "DB0_jump2")}, n_boot=100)
    res["dir"] = gates.scene_alpha(X, {k: X[k].to_numpy() for k in ("DB0_up", "DB0_down")}, n_boot=100)
    res["wet_first"] = gates.scene_alpha(X, {"W0": X["W0"].to_numpy()}, n_boot=100)
    s, j, d = res["steps"], res["jump"], res["dir"]
    a0, a1, a2 = (s[k]["alpha_extra"] for k in ("DB0", "DB1", "DB2"))
    res["judge"] = {"1_steps": bool(a0 > a1 > a2 and abs(a2) < 0.05),
                    "2_jump": bool(j["DB0_jump2"]["alpha_extra"] > j["DB0_jump1"]["alpha_extra"]),
                    "3_both_dirs": bool(d["DB0_up"]["alpha_extra"] > 0 and d["DB0_down"]["alpha_extra"] > 0),
                    "4_size": bool(0.1 <= a0 <= 0.2)}
    res["judge"]["hits"] = int(sum(res["judge"].values()))
    res["judge"]["supports_mechanism"] = bool(res["judge"]["hits"] >= 2 and res["judge"]["1_steps"])
    path = data.OUT / "scene_dist.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
