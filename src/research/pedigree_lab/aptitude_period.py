"""8 観点の残った候補（dirt・short）の牝系の列を、走りの期 2016〜19 と 2020〜24 に分けて当て直す（記述・未確認）。
列は as-of のまま（期ごとに作り直さない）。家系の質（lv5）と父の全体の水準を制御に入れ、市場なし・市場ありの両方。

    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=4 nice -n 19 python -m src.research.pedigree_lab.aptitude_period <観点>
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from src.research.pedigree_lab import aptitude as ap_
from src.research.pedigree_lab import aptitude_checks as ck
from src.research.pedigree_lab import data
import pandas as pd


def attach_part(F: pd.DataFrame, part: str) -> pd.DataFrame:
    """explore5.attach と同じ作りで、OOF の部分（dev・hold）を選ぶ。"""
    O = pd.read_parquet(data.OUT / "kf_v4_pure_dd.parquet")
    O = O[O["part"] == part][["race_id", "horse_id", "p_win"]]
    X = O.merge(F, on=["race_id", "horse_id"], how="left")
    X = X[X["odds"] > 0].sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    rid = X["race_id"].to_numpy()
    inv = 1 / X["odds"].to_numpy()
    X["q"] = inv / pd.Series(inv).groupby(rid).transform("sum").to_numpy()
    X["lp"], X["lq"] = np.log(np.clip(X["p_win"], 1e-5, 1)), np.log(np.clip(X["q"], 1e-5, 1))
    return X


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("what")
    p.add_argument("--boot", type=int, default=200)
    p.add_argument("--mode", choices=["period", "exp", "turf", "hold"], default="period",
                   help="period=2 期・exp=それまでのその面の出走数（0〜1 / 5 以上。1 号の R2）・turf=列を芝の行に当てる補の型・hold=保留 split 2 で 1 回（2 号の R2）")
    a = p.parse_args()
    purpose = "confirm" if a.mode == "hold" else "select"
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), purpose, f"aptitude_period_{a.what}_{a.mode}",
                                   f"8 観点の残った候補の確かめ: {a.what} {a.mode}（尤度だけ・{'保留の 1 回' if a.mode == 'hold' else '記述'}）"))
    F = ap_.base_frame()
    A, keys = ap_.group_a(F, a.what)
    L = ap_.maternal_long(F, drop_common=False)
    F["z_sire"] = ap_.apt_by(F, A, ["sire"] + keys)
    F["z_bms"] = ap_.apt_by(F, A, ["ds"] + keys)
    F["z_mat5"], F["z_lv5"] = ck.mat_cols(F, A, keys, L)
    F["_A"] = A
    # それまでのその芝ダの出走数（その日より前・2016 年以降の手元の走りだけ）
    F["_nsurf"] = F.sort_values("date").groupby(["horse_id", "surf"]).cumcount().reindex(F.index).to_numpy()
    X = attach_part(F, "hold" if a.mode == "hold" else "dev")
    if a.mode == "turf":
        S0 = X[X["surf"] == "T"].reset_index(drop=True)
    else:
        S0 = X[X["_A"].astype(bool)].reset_index(drop=True)
    res = {"what": a.what, "mode": a.mode}
    if a.mode == "exp":
        young = pd.to_numeric(S0["horse_id"].str[:4]) >= 2014    # 2016 年より前の走りが無い古馬を外す（critic 2 回目の 3）
        parts = {"exp0_1": (S0["_nsurf"] <= 1) & young, "exp5p": (S0["_nsurf"] >= 5) & young}
    elif a.mode in ("turf", "hold"):
        parts = {a.mode: pd.Series(True, index=S0.index)}
    else:
        y = S0["date"].dt.year
        parts = {"2016_19": (y >= 2016) & (y <= 2019), "2020_24": (y >= 2020) & (y <= 2024)}
    for per, m in parts.items():
        S = S0[m.to_numpy()].reset_index(drop=True)
        Z = {k: ap_.zA(S, S[f"z_{k}"].to_numpy()) for k in ("sire", "bms", "mat5", "lv5")}
        ctrl = ap_.zA(S, S["sire_all_fpct"].fillna(0).to_numpy())
        for market in (False, True):
            r = ck.fit(S, [ctrl, Z["lv5"], Z["sire"], Z["bms"], Z["mat5"]], ["ctrl", "lv5", "sire", "bms", "mat5"], market, a.boot)
            res[f"{per}_{'mkt' if market else 'nomkt'}"] = {"rows": len(S), "mat5": r["mat5"], "bms_plus_mat5": r["bms_plus_mat5"], "sire": r["sire"]}
            print(per, market, json.dumps(res[f"{per}_{'mkt' if market else 'nomkt'}"], ensure_ascii=False), flush=True)
    path = data.OUT / f"aptitude_period_{a.what}_{a.mode}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
