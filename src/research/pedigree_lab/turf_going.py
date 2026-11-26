"""芝の馬場と時計の出方での、血統の適性（log.md「芝の馬場と時計の出方」。開発の行・記述・新しい凍結はしない）。

    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=4 nice -n 19 python -m src.research.pedigree_lab.turf_going

- 型: 芝の稍重・芝の重不良・芝の時計の遅い日・芝の時計の速い日。
- 時計の出方: その日・その場・芝の、前のラウンドまでの勝ち時計の z の平均（z は同じ場 × 芝 × 距離 × クラスの、その日より前の勝ち時計の平均と sd）。
  判定するレースの結果は入れない。遅い ≥ +0.5・速い ≤ −0.5。その日の前のラウンドに芝のレースが無い行は外す。
- 適性は aptitude.apt_by（型の中のずれ）、牝系は aptitude_checks.mat_cols、当てはめは aptitude_checks.fit（市場あり・家系の質の制御）。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import aptitude as ap_
from src.research.pedigree_lab import aptitude_checks as ck
from src.research.pedigree_lab import data
from src.research.pedigree_lab.aptitude_period import attach_part

TABLES = "/home/jovyan/work/keiba-vpn/data/page_reference/tables"


def day_speed(F: pd.DataFrame) -> np.ndarray:
    fr = []
    for y in range(2016, 2025):
        d = pd.read_parquet(f"{TABLES}/{y}/race_result_flat.parquet", columns=["race_id", "date", "venue", "surface", "distance", "grade", "round", "finish_position", "time_sec"])
        fr.append(d[pd.to_numeric(d["finish_position"], errors="coerce") == 1])
    W = pd.concat(fr, ignore_index=True).drop_duplicates("race_id")
    W["date"] = pd.to_datetime(W["date"])
    W["t"] = pd.to_numeric(W["time_sec"], errors="coerce")
    W = W.sort_values("date").reset_index(drop=True)                 # 時計の欠けたレースも残す（z だけ欠けにする。critic）
    key = ["venue", "surface", "distance", "grade"]
    # その日より前の同じ鍵の勝ち時計の平均と sd（日の単位で累積し、その日を含めない）
    g = W.groupby(key + ["date"])["t"].agg(["sum", "count"])
    g["sq"] = W.assign(t2=W["t"] ** 2).groupby(key + ["date"])["t2"].sum()
    c = g.groupby(level=list(range(len(key)))).cumsum() - g
    m = c["sum"] / c["count"]
    sd = np.sqrt(np.maximum(c["sq"] / c["count"] - m ** 2, 1e-6))
    idx = pd.MultiIndex.from_frame(W[key + ["date"]])
    W["z"] = (W["t"].to_numpy() - m.reindex(idx).to_numpy()) / sd.reindex(idx).to_numpy()
    W.loc[c["count"].reindex(idx).to_numpy() < 5, "z"] = np.nan          # 基準の件数が少ない鍵は使わない
    W["round"] = pd.to_numeric(W["round"], errors="coerce")
    # その日・その場・同じ面の、前のラウンドまでの z の平均
    W = W.sort_values(["date", "venue", "surface", "round"])
    cnt_prev = W.assign(zn=W["z"].notna().astype(int)).groupby(["date", "venue", "surface"])["zn"].cumsum() - W["z"].notna().astype(int)
    s_prev = W.assign(zz=W["z"].fillna(0)).groupby(["date", "venue", "surface"])["zz"].cumsum() - W["z"].fillna(0)
    W["day_bias"] = np.where(cnt_prev >= 2, s_prev / cnt_prev.clip(lower=1), np.nan)     # 前のラウンドの z が 2 本以上（critic）
    return F["race_id"].map(W.set_index("race_id")["day_bias"]).to_numpy(float)


def cushion(F: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """日付 × 場 → クッション値・芝の含水率（ゴール前）。track_bias.data.cushion_info と同じ読み方（その日の朝の測り・発表は発走前）。"""
    import glob
    rows = []
    for p in sorted(glob.glob("/home/jovyan/work/keiba-vpn/data/page_reference/cushion/cushion_20*.json")):
        rows += json.load(open(p, encoding="utf-8"))
    c = pd.DataFrame(rows)
    c = c[c["date"].astype(str).str.len() >= 10].rename(columns={"venue_name": "venue"})
    for k in ("cushion_value", "turf_moisture_goal"):
        c[k] = pd.to_numeric(c.get(k), errors="coerce")
    c["date"] = pd.to_datetime(c["date"])
    c = c.drop_duplicates(["date", "venue"], keep="last").set_index(["date", "venue"])
    idx = pd.MultiIndex.from_arrays([F["date"], F["venue"]])
    return c["cushion_value"].reindex(idx).to_numpy(float), c["turf_moisture_goal"].reindex(idx).to_numpy(float)


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "turf_going" + ("_cushion" if "--cushion" in __import__("sys").argv else ""), "芝の馬場と時計の出方での血統の適性（尤度だけ・記述）"))
    F = ap_.base_frame()
    turf = (F["surf"] == "T").to_numpy()
    db = day_speed(F)
    types = {"芝の稍重": turf & F["going"].eq("S").to_numpy(), "芝の重・不良": turf & F["going"].eq("W").to_numpy(),
             "芝の時計の遅い日": turf & (db >= 0.5), "芝の時計の速い日": turf & (db <= -0.5)}
    import sys
    if "--cushion" in sys.argv:
        cv, mo = cushion(F)
        types = {"芝・クッション値 〜8.9": turf & (cv < 9), "芝・クッション値 9〜9.9": turf & (cv >= 9) & (cv < 10), "芝・クッション値 10〜": turf & (cv >= 10),
                 "芝・含水率 〜9.9": turf & (mo < 10), "芝・含水率 10〜14.9": turf & (mo >= 10) & (mo < 15), "芝・含水率 15〜": turf & (mo >= 15)}
    L = ap_.maternal_long(F, drop_common=False)
    res = {"rows_with_day_bias": int(np.isfinite(db[turf]).sum()), "turf_rows": int(turf.sum()), "types": {}}
    for nm, A in types.items():
        keys = ["surf"]                           # 基準の全条件の平均を芝の中に揃える（ダートの走りを混ぜない。critic）
        G = F.copy()
        G["z_sire"] = ap_.apt_by(G, A, ["sire"] + keys)
        G["z_bms"] = ap_.apt_by(G, A, ["ds"] + keys)
        G["z_mat5"], G["z_lv5"] = ck.mat_cols(G, A, keys, L)
        G["_A"] = A
        X = attach_part(G, "dev")
        S = X[X["_A"].astype(bool)].reset_index(drop=True)
        Z = {k: ap_.zA(S, S[f"z_{k}"].to_numpy()) for k in ("sire", "bms", "mat5", "lv5")}
        ctrl = ap_.zA(S, S["sire_all_fpct"].fillna(0).to_numpy())
        r = {"rows": int(len(S))}
        for market in (False, True):
            f = ck.fit(S, [ctrl, Z["lv5"], Z["sire"], Z["bms"], Z["mat5"]], ["ctrl", "lv5", "sire", "bms", "mat5"], market, 200)
            r["mkt" if market else "nomkt"] = {k: f[k] for k in ("sire", "bms", "mat5")} | {"c": f["c"], "bms_plus_mat5": f["bms_plus_mat5"]}
        res["types"][nm] = r
        print(nm, json.dumps(r, ensure_ascii=False), flush=True)
    path = data.OUT / ("turf_going_cushion.json" if "--cushion" in sys.argv else "turf_going.json")
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
