"""探す側の記述の 4 巡目: 3 代母（ddd）と父の母（sd）の子孫の 1 頭 1 票を第 4 版に足す（案 G。未確認）。

features2.horse_votes と同じ作り（その日より前に走り始めた・1 年以上前に走り始めた・勝ち上がった・1 勝クラス以上で勝った頭数。
自分は引く）。役の名前は ddd（3 代母）・sdm（父の母）。
    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.features7
出力: OUT/feat_v7g.parquet。費用 0 円。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data, features2

ROLES = {"ddd": "ddd", "sdm": "sd"}


def build() -> pd.DataFrame:
    X = pd.read_parquet(data.OUT / "feat_v4.parquet")
    ped = data.load_ped()[["horse_id", "ddd", "sd"]].rename(columns={"ddd": "_ddd", "sd": "_sd"})
    X = X.merge(ped, on="horse_id", how="left")
    for c in ("_ddd", "_sd"):
        X[c] = X[c].fillna("").astype(str)
    D0 = np.datetime64("2010-01-01")
    day = (X["date"].to_numpy("datetime64[D]") - D0).astype(np.int64)
    hh = X.assign(_d=day)
    first_run = hh.groupby("horse_id")["_d"].min()
    first_win = hh[hh["win"] == 1].groupby("horse_id")["_d"].min()
    first_hi = hh[(hh["win"] == 1) & hh["cls"].isin(["C1", "C2", "C3", "OP"])].groupby("horse_id")["_d"].min()
    H = pd.DataFrame({"fr": first_run}).join(first_win.rename("fw")).join(first_hi.rename("fh"))
    hid = X["horse_id"].to_numpy()
    own = H.reindex(hid)
    for role, src in ROLES.items():
        key = X.drop_duplicates("horse_id").set_index("horse_id")[f"_{src}"]
        empty = key == ""
        codes, _ = pd.factorize(key)
        kmap = pd.Series(codes, index=key.index)
        kq = kmap.reindex(hid).to_numpy()
        res = {}
        for ev, shift in (("fr", 0), ("mat", 365), ("fw", 0), ("fh", 0)):
            col = "fr" if ev == "mat" else ev
            e = H[col].dropna()
            cnt = features2._events_count(kmap.reindex(e.index).to_numpy(), e.to_numpy().astype(np.int64) + shift, kq, day)
            mine_before = np.nan_to_num(own[col].to_numpy() + shift, nan=1e12) < day
            res[ev] = np.where(empty.reindex(hid).to_numpy(), 0, cnt - mine_before.astype(int))
        mat = res["mat"]
        X[f"{role}_v_nrun"] = np.log1p(res["fr"]).astype("float32")
        X[f"{role}_v_win"] = ((np.minimum(res["fw"], mat) + 2 * 0.45) / (mat + 2)).astype("float32")
        X[f"{role}_v_hi"] = ((np.minimum(res["fh"], mat) + 2 * 0.15) / (mat + 2)).astype("float32")
        X[f"rel_{role}_v_win"] = (X[f"{role}_v_win"] - X.groupby("race_id")[f"{role}_v_win"].transform("mean")).astype("float32")
    return X.drop(columns=["_ddd", "_sd"])


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "feat_v7g", "探す側の記述の 4 巡目の列（3 代母・父の母の 1 頭 1 票）"))
    X = build()
    X.to_parquet(data.OUT / "feat_v7g.parquet", index=False)
    print("feat_v7g", X.shape, X[["ddd_v_win", "sdm_v_win"]].describe().round(4).to_string())
    data.finish_all(str(data.OUT / "feat_v7g.parquet"))


if __name__ == "__main__":
    main()
