"""V1: 前走の再採点。過去走を「着差」と「その走の相手の後走成績」で採点し直す（記事の主張 A）。

過去走 r（馬 h）の相手 o のその後の成績は、h の今回のレースの日 T より前に走ったものだけを使う（nd < T）。
  - aft  : 相手の後走の着順の相対位置（1 着 = 1）の、縮約した平均。J=1 は次走だけ、J=3 は次走以降 3 走まで。
  - own  : exp(-着差秒/0.5)。着差込みの自分の結果（1 着 = 1）。
  - rescore: own × aft / 0.5。強い相手に僅差 → 高く、弱い相手に勝っただけ → 割り引く。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .common import finish_pct

SHRINK_K = 5.0
PRIOR = 0.5   # 着順の相対位置の平均は定義上 0.5 に近い
MARGIN_SCALE = 0.5
LAGS = (1, 2, 3)
OUT_COLS = ["pr_aft1", "pr_aft1_n3", "pr_aft_l3", "pr_own1", "pr_own_l3", "pr_rescore1", "pr_rescore1_n3", "pr_rescore_l3", "pr_rescore_max3", "pr_gain_l3"]


def _nanagg(fn, arrs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(np.vstack(arrs), axis=0)


def _agg(q: pd.DataFrame, nx: pd.DataFrame) -> pd.DataFrame:
    """q（cur・horse_id・T・pr_race）に、相手の後走（nx）を結び、nd < T かつ相手が自分でないものを数える。"""
    m = q.merge(nx, left_on="pr_race", right_on="race_id", suffixes=("", "_o"))
    m = m[(m["horse_id"] != m["horse_id_o"]) & (m["nd"] < m["T"])]
    return m.groupby(["cur", "horse_id", "L"]).agg(s=("np", "sum"), n=("np", "size")).reset_index()


def _shrunk(a: pd.DataFrame, key: pd.DataFrame, name: str) -> pd.Series:
    x = key.merge(a, on=["cur", "horse_id", "L"], how="left")
    return ((x["s"].fillna(0.0) + SHRINK_K * PRIOR) / (x["n"].fillna(0.0) + SHRINK_K)).rename(name)


def build(runs: pd.DataFrame, first_year: int | None = None) -> pd.DataFrame:
    """runs: load_runs() の出力。返り値は race_id・horse_id と OUT_COLS。first_year 以降のレースだけを出す（履歴は全期間を使う）。"""
    r = runs.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True).copy()
    r["pct"] = finish_pct(r)
    win_t = r.groupby("race_id")["time_sec"].transform("min")
    r["own"] = np.exp(-(r["time_sec"] - win_t).clip(lower=0.0) / MARGIN_SCALE)
    r.loc[r["finish_position"].isna(), "own"] = np.nan
    g = r.groupby("horse_id", sort=False)
    nx = pd.concat([pd.DataFrame({"race_id": r["race_id"], "horse_id": r["horse_id"], "j": j, "nd": g["date"].shift(-j), "np": g["pct"].shift(-j)}) for j in (1, 2, 3)])
    nx = nx.dropna(subset=["nd", "np"])
    nx1 = nx[nx["j"] == 1].drop(columns="j")
    nx3 = nx.drop(columns="j")
    cur = r.rename(columns={"race_id": "cur"})[["cur", "horse_id", "date"]].rename(columns={"date": "T"})
    lagged = {L: pd.DataFrame({"cur": r["race_id"], "horse_id": r["horse_id"], "T": r["date"], "L": L, "pr_race": g["race_id"].shift(L), "own": g["own"].shift(L)}) for L in LAGS}
    years = r["date"].dt.year
    lo = first_year if first_year is not None else int(years.min())
    outs = []
    for y in sorted(set(years[years >= lo])):
        sel = (years == y).to_numpy()
        qs = {L: lagged[L][sel].dropna(subset=["pr_race"]) for L in LAGS}
        base = cur[sel].reset_index(drop=True)
        f = base[["cur", "horse_id"]].copy()
        own = {}
        aft = {}
        for L in LAGS:
            q = qs[L]
            key = lagged[L][sel][["cur", "horse_id", "L"]].reset_index(drop=True)
            has = lagged[L][sel]["pr_race"].notna().to_numpy()
            a1 = _agg(q, nx1)
            v = _shrunk(a1, key, f"aft{L}").to_numpy().copy()
            v[~has] = np.nan
            aft[L] = v
            o = lagged[L][sel]["own"].to_numpy().copy()
            o[~has] = np.nan
            own[L] = o
        q1 = qs[1]
        key1 = lagged[1][sel][["cur", "horse_id", "L"]].reset_index(drop=True)
        has1 = lagged[1][sel]["pr_race"].notna().to_numpy()
        a3 = _shrunk(_agg(q1, nx3), key1, "aft3").to_numpy().copy()
        a3[~has1] = np.nan
        rs = {L: own[L] * aft[L] / PRIOR for L in LAGS}
        f["pr_aft1"] = aft[1]
        f["pr_aft1_n3"] = a3
        f["pr_aft_l3"] = _nanagg(np.nanmean, [aft[L] for L in LAGS])
        f["pr_own1"] = own[1]
        f["pr_own_l3"] = _nanagg(np.nanmean, [own[L] for L in LAGS])
        f["pr_rescore1"] = rs[1]
        f["pr_rescore1_n3"] = own[1] * a3 / PRIOR
        f["pr_rescore_l3"] = _nanagg(np.nanmean, [rs[L] for L in LAGS])
        f["pr_rescore_max3"] = _nanagg(np.nanmax, [rs[L] for L in LAGS])
        f["pr_gain_l3"] = f["pr_rescore_l3"] - f["pr_own_l3"]
        outs.append(f.rename(columns={"cur": "race_id"}))
    return pd.concat(outs, ignore_index=True)
