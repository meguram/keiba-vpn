"""市場に逆らう馬の抽出（log.md「市場に逆らう馬の抽出」。開発の行・時間の順 2020〜24 年・記述・合否にしない）。

    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=4 nice -n 19 python -m src.research.pedigree_lab.against_market

- 逆らい: 血統の順位 < 人気の順位、かつ 4〜10 番人気。選び方は C1（全部）・C2（その中でオッズが一番高い 1 頭）・C3（人気 − 血統の順位の差が一番大きい 1 頭）と、
  C1 を差の段（1〜2・3〜5・6 以上）で分けた形。
- モデル: 第 4 版（oof_v4_pure）・87 列（oof_v4_compact）・ダートの牝系 5 代（ダートのレースだけ・レースの中の値で順位）。
- 人気とオッズ: 確定と T-60 の再現（5 回それぞれ）。払戻は payoff の確定。区間は開催で束ねたブートストラップ 2,000 回。
- 基準線: 同じ人気の順の馬を全部買う（選んだ馬の人気の順の件数で重み付け）・同じオッズの帯（6 帯）を全部買う（同じく重み付け）。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.research.pedigree_lab import aptitude as ap_
from src.research.pedigree_lab import aptitude_checks as ck
from src.research.pedigree_lab import data, pre_odds
from src.research.pedigree_lab.pop_compare import payouts

N_TESTS = 36
YEARS_FULL = 5 * 0.85
OBANDS = [3, 5, 10, 20, 50]


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "against_market",
                                   "市場に逆らう馬の抽出（的中・A/E・回収率の記述・合否にしない）"))
    cols = ["race_id", "horse_id", "date", "split", "win", "top3", "odds", "horse_number", "n_run", "surf"]
    X = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=cols)
    F = ap_.base_frame()
    A, keys = ap_.group_a(F, "dirt")
    L = ap_.maternal_long(F, drop_common=False)
    F["mat5"], _ = ck.mat_cols(F, A, keys, L)
    X = X.merge(F[["race_id", "horse_id", "mat5"]], on=["race_id", "horse_id"], how="left")
    P4 = pd.read_parquet(data.OUT / "oof_v4_pure.parquet")[["race_id", "horse_id", "p_win"]].rename(columns={"p_win": "p_v4"})
    PC = pd.read_parquet(data.OUT / "oof_v4_compact.parquet")[["race_id", "horse_id", "p_win"]].rename(columns={"p_win": "p_c"})
    S = X[X["split"].isin([0, 1]) & (X["odds"] > 0)].merge(P4, on=["race_id", "horse_id"]).merge(PC, on=["race_id", "horse_id"])
    S = S[S.groupby("race_id")["horse_id"].transform("size") == S["n_run"]]
    S = S[S["date"].dt.year >= 2020].sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    rid = S["race_id"].to_numpy()
    inv = 1 / S["odds"].to_numpy(float)
    q = inv / pd.Series(inv).groupby(rid).transform("sum").to_numpy()
    rk_f = S.groupby("race_id")["odds"].rank(method="first").to_numpy()
    sims = pre_odds.simulate(rid, S["odds"].to_numpy(float), rk_f, pre_odds.load_drift("t60"), n=5)
    views = {"final": [(rk_f, q, S["odds"].to_numpy(float))]}
    views["t60"] = []
    for o in sims:
        qq = (1 / o) / pd.Series(1 / o).groupby(rid).transform("sum").to_numpy()
        views["t60"].append((pd.Series(o).groupby(rid).rank(method="first").to_numpy(), qq, o))
    wp, pp = payouts()
    winp = np.array([wp.get(r, {}).get(int(h), 0.0) for r, h in zip(S["race_id"], S["horse_number"])])
    plp = np.array([pp.get(r, {}).get(int(h), 0.0) for r, h in zip(S["race_id"], S["horse_number"])])
    w, t3 = S["win"].to_numpy(float), S["top3"].to_numpy(float)
    dirt = (S["surf"] == "D").to_numpy()
    mt = S["race_id"].str[:8].to_numpy()
    mu, minv = np.unique(mt, return_inverse=True)
    rng = np.random.default_rng(31)
    Wb = np.stack([np.bincount(rng.integers(0, len(mu), len(mu)), minlength=len(mu)) for _ in range(2000)]).astype(float)
    zb = norm.ppf(1 - 0.025 / N_TESTS)
    scores = {"第 4 版": S["p_v4"].to_numpy(), "87 列": S["p_c"].to_numpy(), "ダートの牝系 5 代": S["mat5"].fillna(0).to_numpy()}

    def matched(v, pick, cat, pool):
        """pool の馬を全部買う回収率を、pick の cat（人気の順・オッズの帯）の件数で重み付けた値（点と回ごと）。"""
        nb = int(cat.max()) + 2
        sm = np.zeros((len(mu), nb)); cn = np.zeros((len(mu), nb)); pk = np.zeros((len(mu), nb))
        np.add.at(sm, (minv[pool], cat[pool]), v[pool]); np.add.at(cn, (minv[pool], cat[pool]), 1)
        np.add.at(pk, (minv[pick], cat[pick]), 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            pt = np.nansum(pk.sum(0) * (sm.sum(0) / cn.sum(0))) / pk.sum()
            bb = np.nansum((Wb @ pk) * ((Wb @ sm) / (Wb @ cn)), axis=1) / (Wb @ pk).sum(1)
        return float(pt), bb

    def block(picks, view_list, pool_mask):
        acc = {k: [] for k in ("n", "win", "top3", "wroi", "proi", "ae", "d_pop", "d_odds")}
        bs = {k: [] for k in ("wroi", "proi", "ae", "d_pop", "d_odds")}
        for pk, (rk, qv, od) in zip(picks, view_list):
            n_ = np.bincount(minv[pk], minlength=len(mu)).astype(float)
            acc["n"].append(pk.sum())
            acc["win"].append(w[pk].mean() if pk.any() else np.nan); acc["top3"].append(t3[pk].mean() if pk.any() else np.nan)
            for k, v in (("wroi", winp), ("proi", plp)):
                s_ = np.bincount(minv[pk], weights=v[pk], minlength=len(mu))
                acc[k].append(s_.sum() / max(n_.sum(), 1))
                with np.errstate(invalid="ignore", divide="ignore"):
                    bs[k].append((Wb @ s_) / (Wb @ n_))
            sw, sq = np.bincount(minv[pk], weights=w[pk], minlength=len(mu)), np.bincount(minv[pk], weights=qv[pk], minlength=len(mu))
            acc["ae"].append(sw.sum() / max(sq.sum(), 1e-9))
            with np.errstate(invalid="ignore", divide="ignore"):
                bs["ae"].append((Wb @ sw) / (Wb @ sq))
            pool = pool_mask & (rk >= 4) & (rk <= 10)
            pr, pb = matched(winp, pk, np.clip(rk.astype(int), 1, 18), pool)
            ob = np.digitize(od, OBANDS)
            orr, obb = matched(winp, pk, ob, pool_mask)
            with np.errstate(invalid="ignore"):
                acc["d_pop"].append(acc["wroi"][-1] - pr); bs["d_pop"].append(bs["wroi"][-1] - pb)
                acc["d_odds"].append(acc["wroi"][-1] - orr); bs["d_odds"].append(bs["wroi"][-1] - obb)
        r = {"per_year": round(float(np.mean(acc["n"])) / YEARS_FULL, 1), "win": round(float(np.nanmean(acc["win"])), 4),
             "top3": round(float(np.nanmean(acc["top3"])), 4)}
        for k in ("wroi", "proi", "ae", "d_pop", "d_odds"):
            pt = float(np.nanmean(acc[k])); b = np.nanmean(bs[k], 0); se = float(np.nanstd(b))
            nd = 3 if k == "ae" else 1
            r[k] = [round(pt, nd), np.nanpercentile(b, [2.5, 97.5]).round(nd).tolist(), [round(pt - zb * se, nd), round(pt + zb * se, nd)]]
        return r
    res = {"rows": int(len(S)), "races": int(S["race_id"].nunique()), "n_tests": N_TESTS, "z_bonf": round(float(zb), 3), "table": {}}
    for mname, sc in scores.items():
        scope = dirt if mname == "ダートの牝系 5 代" else np.ones(len(S), bool)
        prk = pd.Series(np.where(scope, sc, -np.inf)).groupby(rid).rank(ascending=False, method="first").to_numpy()
        for vname, vl in views.items():
            sel = {"C1 全部": [], "C2 一番オッズが高い 1 頭": [], "C3 差が一番大きい 1 頭": [], "C1 差 1〜2": [], "C1 差 3〜5": [], "C1 差 6 以上": []}
            for rk, qv, od in vl:
                base = scope & (prk < rk) & (rk >= 4) & (rk <= 10)
                diff = rk - prk
                sel["C1 全部"].append(base)
                o_ = pd.Series(np.where(base, od, -np.inf)).groupby(rid).rank(ascending=False, method="first").to_numpy()
                sel["C2 一番オッズが高い 1 頭"].append(base & (o_ == 1))
                d_ = pd.Series(np.where(base, diff + 1e-6 * od, -np.inf)).groupby(rid).rank(ascending=False, method="first").to_numpy()
                sel["C3 差が一番大きい 1 頭"].append(base & (d_ == 1))
                sel["C1 差 1〜2"].append(base & (diff <= 2)); sel["C1 差 3〜5"].append(base & (diff >= 3) & (diff <= 5)); sel["C1 差 6 以上"].append(base & (diff >= 6))
            for snm, picks in sel.items():
                key = f"{mname}|{vname}|{snm}"
                res["table"][key] = block(picks, vl, scope)
                print(key, json.dumps(res["table"][key], ensure_ascii=False), flush=True)
    path = data.OUT / "against_market.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
