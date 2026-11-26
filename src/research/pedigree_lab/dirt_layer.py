"""ダートの牝系 5 代 × 人気薄 × 道悪の重なりの層の、勝ちの倍率（統括役の 11/26 16:30 の依頼。開発の行・記述・探した数に入れる）。

    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=4 nice -n 19 python -m src.research.pedigree_lab.dirt_layer

- 層の格子（数を見る前に固定・18 個）:
  - 人気: 4 番人気以下・7 番人気以下・10 番人気以下
  - 馬場: すべて・稍重〜不良
  - 牝系の点: レースの中で中心化した dirt_mat5 が、その年より前のダートの開発の行の分位で上位 10%・20%・33%（as-of の閾値。
    最初の年はその年の分位で代える）。血統の無い馬（牝系の祖先が 1 頭も無い）は層に入れない。
- 倍率 = 行ごとの「勝ち ÷ q」の平均（均等買いの回収率に当たる物差し。pl2 の線 E[p ÷ q] と同じ形。critic 5 回目の 1）。
  - 確定の版: 人気も q も確定オッズ。線は 1.27。
  - T-60 の版: 再現オッズ（pre_odds.simulate・5 回）ごとに、人気の順位と q_T60 を作り直して層を選び（確定の人気を使わない。critic の 2）、
    回ごとの倍率を平均する。線は 1.48（= 1.27 / 0.86）。
- 区間: 開催日で束ねたブートストラップ 2,000 回の 95%。選び過ぎの割引きは、正規近似の Bonferroni（確定と T-60 の 2 本 × 18 = 36 本）。
- 牝系の上乗せ: 同じ引き直しで「層の倍率 ÷ 牝系で絞らない層の倍率」の区間も出す。
- 年の点数は、開発の行（開催の 85%）を全体に直した値（pl2 と同じ 8 × 0.85 年）。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.research.pedigree_lab import data, pre_odds
from src.research.pedigree_lab import dirt_deep as dd

LINE_FINAL, LINE_T60 = 1.27, 1.27 / 0.86
N_YEARS_DEV = 8 * 0.85
N_TESTS = 36


def asof_threshold(zc: np.ndarray, year: np.ndarray, ok: np.ndarray, tq: float) -> np.ndarray:
    thr = np.full(len(zc), np.nan)
    ys = np.unique(year)
    for y in ys:
        prior = ok & (year < y)
        src = zc[prior] if prior.sum() > 1000 else zc[ok & (year == y)]
        thr[year == y] = np.quantile(src, 1 - tq)
    return thr


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "dirt_layer",
                                   "ダートの牝系 × 人気薄 × 道悪の層の勝ちの倍率（開発の行・記述・18 個の格子・critic 5 回目の直し）"))
    F, A, keys, L = dd.setup("dirt")
    has_mat = np.zeros(len(F), bool)
    has_mat[np.unique(L["row"].to_numpy())] = True
    F["_hasmat"] = has_mat
    S = dd.dirt_rows(F).sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    rid = S["race_id"].to_numpy()
    zc = S["z_mat5"].to_numpy() - pd.Series(S["z_mat5"].to_numpy()).groupby(rid).transform("mean").to_numpy()
    ok = S["_hasmat"].to_numpy(bool)
    year = S["date"].dt.year.to_numpy()
    wet = S["going"].isin(["S", "W"]).to_numpy()
    win = S["win"].to_numpy(float)
    rk_f = S.groupby("race_id")["odds"].rank(method="first").to_numpy()
    q_f = S["q"].to_numpy()
    pool = pre_odds.load_drift("t60")
    sims = pre_odds.simulate(rid, S["odds"].to_numpy(float), rk_f, pool, n=5)
    views = [("final", rk_f, q_f)]
    for o in sims:
        inv = 1 / o
        qt = inv / pd.Series(inv).groupby(rid).transform("sum").to_numpy()
        rk_t = pd.Series(o).groupby(rid).rank(method="first").to_numpy()
        views.append(("t60", rk_t, qt))
    day = S["date"].dt.strftime("%Y%m%d").to_numpy()
    du, dinv = np.unique(day, return_inverse=True)
    rng = np.random.default_rng(11)
    W = np.stack([np.bincount(rng.integers(0, len(du), len(du)), minlength=len(du)) for _ in range(2000)]).astype(float)
    zb = norm.ppf(1 - 0.025 / N_TESTS)
    thr = {tq: asof_threshold(zc, year, ok, tq) for tq in (0.10, 0.20, 0.33)}

    def ratio(mk, q):
        r_d = np.bincount(dinv[mk], weights=win[mk] / q[mk], minlength=len(du))
        n_d = np.bincount(dinv[mk], minlength=len(du)).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            b = (W @ r_d) / (W @ n_d)
        return r_d.sum() / max(n_d.sum(), 1), b, n_d.sum()

    def summarize(cell_masks):
        """cell_masks: [(view, mask_cell, mask_base)]。確定は 1 本、T-60 は 5 回の平均。"""
        out = {}
        for kind in ("final", "t60"):
            rs, bs, rb, bb, ns = [], [], [], [], []
            for v, mk, mb in cell_masks:
                if v[0] != kind:
                    continue
                r, b, n = ratio(mk, v[2])
                r0, b0, _ = ratio(mb, v[2])
                rs.append(r); bs.append(b); rb.append(r0); bb.append(b0); ns.append(n)
            r, b = float(np.mean(rs)), np.nanmean(np.stack(bs), axis=0)
            lift = np.nanmean(np.stack(bs) / np.stack(bb), axis=0)
            se = float(np.nanstd(b))
            out[kind] = {"rows": int(np.mean(ns)), "per_year_full": round(float(np.mean(ns)) / N_YEARS_DEV, 1),
                         "ratio": round(r, 3), "ci95": np.nanpercentile(b, [2.5, 97.5]).round(3).tolist(),
                         "ci_bonf36": [round(r - zb * se, 3), round(r + zb * se, 3)],
                         "base_ratio": round(float(np.mean(rb)), 3),
                         "lift_vs_base": round(r / float(np.mean(rb)), 3), "lift_ci95": np.nanpercentile(lift, [2.5, 97.5]).round(3).tolist(),
                         "line": round(LINE_FINAL if kind == "final" else LINE_T60, 3)}
        return out

    res = {"n_tests": N_TESTS, "z_bonf": round(float(zb), 3), "cells": {}}
    for pk, plo in {"4 番人気以下": 4, "7 番人気以下": 7, "10 番人気以下": 10}.items():
        for gk, gm in {"全馬場": np.ones(len(S), bool), "稍重〜不良": wet}.items():
            for tq in (0.10, 0.20, 0.33):
                cm = []
                for v in views:
                    base = (v[1] >= plo) & gm & ok
                    cm.append((v, base & (zc >= thr[tq]), base))
                k = f"{pk}・{gk}・牝系 上位 {int(tq * 100)}%"
                res["cells"][k] = summarize(cm)
                print(k, json.dumps(res["cells"][k], ensure_ascii=False), flush=True)
    res["max_final"] = max(res["cells"], key=lambda k: res["cells"][k]["final"]["ratio"])
    res["max_t60"] = max(res["cells"], key=lambda k: res["cells"][k]["t60"]["ratio"])
    path = data.OUT / "dirt_layer.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
