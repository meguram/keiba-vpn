"""PL2-a・PL2-c の門 0（decisions.md「R0-PL1 の門 0 の判断」。モデルの上の算数だけで、払戻は読まない）。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.pl2 <a|c>

PL2-a: 初ダートの馬で、期待値 = p_B × T-60 の再現オッズ ≥ 1.00・1.10・1.20（3 段）。
  p_B は T-60 の再現の q で当て直した Benter（場面の上乗せ込み・レースの単位の 5 分割の交差）。再現は 5 通りの平均。
  線: E[p_B ÷ q_T60] × 0.86 ≥ 1.27、量 ≥ 年 250 点（全体に直して）、門 E: E[log q_final − log q_T60] ≤ +0.05。
PL2-c: 軸 = R-A、相手 = 同じレースで軸を除いた pure の勝率の 1 位（オッズで選ばない）。
  組の確率を Benter（確定の q で当てた場面の上乗せ込み）と市場の両方で Harville に直し、E[p_組 ÷ q_組] ≥ 1.29、量 ≥ 年 250 点。
出力: OUT/pl2_<a|c>.json。費用 0 円。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data, gates, pre_odds
from src.research.pedigree_lab.hyp_tests import clogit
from src.research.pedigree_lab.win_upper import benter_cross

N_YEARS_DEV = 8 * 0.85


def softmax_q(X: pd.DataFrame, odds: np.ndarray) -> np.ndarray:
    inv = 1.0 / odds
    return inv / pd.Series(inv).groupby(X["race_id"].to_numpy()).transform("sum").to_numpy()


def run_a() -> dict:
    X = gates.load("dd", reason="PL2-a の門 0（T-60 の再現の期待値。払戻は読まない）")
    pool = pre_odds.load_drift("t60")
    sims = pre_odds.simulate(X["race_id"].to_numpy(), X["odds"].to_numpy(float), X["popularity"].to_numpy(float), pool, n=5)
    fd = X["FD"].to_numpy()
    yrs = X["year"].to_numpy()
    res = {"n_sim": 5, "thresholds": {}}
    for thr in (1.00, 1.10, 1.20):
        acc = {"ratio": [], "ratio_s": [], "n_per_year_full": [], "gateE": [], "mean_odds_t60": []}
        for k in range(5):
            o60 = sims[k]
            q60 = softmax_q(X, o60)
            Y = X.copy()
            Y["lq"] = np.log(np.clip(q60, 1e-5, 1))
            pb = benter_cross(Y, fd)
            ev = pb * o60
            sel = fd & (ev >= thr)
            if sel.sum() == 0:
                continue
            r = float(np.mean(pb[sel] / q60[sel]))
            acc["ratio"].append(r)
            acc["ratio_s"].append(r * 0.86)
            acc["n_per_year_full"].append(sel.sum() / N_YEARS_DEV)
            acc["gateE"].append(float(np.mean(np.log(X["q"].to_numpy()[sel]) - np.log(q60[sel]))))
            acc["mean_odds_t60"].append(float(np.mean(o60[sel])))
        res["thresholds"][f"{thr:.2f}"] = {k: round(float(np.mean(v)), 4) for k, v in acc.items() if v}
        t = res["thresholds"][f"{thr:.2f}"]
        t["pass"] = bool(t.get("ratio_s", 0) >= 1.27 and t.get("n_per_year_full", 0) >= 250 and t.get("gateE", 1) <= 0.05)
    return res


def pair_probs(p: np.ndarray, rid: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Harville で、行 ia・ib の 2 頭の馬連（1・2 着）とワイド（ともに 3 着内）の確率。ia・ib は同じレースの行番号。"""
    starts = pd.Series(np.arange(len(rid))).groupby(rid).min()
    umaren, wide = np.zeros(len(ia)), np.zeros(len(ia))
    for n, (a, b) in enumerate(zip(ia, ib)):
        r = rid[a]
        s = starts[r]
        idx = np.flatnonzero(rid[s:s + 18] == r) + s
        q = p[idx] / p[idx].sum()
        i, j = a - s, b - s
        pi, pj = q[i], q[j]
        u = pi * pj / (1 - pi) + pj * pi / (1 - pj)
        w = u
        for kk in range(len(q)):
            if kk in (i, j):
                continue
            pk = q[kk]
            # 3 着が i か j で、もう 1 頭が k の 4 通り
            w += pi * pk / (1 - pi) * pj / (1 - pi - pk)
            w += pk * pi / (1 - pk) * pj / (1 - pk - pi)
            w += pj * pk / (1 - pj) * pi / (1 - pj - pk)
            w += pk * pj / (1 - pk) * pi / (1 - pk - pj)
        umaren[n], wide[n] = u, w
    return umaren, wide


def run_c() -> dict:
    X = gates.load("dd", reason="PL2-c の門 0（軸 R-A の馬連・ワイドの組の比。払戻は読まない）")
    th = gates.thresholds(X)
    R = gates.rules(X, th)
    ax = np.flatnonzero(R["R-A"])
    rid = X["race_id"].to_numpy()
    p = X["p_win"].to_numpy()
    # 相手: 同じレースで軸を除いた pure の勝率の 1 位
    v = p.copy()
    v[ax] = -1
    best = pd.Series(v).groupby(rid).idxmax()
    partner = best.reindex(rid[ax]).to_numpy().astype(int)
    pb = benter_cross(X, X["FD"].to_numpy())
    q = X["q"].to_numpy()
    ub, wb = pair_probs(pb, rid, ax, partner)
    uq, wq = pair_probs(q, rid, ax, partner)
    n8 = X["n_run"].to_numpy()[ax] >= 8
    res = {"pairs": int(len(ax)), "per_year_full": round(len(ax) / N_YEARS_DEV, 1),
           "umaren_E_ratio": round(float(np.mean(ub / uq)), 4), "umaren_sum_ratio": round(float(ub.sum() / uq.sum()), 4),
           "wide_E_ratio": round(float(np.mean(wb[n8] / wq[n8])), 4), "wide_sum_ratio": round(float(wb[n8].sum() / wq[n8].sum()), 4),
           "partner_mean_q": round(float(q[partner].mean()), 4), "axis_mean_q": round(float(q[ax].mean()), 4),
           "partner_E_pb_over_q": round(float(np.mean(pb[partner] / q[partner])), 4)}
    res["pass_umaren"] = bool(res["umaren_E_ratio"] >= 1.29 and res["per_year_full"] >= 250)
    res["wide_per_year_full"] = round(int(n8.sum()) / N_YEARS_DEV, 1)          # ワイドの量は 8 頭以上の組で数える（critic の指摘 6）
    res["pass_wide"] = bool(res["wide_E_ratio"] >= 1.29 and res["wide_per_year_full"] >= 250)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["a", "c"])
    a = ap.parse_args()
    res = run_a() if a.which == "a" else run_c()
    path = data.OUT / f"pl2_{a.which}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
