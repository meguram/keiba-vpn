"""tp-36 (2 続き) 馬連・ワイド・馬単の複合買い: v4 の確率（市場なし）と Benter 混合（市場込み）から割引 Harville で組合せ確率を出し、期待値 EV=確率×確定オッズ で均等買いの回収率を測る。

python -m src.scripts.data.tp36_exotic --out data/local/modeling/betting/tp36/exotic.json
入力: tp36_edge.py が作る edge_frame.parquet（2023〜2025 年は p_blend = walk-forward の混合）、odds_pair / payoff（確定オッズ・払戻）。
ワイドの判断用オッズは (最小+最大)/2、精算は実際の払戻。混合の確率は市場オッズを使うので『市場なし』とは分けて報告する。
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.inference.exotic_probabilities import RankDistribution, fit_rank_distribution
from src.scripts.data.tp36_edge import boot

ROOT = Path(__file__).resolve().parents[3]
BET = ROOT / "data/local/modeling/betting"
TP = BET / "tp36"


def pair_probs(win, top2, top3) -> dict[str, np.ndarray]:
    dist, _ = fit_rank_distribution(win, top2, top3)
    p2, p3 = dist._p2, dist._p3
    umaren = p2 + p2.T
    x12, x13, x23 = p3.sum(2), p3.sum(1), p3.sum(0)
    w = x12 + x13 + x23
    wide = w + w.T
    return {"umaren": umaren, "wide": wide, "umatan": p2}


def race_rows(g: pd.DataFrame, scheme: str) -> list[tuple]:
    g = g.sort_values("horse_number")
    nums = g["horse_number"].astype(int).to_numpy()
    if scheme == "model":
        w, t2, t3 = g["p_win"].to_numpy(), g["p_top2"].to_numpy(), g["p_top3"].to_numpy()
    else:
        w = g["p_blend"].to_numpy()
        r = (g["p_blend"] / g["p_win"].clip(lower=1e-6)).clip(0.2, 5.0).to_numpy()
        t2, t3 = np.clip(g["p_top2"].to_numpy() * r, 1e-6, 0.99), np.clip(g["p_top3"].to_numpy() * r, 1e-6, 0.99)
    pp = pair_probs(w, t2, t3)
    rows = []
    n = len(nums)
    for i, j in combinations(range(n), 2):
        a, b = int(nums[i]), int(nums[j])
        rows.append(("umaren", f"{a}-{b}", pp["umaren"][i, j]))
        rows.append(("wide", f"{a}-{b}", pp["wide"][i, j]))
    for i in range(n):
        for j in range(n):
            if i != j:
                rows.append(("umatan", f"{int(nums[i])}>{int(nums[j])}", pp["umatan"][i, j]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(TP / "exotic.json"))
    ap.add_argument("--max-races", type=int, default=0)
    a = ap.parse_args()
    d = pd.read_parquet(TP / "edge_frame.parquet")
    d = d[d["year"] >= 2023].copy()
    rids = d["race_id"].unique()
    if a.max_races:
        rids = rids[: a.max_races]
        d = d[d["race_id"].isin(set(rids))]
    pair = pd.read_parquet(BET / "odds_pair.parquet", columns=["race_id", "bet_type", "combo", "odds", "odds_min", "odds_max"])
    pair = pair[pair["race_id"].isin(set(rids))]
    pair["odds_dec"] = np.where(pair["bet_type"] == "wide", (pair["odds_min"] + pair["odds_max"]) / 2, pair["odds"])
    pay = pd.read_parquet(BET / "payoff.parquet")
    pay = pay[pay["race_id"].isin(set(rids)) & pay["bet_type"].isin(["umaren", "wide", "umatan"])]
    pay = pay[~pay["ambiguous"] & pay["parsed"]][["race_id", "bet_type", "combo", "payout"]]
    bad = set(pd.read_parquet(BET / "payoff.parquet", columns=["race_id", "ambiguous", "parsed"]).query("ambiguous or not parsed")["race_id"])
    out: dict = {"races": int(len(rids)), "schemes": {}}
    for scheme in ("model", "mixed"):
        rows = []
        for k, (rid, g) in enumerate(d.groupby("race_id", sort=False)):
            for bt, combo, p in race_rows(g, scheme):
                rows.append((rid, bt, combo, p))
            if k % 2000 == 0:
                print(scheme, k, flush=True)
        pr = pd.DataFrame(rows, columns=["race_id", "bet_type", "combo", "p"])
        m = pr.merge(pair[["race_id", "bet_type", "combo", "odds_dec"]], on=["race_id", "bet_type", "combo"], how="inner").merge(pay, on=["race_id", "bet_type", "combo"], how="left")
        m = m[~m["race_id"].isin(bad) & (m["odds_dec"] > 0)]
        m["payout"] = m["payout"].fillna(0.0)
        m["ev"] = m["p"] * m["odds_dec"]
        m["year"] = m["race_id"].str[:4].astype(int)
        res = {}
        for bt in ("umaren", "wide", "umatan"):
            mb = m[m["bet_type"] == bt]
            lst = []
            for tau in (0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0):
                s = mb[mb["ev"] >= tau]
                if len(s) == 0:
                    continue
                agg = pd.DataFrame({"race_id": s["race_id"], "stake": 100.0, "ret": s["payout"]}).groupby("race_id").sum().reset_index()
                roi, lo, hi, n, hit = boot(agg, n_boot=600)
                row = {"tau": tau, "roi": roi, "lo": lo, "hi": hi, "n_bets": n, "race_hit": hit, "bet_hit": float((s["payout"] > 0).mean()), "races": int(len(agg))}
                for y in (2023, 2024, 2025):
                    sy = s[s["year"] == y]
                    row[str(y)] = float(sy["payout"].sum() / (100.0 * len(sy))) if len(sy) else None
                lst.append(row)
            res[bt] = lst
            print(scheme, bt, "done", flush=True)
        out["schemes"][scheme] = res
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("saved", a.out)


if __name__ == "__main__":
    main()
