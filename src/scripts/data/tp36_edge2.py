"""tp-36 (2 続き) Benter 混合の中身の分解と頑健性: 市場だけの傾き（人気薄の過大評価の補正）とモデルの寄与の分離、購入時点のオッズの誤差に対する感度、Kelly 配分。

python -m src.scripts.data.tp36_edge2 --out data/local/modeling/betting/tp36/edge2.json
入力: tp36_edge.py が作る edge_frame.parquet（p_win, q_win, win_odds, ret_win, p_blend）。評価年は 2023〜2025（混合の重みは評価年より前で当てはめ済み）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from src.scripts.data.tp36_edge import blend, boot, fit_benter

ROOT = Path(__file__).resolve().parents[3]
TP = ROOT / "data/local/modeling/betting/tp36"


def scan(d: pd.DataFrame, ev: pd.Series, taus: list[float], years=(2023, 2024, 2025), ret="ret_win") -> list[dict]:
    rows = []
    for t in taus:
        sub = d[(ev >= t) & d["year"].isin(years)]
        if len(sub) == 0:
            continue
        agg = pd.DataFrame({"race_id": sub["race_id"], "stake": 100.0, "ret": sub[ret]}).groupby("race_id").sum().reset_index()
        roi, lo, hi, n, hit = boot(agg, n_boot=600)
        row = {"tau": t, "roi": roi, "lo": lo, "hi": hi, "n_bets": n, "race_hit": hit, "races": int(sub["race_id"].nunique())}
        for y in years:
            a = sub[sub["year"] == y]
            row[str(y)] = float(a[ret].sum() / (100.0 * len(a))) if len(a) else None
        rows.append(row)
    return rows


def market_only(d: pd.DataFrame) -> pd.Series:
    """a=0（モデルを使わず市場だけ）で b を当てはめ、人気薄の過大評価を補正した確率。評価年ごとに、それより前の年で b を当てはめる。"""
    out = pd.Series(np.nan, index=d.index)
    for ey in (2023, 2024, 2025):
        tr = d[d["year"] < ey]
        groups = [(np.log(np.clip(g["q_win"].to_numpy(), 1e-6, 1)), int(np.argmax(g["win"].to_numpy()))) for _, g in tr.groupby("race_id") if g["win"].sum() == 1]

        def nll(b):
            t = 0.0
            for lq, k in groups:
                z = b * lq
                z = z - z.max()
                t -= z[k] - np.log(np.exp(z).sum())
            return t / len(groups)

        b = float(minimize_scalar(nll, bounds=(0.5, 1.5), method="bounded").x)
        m = d["year"] == ey
        z = b * np.log(np.clip(d.loc[m, "q_win"], 1e-6, 1))
        z = z - z.groupby(d.loc[m, "race_id"]).transform("max")
        e = np.exp(z)
        out[m] = e / e.groupby(d.loc[m, "race_id"]).transform("sum")
        out.attrs[f"b{ey}"] = b
    return out


def noisy_decision(d: pd.DataFrame, sigma: float, seed: int, taus: list[float]) -> list[dict]:
    """購入時点のオッズが確定オッズと違う場合の感度: 判断用のオッズに対数正規の誤差を足し（レース内で再正規化）、払戻は確定オッズで精算。"""
    rng = np.random.default_rng(seed)
    d = d.copy()
    d["odds_dec"] = d["win_odds"] * np.exp(rng.normal(0, sigma, len(d)))
    d["q_dec"] = (1 / d["odds_dec"]) / (1 / d["odds_dec"]).groupby(d["race_id"]).transform("sum")
    out = []
    for ey in (2023, 2024, 2025):
        tr = d[d["year"] < ey]
        a, b = fit_benter(tr.assign(q_win=tr["q_dec"]))
        m = d["year"] == ey
        sub = d[m].assign(q_win=d.loc[m, "q_dec"])
        d.loc[m, "p_blend_dec"] = blend(sub, a, b)
    ev = d["p_blend_dec"] * d["odds_dec"]
    return scan(d, ev, taus)


def kelly_growth(d: pd.DataFrame, pcol: str, frac: float, cap: float, min_ev: float, years=(2023, 2024, 2025)) -> dict:
    """配分: 各馬 f_i = frac × max(0, (p×o−1)/(o−1))、レース合計が cap 以内に収まるよう縮小。資金に対する割合で逐次複利。"""
    res = {}
    for y in years:
        a = d[d["year"] == y].sort_values(["date", "race_id"])
        W, peak, mdd = 1.0, 1.0, 0.0
        nb = 0
        for _, g in a.groupby("race_id", sort=False):
            o = g["win_odds"].to_numpy()
            p = g[pcol].to_numpy()
            ev = p * o
            f = np.where(ev >= min_ev, frac * np.clip((ev - 1) / np.maximum(o - 1, 1e-9), 0, None), 0.0)
            tot = f.sum()
            if tot > cap:
                f = f * cap / tot
            if f.sum() <= 0:
                continue
            nb += int((f > 0).sum())
            ret = g["ret_win"].to_numpy() / 100.0
            W *= 1.0 + float((f * (ret - 1.0)).sum())
            peak = max(peak, W)
            mdd = max(mdd, 1 - W / peak)
        res[str(y)] = {"final_wealth": W, "max_drawdown": mdd, "n_bets": nb}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(TP / "edge2.json"))
    a = ap.parse_args()
    d = pd.read_parquet(TP / "edge_frame.parquet")
    d = d[d["year"] >= 2022].reset_index(drop=True)
    taus = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]
    out: dict = {}
    dd = d[d["year"] >= 2023]
    # (a) モデルあり（混合）
    out["blend"] = scan(dd, dd["p_blend"] * dd["win_odds"], taus)
    # (b) 市場だけの傾き（モデルを使わない）
    pm = market_only(d)
    out["market_only_b"] = {k: v for k, v in pm.attrs.items()}
    dm = d.assign(p_mkt=pm)
    dm = dm[dm["year"] >= 2023]
    out["market_only"] = scan(dm, dm["p_mkt"] * dm["win_odds"], taus)
    # (c) 購入時点のオッズの誤差への感度
    out["noise"] = {}
    for sg in (0.05, 0.10, 0.20):
        out["noise"][str(sg)] = noisy_decision(d, sg, seed=1, taus=[0.9, 1.0, 1.1])
        print("noise", sg, flush=True)
    # (d) 帯別（混合 EV≥0.9 の中身）
    sub = dd[(dd["p_blend"] * dd["win_odds"]) >= 0.9]
    rows = []
    for lo, hi in [(1, 3), (3, 6), (6, 10), (10, 20), (20, 50), (50, 10000)]:
        s2 = sub[(sub["win_odds"] >= lo) & (sub["win_odds"] < hi)]
        agg = pd.DataFrame({"race_id": s2["race_id"], "stake": 100.0, "ret": s2["ret_win"]}).groupby("race_id").sum().reset_index()
        roi, l_, h_, n, hit = boot(agg, n_boot=600)
        rows.append({"odds": f"{lo}-{hi}", "roi": roi, "lo": l_, "hi": h_, "n_bets": n})
    out["blend_ev09_by_odds"] = rows
    # (e) Kelly 配分（混合）
    out["kelly"] = {}
    for fr, cap, mev in [(0.25, 0.05, 0.95), (0.25, 0.10, 1.0), (0.5, 0.10, 1.0), (0.25, 0.10, 1.1)]:
        out["kelly"][f"f{fr}_cap{cap}_ev{mev}"] = kelly_growth(dd, "p_blend", fr, cap, mev)
        print("kelly", fr, cap, mev, flush=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("saved", a.out)


if __name__ == "__main__":
    main()
