"""tp-70 較正後の三確率（勝率・連対・複勝）を割引 Harville に入れたときの、馬連・ワイド・3連複の確率の当たり具合。

python -m src.scripts.data.tp70_exotic run --preds tp70_cal_preds.parquet --races 600 --out DIR

組合せ（全ペア・全 3 頭組）ごとに『予測確率 vs 的中』の LogLoss と ECE を、確率の入力別（生 / 市場のみ / C3 / C6）に比べる。
レースは評価期間ごとに無作為抽出（seed 固定）。払戻は使わない（確率の当たり具合のみ）。
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.inference.exotic_probabilities import fit_rank_distribution
from src.scripts.data.tp70_lib import logloss

VARIANTS = {
    "raw": ("p_win", "p_top2", "p_top3"),
    "mkt": ("m_win", "m_top2", "m_top3"),
    "C3": ("cal_C3_win", "cal_C3_top2", "cal_C3_top3"),
    "C6": ("cal_C6_win", "cal_C6_top2", "cal_C6_top3"),
}


def one_race(g: pd.DataFrame) -> dict:
    rk = g["rank"].to_numpy()
    order = np.argsort(rk)
    first2 = set(order[:2].tolist())
    first3 = set(order[:3].tolist())
    n = len(g)
    pairs = list(combinations(range(n), 2))
    trios = list(combinations(range(n), 3))
    y_um = np.array([1.0 if set(p) == first2 else 0.0 for p in pairs])
    y_wd = np.array([1.0 if set(p) <= first3 else 0.0 for p in pairs])
    y_tr = np.array([1.0 if set(t) == first3 else 0.0 for t in trios])
    out = {}
    for name, (cw, c2, c3) in VARIANTS.items():
        dist, _ = fit_rank_distribution(g[cw].to_numpy(), g[c2].to_numpy(), g[c3].to_numpy(), default_alpha2=1.0, default_alpha3=1.0)
        um = dist.all_umaren()
        wd = dist.all_wide()
        tr = dist.all_sanrenpuku()
        out[name] = {"um": np.array([um[p] for p in pairs]), "wd": np.array([wd[p] for p in pairs]), "tr": np.array([tr[t] for t in trios])}
    return {"y_um": y_um, "y_wd": y_wd, "y_tr": y_tr, "pred": out}


def run(args) -> None:
    d = pd.read_parquet(args.preds)
    d = d[d["year"] >= 2023]
    rng = np.random.default_rng(0)
    res = {}
    for per, ys, nr in (("2023-25", [2023, 2024, 2025], args.races), ("2026", [2026], args.races // 2)):
        ids = d[d["year"].isin(ys)]["race_id"].unique()
        pick = set(rng.choice(ids, min(nr, len(ids)), replace=False).tolist())
        acc = {v: {k: [] for k in ("um", "wd", "tr")} for v in VARIANTS}
        ys_ = {k: [] for k in ("um", "wd", "tr")}
        for rid, g in d[d["race_id"].isin(pick)].groupby("race_id"):
            r = one_race(g.reset_index(drop=True))
            for k in ("um", "wd", "tr"):
                ys_[k].append(r[f"y_{k}"])
                for v in VARIANTS:
                    acc[v][k].append(r["pred"][v][k])
        res[per] = {}
        for k in ("um", "wd", "tr"):
            y = np.concatenate(ys_[k])
            row = {"n_items": int(len(y)), "hit_rate": float(y.mean())}
            for v in VARIANTS:
                p = np.concatenate(acc[v][k])
                row[v] = {"ll": logloss(y, p), "mean_pred": float(p.mean())}
                # 高確率側 1% の実績（買う候補になる部分）の予測と実績
                top = p >= np.quantile(p, 0.99)
                row[v]["top1pct_pred"] = float(p[top].mean())
                row[v]["top1pct_act"] = float(y[top].mean())
            res[per][k] = row
        print(per, json.dumps({k: {v: round(res[per][k][v]["ll"], 5) for v in VARIANTS} for k in res[per]}, ensure_ascii=False))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tp70_exotic.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("run")
    p.add_argument("--preds", required=True)
    p.add_argument("--races", type=int, default=600)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=run)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
