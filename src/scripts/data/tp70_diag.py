"""tp-70 診断: 予測確率（勝率・連対・複勝）と実績のずれが『どこで』『どれだけ』あるか。

python -m src.scripts.data.tp70_diag prep --frame frame_all.parquet --rank-dir DIR --out DIR
python -m src.scripts.data.tp70_diag run  --base base.parquet --out DIR

(a) 予測ビン別の信頼性 (b) 予測 × 市場（人気区分・市場確率）の二次元 (c) 食い違い（モデル logit − 市場 logit）の十分位別
(d) 年・月別の ECE と食い違い (e) EV 帯別の予測 EV と実現回収率。評価は V4 の OOF（2022〜2025 は leave-one-year-out、2026 は最終モデル）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data.tp70_lib import HIT, KSUM, MKT, PRED, TARGETS, ece, load_base, logloss, logit, pop_class

P_BINS = {"win": [0, 0.01, 0.02, 0.04, 0.07, 0.10, 0.15, 0.20, 0.30, 0.45, 1.0],
          "top2": [0, 0.02, 0.04, 0.07, 0.12, 0.18, 0.25, 0.35, 0.50, 0.70, 1.0],
          "top3": [0, 0.03, 0.06, 0.10, 0.15, 0.22, 0.30, 0.40, 0.55, 0.75, 1.0]}


def prep(a) -> None:
    m = load_base(a.frame, a.rank_dir)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m.to_parquet(out / "base.parquet")
    print(m.shape, m.groupby("year")["race_id"].nunique().to_dict())


def _rel(df: pd.DataFrame, t: str, col: str | None = None) -> pd.DataFrame:
    col = col or PRED[t]
    g = df.assign(b=pd.cut(df[col], P_BINS[t], include_lowest=True)).groupby("b", observed=True).agg(n=(HIT[t], "size"), pred=(col, "mean"), act=(HIT[t], "mean"))
    g["gap"] = g["act"] - g["pred"]
    g.index = g.index.astype(str)
    return g


def run(a) -> None:
    m = pd.read_parquet(a.base)
    m["pc"] = pop_class(m["pop"]).astype(str)
    out = Path(a.out)
    res: dict = {}
    # (a) 全体の信頼性
    res["reliability"] = {t: _rel(m[m["year"] <= 2025], t).round(4).reset_index().to_dict("records") for t in TARGETS}
    # 人気区分 × 予測ビン（複勝・連対・勝率）
    pc_rel = {}
    for t in TARGETS:
        rows = []
        for c, g in m[m["year"] <= 2025].groupby("pc"):
            r = _rel(g, t).reset_index()
            r["pc"] = c
            rows.append(r)
        pc_rel[t] = pd.concat(rows).round(4).to_dict("records")
    res["by_popularity"] = pc_rel
    # (b) 人気区分別の平均と ECE、予測−市場の LogLoss
    rows = []
    for yrs, name in (([2022, 2023, 2024, 2025], "2022-25"), ([2026], "2026")):
        x = m[m["year"].isin(yrs)]
        for t in TARGETS:
            for c, g in list(x.groupby("pc")) + [("全体", x)]:
                rows.append({"period": name, "target": t, "pc": c, "n": len(g), "pred": g[PRED[t]].mean(), "mkt": g[MKT[t]].mean(), "act": g[HIT[t]].mean(),
                             "ll_model": logloss(g[HIT[t]], g[PRED[t]]), "ll_mkt": logloss(g[HIT[t]], g[MKT[t]]), "ece_model": ece(g[HIT[t]], g[PRED[t]]), "ece_mkt": ece(g[HIT[t]], g[MKT[t]])})
    res["summary"] = pd.DataFrame(rows).round(5).to_dict("records")
    # (c) 食い違い（logit 差）十分位別: 実績が市場寄りか
    dis = {}
    for t in TARGETS:
        x = m[m["year"] <= 2025].copy()
        x["gap"] = logit(x[PRED[t]]) - logit(x[MKT[t]])
        x["dq"] = pd.qcut(x["gap"], 10, labels=False, duplicates="drop")
        g = x.groupby("dq").agg(gap=("gap", "mean"), n=("gap", "size"), pred=(PRED[t], "mean"), mkt=(MKT[t], "mean"), act=(HIT[t], "mean"))
        # 実績が モデル と 市場 の間のどこにあるか（0=市場、1=モデル）
        g["w_model"] = (g["act"] - g["mkt"]) / (g["pred"] - g["mkt"]).replace(0, np.nan)
        dis[t] = g.round(4).reset_index().to_dict("records")
    res["disagreement"] = dis
    # (d) 年・月別
    ym = []
    for (y, mo), g in m.groupby(["year", "month"]):
        if len(g) < 800:
            continue
        r = {"year": int(y), "month": int(mo), "n": len(g)}
        for t in TARGETS:
            r[f"ll_gap_{t}"] = logloss(g[HIT[t]], g[PRED[t]]) - logloss(g[HIT[t]], g[MKT[t]])
            r[f"bias_{t}"] = g[HIT[t]].mean() - g[PRED[t]].mean()
        ym.append(r)
    res["year_month"] = pd.DataFrame(ym).round(5).to_dict("records")
    # (e) EV 帯別（単勝・複勝）の予測 EV と実現回収率
    ev = []
    x = m.copy()
    x["ev_win"] = x["p_win"] * x["win_odds"]
    x["real_win"] = x["ret_win"] / 100.0
    pl = x[x["place_odds_min"].notna()].copy()
    pl["po"] = pl["place_odds_min"] + 0.3 * (pl["place_odds_max"] - pl["place_odds_min"])
    pl["ev_place"] = pl["p_top3"] * pl["po"]
    pl["real_place"] = pl["ret_place"] / 100.0
    for name, df, e, r in (("win", x, "ev_win", "real_win"), ("place", pl, "ev_place", "real_place")):
        for period, yrs in (("2022-25", [2022, 2023, 2024, 2025]), ("2026", [2026])):
            d = df[df["year"].isin(yrs)]
            if d.empty:
                continue
            d = d.assign(b=pd.cut(d[e], [0, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 99]))
            g = d.groupby("b", observed=True).agg(n=(e, "size"), ev=(e, "mean"), real=(r, "mean"))
            g.index = g.index.astype(str)
            for b, row in g.iterrows():
                ev.append({"kind": name, "period": period, "bin": b, "n": int(row["n"]), "ev": round(row["ev"], 4), "real": round(row["real"], 4)})
    res["ev_bins"] = ev
    (out / "tp70_diag.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    s = pd.DataFrame(res["summary"])
    print(s[(s["pc"] == "全体")].to_string())
    print(pd.DataFrame(ev).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep")
    p.add_argument("--frame", required=True)
    p.add_argument("--rank-dir", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=prep)
    p = sp.add_parser("run")
    p.add_argument("--base", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=run)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
