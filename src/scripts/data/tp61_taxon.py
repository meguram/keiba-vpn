"""tp-61: 文献（解説記事・指標）の分類を軸に分解した MECE のラップ型（規則）を実測に当てはめ、頻度・脚質別成績・データ駆動クラスタとの一致を見る。

python -m src.scripts.data.tp61_taxon dist     # 規則に使う量の分布（距離帯・芝ダ別）
python -m src.scripts.data.tp61_taxon assign   # しきい値を 2020〜2023 年で決め、全レースを分類。頻度・脚質別・クラスタとの一致・LogLoss
規則は src/pipeline/inference/pace_type.py（lap_type_rule）。しきい値は config/pace_type.json に出力する。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.inference import pace_type as pt

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "data/local/modeling/tp61"
REF = (2020, 2021, 2022, 2023)


def frame() -> pd.DataFrame:
    f = pd.read_parquet(D / "lapshape.parquet")
    r = pd.read_parquet(D / "race_cls.parquet")[["race_id", "year", "surface", "band", "distance", "pace_cls", "shape_cls", "bal", "t3f_dev", "race_l3f_dev"]]
    d = f.merge(r, on="race_id", how="inner").dropna(subset=["f_front", "f_last3", "mid_max", "accel_len", "bal"])
    d["lull_amp"] = d["mid_max"] - (d["f_front"] + d["f_last3"]) / 2
    return d


def cmd_dist() -> int:
    d = frame()
    ref = d[d["year"].isin(REF)]
    pd.set_option("display.width", 200)
    print(ref[["bal", "lull_amp", "accel_len", "l2_step", "last_drop", "f_front", "f_last3"]].describe(percentiles=[.1, .25, .5, .75, .9]).round(3).to_string())
    print(ref.groupby(["surface", "band"])[["lull_amp", "accel_len", "bal"]].quantile([.25, .5, .75]).round(3).unstack().to_string())
    return 0


def cmd_assign() -> int:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss

    from src.scripts.data.tp61_analysis import STYLE_CODES, _boot, horse_frame, poly_design

    d = frame()
    ref = d[d["year"].isin(REF)]
    th = {"bal_q25": float(np.nanpercentile(ref["bal"], 25)), "bal_q75": float(np.nanpercentile(ref["bal"], 75)), "by_band": {}}
    for bd, g in ref.groupby("band"):
        th["by_band"][bd] = {"lull_amp": round(float(np.nanpercentile(g["lull_amp"], 70)), 3), "accel_len": round(float(np.nanpercentile(g["accel_len"], 50)), 3)}
    print("thresholds", th)
    d["lap_type"] = [pt.lap_type_rule(b, la, ac, th, bd) for b, la, ac, bd in zip(d["bal"], d["lull_amp"], d["accel_len"], d["band"])]
    out_th = {"thresholds": th, "types": pt.LAP_TYPES}
    (D / "taxon_thresholds.json").write_text(json.dumps(out_th, ensure_ascii=False, indent=1), encoding="utf-8")
    print("頻度（全体）\n", d[d["year"].isin(REF)]["lap_type"].value_counts(normalize=True).round(3).to_string())
    t = pd.crosstab([d["surface"], d["band"]], d["lap_type"], normalize="index").round(3)
    print(t.to_string())
    print("件数（距離帯 × 型 の最小）", pd.crosstab([d["surface"], d["band"]], d["lap_type"]).min().min())
    # データ駆動クラスタとの一致
    k6 = pd.read_parquet(D / "laptype_k6.parquet")[["race_id", "lap_type"]].rename(columns={"lap_type": "km6"})
    m = d.merge(k6, on="race_id")
    print("ARI(rule, km6)", round(adjusted_rand_score(m["lap_type"], m["km6"]), 3), "NMI", round(normalized_mutual_info_score(m["lap_type"], m["km6"]), 3))
    print(pd.crosstab(m["lap_type"], m["km6"], normalize="index").round(2).to_string())
    d[["race_id", "lap_type", "lull_amp"]].to_parquet(D / "taxon.parquet", index=False)

    h = horse_frame().merge(d[["race_id", "lap_type"]], on="race_id", how="inner")
    h = h[h["field_size"] >= 8].dropna(subset=["early_norm", "bal", "c4_gap_per_horse", "c4_lead_group_frac"])
    ref_rate = h.groupby("style")[["is_top3", "is_win"]].mean()
    rows = []
    for ty, g in h.groupby("lap_type"):
        for st in STYLE_CODES:
            gs = g[g["style"] == st]
            per = gs.groupby("race_id").agg(top3=("is_top3", "sum"), win=("is_win", "sum"), n=("is_top3", "size"))
            b3 = _boot(pd.DataFrame({"a": per["top3"], "b": ref_rate.loc[st, "is_top3"] * per["n"]}), ["a", "b"])
            bw = _boot(pd.DataFrame({"a": per["win"], "b": ref_rate.loc[st, "is_win"] * per["n"]}), ["a", "b"])
            rows.append({"type": ty, "style": st, "n_races": len(per), "top3_lift": b3["mean"], "t3_lo": b3["lo"], "t3_hi": b3["hi"], "win_lift": bw["mean"],
                         "w_lo": bw["lo"], "w_hi": bw["hi"], "l3_rank_pct": float(gs["l3f_rank_pct"].mean())})
    ct = pd.DataFrame(rows)
    ct.to_csv(D / "taxon_cells.csv", index=False)
    pd.set_option("display.width", 200)
    print(ct.round(3).to_string())
    # LogLoss
    types = pt.LAP_TYPES
    for i, ty in enumerate(types[1:], 1):
        h[f"ty_{i}"] = (h["lap_type"] == ty).astype(float)
    tyc = [f"ty_{i}" for i in range(1, len(types))]
    for c in ("slow", "high"):
        h[f"pc_{c}"] = (h["pace_cls"] == c).astype(float)
    h["gap_z"] = (h["c4_gap_per_horse"] - h["c4_gap_per_horse"].mean()) / h["c4_gap_per_horse"].std()
    h["lead_z"] = (h["c4_lead_group_frac"] - h["c4_lead_group_frac"].mean()) / h["c4_lead_group_frac"].std()
    specs = {"M0 位置のみ": ([], []), "M1 ペース3段階": (["pc_slow", "pc_high"], ["pc_slow", "pc_high"]),
             "M7' 文献MECE型": (tyc, tyc), "M8' 文献MECE型+凝縮度": (tyc + ["gap_z", "lead_z"], tyc + ["gap_z", "lead_z"]),
             "M1+凝縮度": (["pc_slow", "pc_high", "gap_z", "lead_z"], ["pc_slow", "pc_high", "gap_z", "lead_z"])}
    tr, va = h[h["year"].isin(REF)], h[h["year"].isin([2024, 2025])]
    res = {}
    for tgt in ("is_win", "is_top3"):
        for name, (ex, it) in specs.items():
            Xtr, Xva = poly_design(tr, ex, it), poly_design(va, ex, it)
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            mdl = LogisticRegression(C=1.0, max_iter=500).fit((Xtr - mu) / sd, tr[tgt])
            res[(tgt, name)] = log_loss(va[tgt], mdl.predict_proba((Xva - mu) / sd)[:, 1])
    for (tgt, name), v in res.items():
        print(f"{tgt:8s} {name:22s} LL={v:.5f} vs M0 {v - res[(tgt, 'M0 位置のみ')]:+.5f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dist", "assign"])
    a = ap.parse_args()
    return {"dist": cmd_dist, "assign": cmd_assign}[a.cmd]()


if __name__ == "__main__":
    sys.exit(main())
