"""tp-61: 実測の展開（ペース × 隊列の凝縮度）で『前が残る／差しが届く』を区別できるか。

python -m src.scripts.data.tp61_analysis classes   # レースの分類（しきい値は 2020〜2023 年で固定）と分布
python -m src.scripts.data.tp61_analysis cells     # 展開セル × 脚質の成績（レース単位ブートストラップ）
python -m src.scripts.data.tp61_analysis lltest    # 馬単位の LogLoss（学習 2020〜2023 → 検証 2024・2025）
入力: data/local/modeling/tp61/{race,horse}.parquet（tp61_build）。出力: 同ディレクトリの json / parquet。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "data/local/modeling/tp61"
REF_YEARS = (2020, 2021, 2022, 2023)
FRONT_MAX, STALKER_MAX, MID_MAX = 0.15, 0.35, 0.65
STYLE_CODES = ["front", "stalker", "mid", "closer"]
STYLE_JA = {"front": "逃げ", "stalker": "先行", "mid": "差し", "closer": "追込"}


def dist_band(d: float) -> str:
    d = float(d)
    return "短(~1400)" if d <= 1400 else "マイル(1500~1800)" if d <= 1800 else "中(1900~2200)" if d <= 2200 else "長(2300~)"


def style_code(x: float) -> str | float:
    if x != x:
        return np.nan
    return "front" if x <= FRONT_MAX else "stalker" if x <= STALKER_MAX else "mid" if x <= MID_MAX else "closer"


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    r = pd.read_parquet(D / "race.parquet")
    h = pd.read_parquet(D / "horse.parquet")
    r["band"] = r["distance"].map(dist_band)
    r["k1"] = r["venue"].astype(str) + "|" + r["surface"].astype(str) + "|" + r["distance"].astype(int).astype(str) + "|" + r["going_group"]
    r["k2"] = r["surface"].astype(str) + "|" + r["distance"].astype(int).astype(str) + "|" + r["going_group"]
    r["k3"] = r["surface"].astype(str) + "|" + r["band"] + "|" + r["going_group"]
    return r, h


def add_dev(r: pd.DataFrame) -> pd.DataFrame:
    ref = r[r["year"].isin(REF_YEARS)]
    for col in ("t3f", "race_l3f"):
        out = np.full(len(r), np.nan)
        for key, minn in (("k3", 1), ("k2", 20), ("k1", 15)):          # 細かい順に上書き（件数が足りるものだけ）
            g = ref.groupby(key)[col].agg(["median", "count"])
            g = g[g["count"] >= minn]["median"]
            m = r[key].map(g).to_numpy(dtype=float)
            out = np.where(np.isnan(m), out, m)
        r[f"{col}_med"] = out
        r[f"{col}_dev"] = r[col] - out
    r["bal"] = r["race_l3f_dev"] - r["t3f_dev"]
    return r


def fit_thresholds(r: pd.DataFrame) -> dict:
    ref = r[r["year"].isin(REF_YEARS)]
    q25, q75 = np.nanpercentile(ref["bal"], [25, 75])
    th = {"bal_q25": float(q25), "bal_q75": float(q75)}
    # 凝縮度: 距離帯 × 面ごとの 4C の gap_per_horse の 1/3 分位・2/3 分位
    th["gap"] = {}
    for (s, b), g in ref.groupby(["surface", "band"]):
        x = g["c4_gap_per_horse"].dropna()
        if len(x) >= 100:
            th["gap"][f"{s}|{b}"] = [float(np.nanpercentile(x, 33.3)), float(np.nanpercentile(x, 66.7))]
    return th


def classify(r: pd.DataFrame, th: dict) -> pd.DataFrame:
    r["pace_cls"] = np.where(r["bal"] <= th["bal_q25"], "slow", np.where(r["bal"] >= th["bal_q75"], "high", "mid"))
    r.loc[r["bal"].isna(), "pace_cls"] = np.nan

    def _c(row):
        v = row["c4_gap_per_horse"]
        t = th["gap"].get(f"{row['surface']}|{row['band']}")
        if v != v or t is None:
            return np.nan
        return "compact" if v <= t[0] else "stretched" if v >= t[1] else "normal"

    r["shape_cls"] = r.apply(_c, axis=1)
    return r


def cmd_classes() -> int:
    r, _ = load()
    r = add_dev(r)
    th = fit_thresholds(r)
    r = classify(r, th)
    r.to_parquet(D / "race_cls.parquet", index=False)
    (D / "thresholds.json").write_text(json.dumps(th, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(th, ensure_ascii=False))
    print("race n", len(r), "bal valid", int(r["bal"].notna().sum()), "shape valid", int(r["shape_cls"].notna().sum()))
    print(pd.crosstab(r["pace_cls"], r["shape_cls"], margins=True))
    print(r.groupby("year")[["c4_gap_total", "c4_gap_per_horse", "c4_n_groups", "c4_lead_group_frac", "c4_gap_top5", "fin_spread5", "bal"]].mean().round(3))
    print(r[["bal", "t3f_dev", "race_l3f_dev", "c4_gap_per_horse", "c4_gap_total", "c4_n_groups", "c4_lead_group_frac", "c4_gap_top5", "c3_gap_per_horse", "fin_spread5", "field_size", "distance"]].corr(method="spearman").round(2).to_string())
    return 0


def horse_frame() -> pd.DataFrame:
    r = pd.read_parquet(D / "race_cls.parquet")
    h = pd.read_parquet(D / "horse.parquet")
    h = h.merge(r[["race_id", "pace_cls", "shape_cls", "bal", "t3f_dev", "race_l3f_dev", "band", "c4_gap_per_horse", "c4_gap_total", "c4_n_groups",
                   "c4_lead_group_frac", "c4_gap_top5", "c4_gap_lead", "fin_spread5"]], on="race_id", how="inner")
    h["style"] = h["early_norm"].map(style_code)
    h = h[h["year"] <= 2025]                     # 2026 年はテストセット②（記述・選択に使わない）
    return h.dropna(subset=["style"])


def _boot(rs: pd.DataFrame, value_cols: list[str], n: int = 400, seed: int = 0) -> dict:
    """rs: レース × (value_cols の合計)。レースを復元抽出して比の区間を返す。value_cols = [分子, 分母]"""
    rng = np.random.default_rng(seed)
    num, den = rs[value_cols[0]].to_numpy(float), rs[value_cols[1]].to_numpy(float)
    k = len(rs)
    idx = rng.integers(0, k, size=(n, k))
    vals = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-9)
    return {"mean": float(num.sum() / max(den.sum(), 1e-9)), "lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5))}


def cmd_cells() -> int:
    h = horse_frame()
    h = h[h["field_size"] >= 8]
    ref_rate = h.groupby("style")[["is_top3", "is_win"]].mean()      # 脚質ごとの全体の率（無条件）
    out = []
    for (pc, sc), g in h.dropna(subset=["pace_cls", "shape_cls"]).groupby(["pace_cls", "shape_cls"]):
        for st in STYLE_CODES:
            gs = g[g["style"] == st]
            if len(gs) < 200:
                continue
            per = gs.groupby("race_id").agg(top3=("is_top3", "sum"), win=("is_win", "sum"), n=("is_top3", "size"), l3=("l3f_rank_pct", "sum"), pg=("pos_gain", "sum"))
            exp3 = ref_rate.loc[st, "is_top3"] * per["n"]
            b3 = _boot(pd.DataFrame({"a": per["top3"], "b": exp3}), ["a", "b"])
            bw = _boot(pd.DataFrame({"a": per["win"], "b": ref_rate.loc[st, "is_win"] * per["n"]}), ["a", "b"])
            out.append({"pace": pc, "shape": sc, "style": st, "n_races": int(len(per)), "n_horses": int(len(gs)),
                        "top3_rate": float(gs["is_top3"].mean()), "top3_lift": b3["mean"], "top3_lo": b3["lo"], "top3_hi": b3["hi"],
                        "win_lift": bw["mean"], "win_lo": bw["lo"], "win_hi": bw["hi"],
                        "l3_rank_pct": float(gs["l3f_rank_pct"].mean()), "pos_gain": float(gs["pos_gain"].mean())})
    df = pd.DataFrame(out)
    df.to_csv(D / "cells.csv", index=False)
    pd.set_option("display.width", 220)
    print(df.round(3).to_string())
    return 0


def poly_design(h: pd.DataFrame, extra: list[str], inter: list[str]) -> np.ndarray:
    e = h["early_norm"].to_numpy(float)
    base = np.column_stack([e, e ** 2, e ** 3])
    ctl = np.column_stack([(h["distance"].to_numpy(float) - 1600) / 1000, (h["surface"] == "芝").to_numpy(float), (h["field_size"].to_numpy(float) - 14) / 4])
    cols = [base, ctl]
    for c in extra:
        cols.append(h[[c]].to_numpy(float))
    for c in inter:
        v = h[[c]].to_numpy(float)
        cols.append(base * v)
    return np.hstack(cols)


def cmd_lltest() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss

    h = horse_frame()
    h = h[(h["field_size"] >= 8)].dropna(subset=["early_norm", "bal", "c4_gap_per_horse", "pace_cls", "shape_cls"]).copy()
    for c in ("slow", "high"):
        h[f"pc_{c}"] = (h["pace_cls"] == c).astype(float)
    for c in ("compact", "stretched"):
        h[f"sc_{c}"] = (h["shape_cls"] == c).astype(float)
    h["bal_z"] = (h["bal"] - h["bal"].mean()) / h["bal"].std()
    h["gap_z"] = (h["c4_gap_per_horse"] - h["c4_gap_per_horse"].mean()) / h["c4_gap_per_horse"].std()
    h["lead_z"] = (h["c4_lead_group_frac"] - h["c4_lead_group_frac"].mean()) / h["c4_lead_group_frac"].std()
    h["slow_x_gap"] = h["pc_slow"] * h["gap_z"]
    h["bal_x_gap"] = h["bal_z"] * h["gap_z"]
    tr, va = h[h["year"].isin(REF_YEARS)], h[h["year"].isin([2024, 2025])]
    specs = {
        "M0 位置のみ": ([], []),
        "M1 ペース3段階": (["pc_slow", "pc_high"], ["pc_slow", "pc_high"]),
        "M2 ペース連続(bal)": (["bal_z"], ["bal_z"]),
        "M3 3段階+凝縮度3段階": (["pc_slow", "pc_high", "sc_compact", "sc_stretched"], ["pc_slow", "pc_high", "sc_compact", "sc_stretched"]),
        "M4 連続bal+gap": (["bal_z", "gap_z", "lead_z"], ["bal_z", "gap_z", "lead_z"]),
        "M5 M4+交互作用": (["bal_z", "gap_z", "lead_z", "bal_x_gap"], ["bal_z", "gap_z", "lead_z", "bal_x_gap"]),
        "M6 3段階+スロー×gap": (["pc_slow", "pc_high", "gap_z", "slow_x_gap"], ["pc_slow", "pc_high", "gap_z", "slow_x_gap"]),
    }
    res = {}
    for tgt in ("is_win", "is_top3"):
        for name, (ex, it) in specs.items():
            Xtr, Xva = poly_design(tr, ex, it), poly_design(va, ex, it)
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            m = LogisticRegression(C=1.0, max_iter=400).fit((Xtr - mu) / sd, tr[tgt])
            p = m.predict_proba((Xva - mu) / sd)[:, 1]
            res[(tgt, name)] = log_loss(va[tgt], p)
    base = {t: res[(t, "M0 位置のみ")] for t in ("is_win", "is_top3")}
    for (tgt, name), v in res.items():
        print(f"{tgt:8s} {name:22s} LL={v:.5f}  vs M0 {v - base[tgt]:+.5f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["classes", "cells", "lltest"])
    a = ap.parse_args()
    return {"classes": cmd_classes, "cells": cmd_cells, "lltest": cmd_lltest}[a.cmd]()


if __name__ == "__main__":
    sys.exit(main())
