"""tp-27: 未検証の SmartRC 由来の群（SRC_rota / SRC_equip / SRC_owner / SRC_travel / SRC_ten_agari）の先読み検査。

python -m src.scripts.data.tp27_smartrc_check
  (a) 年ごとの増分: 段階評価 C（stages_C/results.json）の by_fold（valid 2022/2023/2024）から群ごとの年別の Δ勝率。
  (b) 取得日の遅れ: smartrc_race の _meta.scraped_at とレース日の差（日）を年別に集計（tp27/smartrc_scraped_at.parquet）。学習用の過去分は全て出走後の取得。
  (c) 列ごとの単独の結果との関係: 勝ちの AUC（全体・年別）とレース内 Spearman（着順）。V0 の 175 列の同じ指標の分布と比べ、不自然に高い列を挙げる。
出力: data/local/modeling/training_pipeline/tp27/smartrc_check.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import tp27_eval as t

OUT = t.OUT


def _auc(score: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(score)
    s, yy = score[m], y[m]
    if yy.sum() < 50 or (1 - yy).sum() < 50:
        return float("nan")
    r = pd.Series(s).rank(method="average").to_numpy()
    n1, n0 = yy.sum(), len(yy) - yy.sum()
    return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _within_race_rho(df: pd.DataFrame, col: str) -> float:
    d = df[["race_id", "finish_position", col]].dropna()
    if len(d) < 1000:
        return float("nan")
    g = d.groupby("race_id")
    r1 = g[col].rank() - g[col].rank().groupby(d["race_id"]).transform("mean")           # レース内で中心化（レース内で一定の列は 0 になる）
    r2 = g["finish_position"].rank() - g["finish_position"].rank().groupby(d["race_id"]).transform("mean")
    den = np.sqrt((r1 ** 2).sum() * (r2 ** 2).sum())
    return float((r1 * r2).sum() / den) if den > 0 else 0.0


def main() -> None:
    df, base, meta, folds, lp = t.setup()
    raw = t._raw_keys()
    res = t._jload(OUT / "stages_C/results.json")
    report: dict = {}
    # (a) 年別の増分
    per_year = {}
    for s in res.get("steps", []):
        if s.get("kind") == "smartrc" or s["group"] in t.SMARTRC_G4:
            prev = s.get("prev_levels", {}).get("logloss_win")
            per_year[s["group"]] = {"vs_prev": s.get("vs_prev", {}), "by_fold": s.get("by_fold"), "adopted": s.get("adopted")}
    report["per_year_steps"] = per_year
    # (b) 取得日の遅れ
    sc = OUT / "smartrc_scraped_at.parquet"
    if sc.exists():
        a = pd.read_parquet(sc)
        a["race_id"] = a["race_id"].astype(str)
        d = df[["race_id", "date", "year"]].drop_duplicates("race_id").merge(a, on="race_id", how="left")
        d["lag_days"] = (pd.to_datetime(d["scraped_at"], unit="s") + pd.Timedelta(hours=9) - pd.to_datetime(d["date"])).dt.days
        g = d.groupby("year")["lag_days"].describe(percentiles=[0.1, 0.5, 0.9])[["count", "min", "10%", "50%", "90%", "max"]]
        report["lag_days_by_year"] = g.round(1).to_dict("index")
        report["lag_note"] = "lag_days<=0 は発走前の取得。学習用の過去分（2020〜2025）に発走前の取得があるか: " + str(int((d["lag_days"] <= 0).sum())) + " レース"
    # (c) 列ごとの単独指標
    y = (df["finish_position"].to_numpy() == 1).astype(int)
    years = df["year"].to_numpy()
    base_stats = {}
    for c in base:
        if str(df[c].dtype) == "category":
            continue
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float64")
        base_stats[c] = abs(_auc(x, y) - 0.5)
    ref = np.array([v for v in base_stats.values() if np.isfinite(v)])
    report["base_abs_auc_dev"] = {"p50": float(np.percentile(ref, 50)), "p95": float(np.percentile(ref, 95)), "max": float(ref.max())}
    cols_rep = {}
    for g in t.SMARTRC_G4:
        G = t.load_group(f"src:{g}", df, raw)
        rows = []
        for c in G.columns:
            x = pd.to_numeric(G[c], errors="coerce").to_numpy(dtype="float64")
            au = _auc(x, y)
            yr = {int(yv): _auc(x[years == yv], y[years == yv]) for yv in sorted(set(years)) if yv <= 2025}
            rows.append({"col": c, "auc": au, "abs_dev": abs(au - 0.5) if np.isfinite(au) else None, "auc_by_year": yr,
                         "rho_within_race": _within_race_rho(pd.concat([df[["race_id", "finish_position"]], G[[c]]], axis=1), c),
                         "coverage": float(np.isfinite(x).mean())})
        cols_rep[g] = rows
    report["columns"] = cols_rep
    flag = []
    thr = report["base_abs_auc_dev"]["p95"]
    for g, rows in cols_rep.items():
        for r in rows:
            ys = [v for v in r["auc_by_year"].values() if np.isfinite(v)]
            jump = (max(ys) - min(ys)) if ys else 0
            if (r["abs_dev"] or 0) > thr or abs(r["rho_within_race"] or 0) > 0.25 or jump > 0.06:
                flag.append({"group": g, "col": r["col"], "auc": round(r["auc"], 4), "rho": round(r["rho_within_race"], 4), "auc_year_range": round(jump, 4)})
    report["flagged_columns"] = flag
    t._jsave(OUT / "smartrc_check.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("columns", "per_year_steps")}, ensure_ascii=False, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
