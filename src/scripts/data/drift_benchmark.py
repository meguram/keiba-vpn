"""ドリフト診断: 『市場が良くなった／世の中が変わった』と『こちらのモデル・入力が悪くなった』を切り分けるための、固定の簡素な基準モデル。

  python -m src.scripts.data.drift_benchmark --out /home/jovyan/work/drift_out/benchmark.json

結果の表(race_result_flat)だけから、as-of（そのレース日より前）の成績の統計（馬・騎手・調教師の勝率/複勝率/着順比、間隔、頭数、馬番）を作り、
2020-2024 年で学習した LightGBM（勝率）を 2025・2026 年にそのまま当てて、月ごとに『基準モデル − 市場』の LogLoss 差を出す。
このモデルは 2026 年を学習に使わず、調教・血統・オッズも使わない。2026 年の 6〜10 月に差が開くなら、こちらの大きなモデルや入力の問題ではなく、市場側や世の中の変化。開かないなら、こちらのモデル・入力側。
指標は報告のみ。モデルの選択には使わない。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import drift_metrics as M
from src.scripts.data.drift_data import load_results_year, market_probs

PRIOR_W, PRIOR_WIN, PRIOR_T3 = 8.0, 0.07, 0.21


def asof_stats(df: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    """key（馬/騎手/調教師）ごとの、そのレース日より前の成績（同日のレースは含めない）。"""
    d = df[[key, "date", "y_win", "y_top3", "pct"]].copy()
    daily = d.groupby([key, "date"]).agg(n=("y_win", "size"), w=("y_win", "sum"), t3=("y_top3", "sum"), pc=("pct", "sum")).reset_index().sort_values([key, "date"])
    for c in ("n", "w", "t3", "pc"):
        daily[f"c_{c}"] = daily.groupby(key)[c].cumsum() - daily[c]
    daily = daily.rename(columns={"c_n": f"{prefix}_n", "c_w": f"{prefix}_w", "c_t3": f"{prefix}_t3", "c_pc": f"{prefix}_pc"})
    out = df[[key, "date"]].merge(daily[[key, "date", f"{prefix}_n", f"{prefix}_w", f"{prefix}_t3", f"{prefix}_pc"]], on=[key, "date"], how="left")
    n = out[f"{prefix}_n"].fillna(0)
    res = pd.DataFrame({
        f"{prefix}_n": n.to_numpy(),
        f"{prefix}_win": ((out[f"{prefix}_w"].fillna(0) + PRIOR_W * PRIOR_WIN) / (n + PRIOR_W)).to_numpy(),
        f"{prefix}_t3": ((out[f"{prefix}_t3"].fillna(0) + PRIOR_W * PRIOR_T3) / (n + PRIOR_W)).to_numpy(),
        f"{prefix}_pct": ((out[f"{prefix}_pc"].fillna(0) + PRIOR_W * 0.5) / (n + PRIOR_W)).to_numpy()}, index=df.index)
    return res


def build(root: Path) -> pd.DataFrame:
    tables = root / "data/page_reference/tables"
    parts = []
    for y in range(2020, 2027):
        r = load_results_year(tables, y)
        r["year"] = y
        parts.append(r)
    d = pd.concat(parts, ignore_index=True)
    ex = []
    for y in range(2020, 2027):
        x = pd.read_parquet(tables / str(y) / "race_result_flat.parquet", columns=["race_id", "horse_id", "jockey_id", "trainer_id"])
        ex.append(x.assign(race_id=x["race_id"].astype(str), horse_id=x["horse_id"].astype(str)))
    d = d.merge(pd.concat(ex).drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left")
    d["y_win"] = (d["finish_position"] == 1).astype(float)
    d["y_top3"] = (d["finish_position"] <= 3).astype(float)
    d["pct"] = d["finish_position"] / d["field_size"].astype(float)
    d = d.sort_values(["date", "race_id"]).reset_index(drop=True)
    for key, pre in (("horse_id", "h"), ("jockey_id", "j"), ("trainer_id", "t")):
        d = d.join(asof_stats(d, key, pre))
    last = d.groupby("horse_id")["date"].shift(1)
    d["days_since"] = (d["date"] - last).dt.days
    d["age"] = pd.to_numeric(d["sex_age"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    d["dist"] = d["distance"].astype(float)
    d["surf"] = (d["surface"].astype(str) == "芝").astype(int)
    d["fs"] = d["field_size"].astype(float)
    d["hn_rel"] = d["horse_number"].astype(float) / d["fs"]
    return d


FEATS = ["h_n", "h_win", "h_t3", "h_pct", "j_n", "j_win", "j_t3", "j_pct", "t_n", "t_win", "t_t3", "t_pct", "days_since", "age", "dist", "surf", "fs", "hn_rel"]


def main() -> None:
    import lightgbm as lgb

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root).resolve()
    d = build(root)
    tr = d[(d["year"] <= 2024) & (d["year"] >= 2021)]
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                               reg_lambda=5.0, verbose=-1, random_state=0)
    model.fit(tr[FEATS], tr["y_win"])
    te = d[d["year"] >= 2025].copy()
    te["p_raw"] = model.predict_proba(te[FEATS])[:, 1]
    te["p_b"] = te["p_raw"] / te.groupby("race_id")["p_raw"].transform("sum")
    mk = []
    for y in (2025, 2026):
        m, _ = market_probs(root / "data/page_reference/tables", y)
        mk.append(m)
    te = te.merge(pd.concat(mk), on=["race_id", "horse_number"], how="inner")
    te["month"] = te["date"].dt.month
    te["dd"] = M.bin_ll(te["y_win"], te["p_b"]) - M.bin_ll(te["y_win"], te["p_mkt"])
    te["ll_b"], te["ll_k"] = M.bin_ll(te["y_win"], te["p_b"]), M.bin_ll(te["y_win"], te["p_mkt"])
    rows = []
    for (y, mth), g in te.groupby(["year", "month"]):
        gap, se = M.cluster_gap_se(g, "dd")
        rows.append({"year": int(y), "month": int(mth), "races": int(g["race_id"].nunique()), "gap": gap, "se": se, "ll_bench": float(g["ll_b"].mean()), "ll_market": float(g["ll_k"].mean())})
    seas = {}
    for y in (2025, 2026):
        for name, lo, hi in (("JanMay", 1, 5), ("JunOct", 6, 10)):
            g = te[(te["year"] == y) & te["month"].between(lo, hi)]
            gap, se = M.cluster_gap_se(g, "dd")
            seas[f"{y}_{name}"] = {"gap": gap, "se": se, "races": int(g["race_id"].nunique()), "ll_bench": float(g["ll_b"].mean()), "ll_market": float(g["ll_k"].mean())}
    imp = sorted(zip(FEATS, model.booster_.feature_importance(importance_type="gain")), key=lambda x: -x[1])
    out = {"train": "2021-2024", "features": FEATS, "monthly": rows, "seasonal": seas, "top_features": [(f, float(v)) for f, v in imp[:8]]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(seas, ensure_ascii=False))


if __name__ == "__main__":
    main()
