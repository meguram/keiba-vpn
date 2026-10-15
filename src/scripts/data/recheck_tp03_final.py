#!/usr/bin/env python3
"""特徴量（train_table）を作り直した後に、tp-03 の最終候補の評価をやり直し、採用の結論が変わらないかを確かめる。

tp-03 の ablation は結果をキャッシュして再利用するため、データが変わったときに notebook をそのまま回すと古い数値を使ってしまう。
ここでは同じ条件（LightGBM 固定・3 seed・valid 2022〜2024）で、B0′ / ALL（採用） / 採用に迷った 3 ファミリーを外した案だけを新しく評価する。

  python -m src.scripts.data.recheck_tp03_final [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.pipeline.models import ablation as ab
from src.pipeline.models.compute import detect_compute
from src.pipeline.models.time_split import yearly_expanding_folds

WORK = Path("data/local/modeling/training_pipeline")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=WORK / "recheck_tp03")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    threads = detect_compute(reserve_cores=6).threads

    adopted = json.loads((WORK / "ablation_nospeed/adopted_features.json").read_text(encoding="utf-8"))
    fam = json.loads((WORK / "features/feature_families.json").read_text(encoding="utf-8"))
    b0p = ab.b0_columns(include_speed=False)
    hist = list(adopted["features"])
    df = ab.load_table(WORK / "features/train_table.parquet")
    folds = list(yearly_expanding_folds(df))
    drop = lambda fams: [c for c in hist if not any(c in fam[f] for f in fams)]
    specs = [ab.Spec("B0'", b0p), ab.Spec("ALL(採用)", b0p + hist)]
    specs += [ab.Spec(f"ALL−{f}", b0p + drop([f])) for f in ("E", "F1", "F5")]
    rows = []
    for sp in specs:
        rows += ab.evaluate(df, folds, sp, [0, 1, 2], threads, targets=("win",))
        r = pd.DataFrame(rows)
        print(sp.name, round(float(r[r["exp"] == sp.name]["logloss"].mean()), 5), flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(args.out / "results.csv", index=False)
    s = ab.summarize(r)
    s.to_csv(args.out / "summary.csv")
    print(s.round(5).to_string())
    base = s.loc["ALL(採用)"]
    for name in s.index:
        if name.startswith("ALL−"):
            print(name, "ALL との差(平均)", round(float(s.loc[name, "mean"] - base["mean"]), 5))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
