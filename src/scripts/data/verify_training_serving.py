"""推論用の追い切り特徴（training_serving）が、学習時の TL_A（tp-19b の cand_TL_A.parquet）と一致するかを、実データで確かめる。

python -m src.scripts.data.verify_training_serving --races 150 --year 2025
読み取りのみ（GCS の horse_training を読む）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features import training_serving as ts
from src.scraper.storage import HybridStorage

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "data/local/modeling/training_pipeline"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--races", type=int, default=150)
    ap.add_argument("--year", type=int, default=2025)
    a = ap.parse_args()
    laps = pd.read_parquet(WORK / "features/training_laps.parquet")
    ref = ts.build_reference_table(laps)
    print("参照表", ref.shape, flush=True)
    tl_train = pd.read_parquet(WORK / "training_laps_b/cand_TL_A.parquet")
    tbl = pd.read_parquet(WORK / "features/train_table.parquet", columns=["race_id", "horse_id", "date"])
    tbl["race_id"] = tbl["race_id"].astype(str); tbl["horse_id"] = tbl["horse_id"].astype(str)
    tbl["date"] = pd.to_datetime(tbl["date"])
    sub = tbl[tbl["date"].dt.year == a.year]
    races = sub.drop_duplicates("race_id").sample(n=min(a.races, sub["race_id"].nunique()), random_state=0)["race_id"].tolist()
    st = HybridStorage(".")
    cols = [c for c in tl_train.columns if c not in ("race_id", "horse_id")]
    tl_idx = tl_train.set_index(["race_id", "horse_id"])
    stat = {c: [0, 0] for c in cols}          # [一致, 比較した行]
    n_rows = n_missing = 0
    for i, rid in enumerate(races):
        g = sub[sub["race_id"] == rid]
        rd = g["date"].iloc[0].strftime("%Y-%m-%d")
        recs = {h: st.load("horse_training", h) for h in g["horse_id"]}
        got = ts.tl_a_for_race(recs, rd, ref)
        for h in g["horse_id"]:
            n_rows += 1
            want = tl_idx.loc[(rid, h)] if (rid, h) in tl_idx.index else None
            if want is None:
                continue
            if h not in got.index:
                n_missing += int(want["has_training"] == 1) if "has_training" in want.index else 0
                continue
            for c in cols:
                if c not in got.columns:
                    continue
                x, y = float(got.loc[h, c]), float(want[c])
                ok = (np.isnan(x) and np.isnan(y)) or (np.isfinite(x) and np.isfinite(y) and abs(x - y) <= 1e-3 + 1e-3 * abs(y))
                stat[c][0] += int(ok); stat[c][1] += 1
        if (i + 1) % 25 == 0:
            print(i + 1, "レース", flush=True)
    rep = {c: {"match": v[0], "n": v[1], "rate": (v[0] / v[1] if v[1] else None)} for c, v in stat.items()}
    bad = {c: r for c, r in rep.items() if r["rate"] is not None and r["rate"] < 0.99}
    print(json.dumps({"races": len(races), "rows": n_rows, "has_training_but_missing_in_serving": n_missing, "columns": len(cols), "columns_below_99pct": bad}, ensure_ascii=False, indent=1))
    (WORK / "training_laps_b/serving_parity.json").write_text(json.dumps({"races": len(races), "rows": n_rows, "missing": n_missing, "columns": rep}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
