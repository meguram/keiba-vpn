"""V3Serving に V4 の群を組み込んだ通しの経路が作る列を、学習時の V4 特徴量（tp-27）と比べる（バンドルの再学習を待たずに確認する）。

KEIBA_INFERENCE_ENGINE=full python -m src.scripts.data.verify_v4_serving_e2e --days 20250614 --max-races 6 --out /tmp/serve_v4_e2e.json
  V0 の列は推論経路で作る（学習表の値を直接渡さない）ので、field7 は推論経路の V0 の再現誤差を含む。
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from src.pipeline.features import serving_v4 as sv
from src.pipeline.inference import weight_triggered_predict as wtp
from src.pipeline.inference.v3_serving import V3Serving, WORK

KEY = ["race_id", "horse_id"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--max-races", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from src.scraper.storage import HybridStorage

    storage = HybridStorage()
    ctx = wtp.InferenceContext()
    ctx.warm()
    raw_full = ctx.raw_hist
    v3 = V3Serving(WORK / "final_v3/bundle_2024", WORK / "final_stage/verify_2024")
    v3.warm()
    fixed = WORK.parent / "training_pipeline_fixed/features/train_table.parquet"
    v3.v4 = sv.V4Features(None, fixed)
    v3.v4.warm()
    cols = sv.group_columns()
    v3.feature_cols = list(dict.fromkeys(v3.feature_cols + [c for g in cols.values() for c in g]))
    ref = pd.read_parquet(WORK / "tp27/v4_features.parquet")
    ref["race_id"], ref["horse_id"] = ref["race_id"].astype(str), ref["horse_id"].astype(str)
    hist_full = v3.history.copy()
    rep: dict = {}
    for day in a.days:
        d = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        flat = pl.read_parquet(sv.TABLES / day[:4] / "race_shutuba_flat.parquet", columns=["race_id", "date"]).filter(pl.col("date").cast(pl.Utf8) == d)
        ids = sorted(flat["race_id"].unique().to_list())[: a.max_races]
        card = wtp.fill_card_meta(wtp.card_frame(storage, ids), [])
        cut = datetime.date.fromisoformat(d) if raw_full.schema["date"] == pl.Date else d
        ctx.raw_hist = raw_full.filter(pl.col("date") < cut)
        F, _ = ctx.build(card)
        t0 = time.time()
        h = hist_full[hist_full["date"].astype(str) < d]
        X = v3.features(card, F, storage, history=h)
        sec = round(time.time() - t0, 1)
        m = X.merge(ref, on=KEY, suffixes=("", "__r"))
        res = {"rows": int(len(m)), "sec": sec}
        for g, cs in cols.items():
            bad = {}
            for c in cs:
                if c not in X.columns or c + "__r" not in m.columns:
                    continue
                x, y = m[c].to_numpy(dtype=float), m[c + "__r"].to_numpy(dtype=float)
                ok = (np.isnan(x) & np.isnan(y)) | np.isclose(x, y, rtol=1e-3, atol=1e-4)
                if not ok.all():
                    bad[c] = float(1 - ok.mean())
            res[g] = {"cols": len(cs), "bad": len(bad), "worst": dict(sorted(bad.items(), key=lambda kv: -kv[1])[:4]), "bad_names": sorted(bad)}
        rep[day] = res
        print(day, json.dumps({k: (v if not isinstance(v, dict) else (v["bad"], v["cols"])) for k, v in res.items()}), flush=True)
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
