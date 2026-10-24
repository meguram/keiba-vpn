"""V3Serving（v5 バンドル）の通しの経路が作る確率を、最終学習の学習側の予測（final_eval_v5/test2025_predictions.parquet）と test 2025 の日で比べる。

KEIBA_ENV=stg KEIBA_V3_AUX=<final_aux> python -m src.scripts.data.verify_v5_serving_e2e --days 20250614 --max-races 6 --out /tmp/verify_v5_e2e.json
V0 の列は推論経路で作る（学習表の値を直接渡さない）ので、v4 の同じ検証（相関 0.9987）と同じく、V0 の再現誤差を含む。バンドルは bundle_v5_2024（test 2025 を含まない）。
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
ROOT = Path(__file__).resolve().parents[3]
V4 = ROOT / "data/local/modeling/training_pipeline_v4"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--max-races", type=int, default=6)
    ap.add_argument("--bundle", default=str(WORK / "final_v5/bundle_v5_2024"))
    ap.add_argument("--stage", default=str(WORK / "final_stage/v4_verify_2024"))
    ap.add_argument("--tl-ref", default=str(V4 / "tp35/tl_reference_v2.parquet"))
    ap.add_argument("--ref", default=str(WORK / "final_eval_v5/test2025_predictions.parquet"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from src.scraper.storage import HybridStorage

    storage = HybridStorage()
    ctx = wtp.InferenceContext()
    ctx.warm()
    raw_full = ctx.raw_hist
    v3 = V3Serving(a.bundle, a.stage, tl_reference=a.tl_ref)
    v3.warm()
    ref = pd.read_parquet(a.ref)
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
        P = v3.predict(X)
        sec = round(time.time() - t0, 1)
        m = P.merge(ref, on=KEY, suffixes=("", "__r"))
        res = {"rows": int(len(m)), "sec": sec}
        for t in ("win", "top2", "top3"):
            x, y = m[f"p_{t}"].to_numpy(dtype=float), m[f"p_{t}__r"].to_numpy(dtype=float)
            yy = (m["finish_position"].to_numpy() <= {"win": 1, "top2": 2, "top3": 3}[t]).astype(float)
            ll = lambda p: float(-np.mean(yy * np.log(np.clip(p, 1e-9, 1)) + (1 - yy) * np.log(np.clip(1 - p, 1e-9, 1))))
            res[t] = {"corr": float(np.corrcoef(x, y)[0, 1]), "ll_serving": ll(x), "ll_batch": ll(y), "diff": ll(x) - ll(y)}
        rep[day] = res
        print(day, json.dumps(res, ensure_ascii=False), flush=True)
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
