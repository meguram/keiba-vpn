"""bundle_v5_2025（本番用）を V3Serving で読み、過去の日（2025・2026 年）で動作と確率の整合を確かめる。

KEIBA_ENV=stg python -m src.scripts.data.verify_v5_bundle_sanity --days 20250614 20261018 --max-races 6 --out /tmp/verify_v5_2025_sanity.json
確認: (1) 列がそろって予測が出る、(2) レースごとの Σp_win≈1・Σp_top2≈2・Σp_top3≈3、欠損なし、(3) bundle_v5_2024 の予測との相関（2025 年の日のみ。学習期間が違うので完全一致ではない）。
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from src.pipeline.features import serving_v4 as sv
from src.pipeline.inference import weight_triggered_predict as wtp
from src.pipeline.inference.v3_serving import V3Serving, WORK

KEY = ["race_id", "horse_id"]
V4 = WORK.parent / "training_pipeline_v4"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--max-races", type=int, default=6)
    ap.add_argument("--bundle", default=str(WORK / "final_v5/bundle_v5_2025"))
    ap.add_argument("--stage", default=str(WORK / "final_stage/v4_fixed"))
    ap.add_argument("--tl-ref", default=str(V4 / "tp35/tl_reference_v2.parquet"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from src.scraper.storage import HybridStorage

    storage = HybridStorage()
    ctx = wtp.InferenceContext()
    ctx.warm()
    raw_full = ctx.raw_hist
    v5 = V3Serving(a.bundle, a.stage, tl_reference=a.tl_ref)
    v5.warm()
    ref = pd.read_parquet(WORK / "final_eval_v5/test2025_predictions.parquet")
    ref["race_id"], ref["horse_id"] = ref["race_id"].astype(str), ref["horse_id"].astype(str)
    hist_full = v5.history.copy()
    rep: dict = {"bundle": a.bundle, "n_features": len(v5.feature_cols)}
    for day in a.days:
        d = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        flat = pl.read_parquet(sv.TABLES / day[:4] / "race_shutuba_flat.parquet", columns=["race_id", "date"]).filter(pl.col("date").cast(pl.Utf8) == d)
        ids = sorted(flat["race_id"].unique().to_list())[: a.max_races]
        if not ids:
            rep[day] = {"error": "出走表なし"}
            continue
        card = wtp.fill_card_meta(wtp.card_frame(storage, ids), [])
        cut = datetime.date.fromisoformat(d) if raw_full.schema["date"] == pl.Date else d
        ctx.raw_hist = raw_full.filter(pl.col("date") < cut)
        F, _ = ctx.build(card)
        h = hist_full[hist_full["date"].astype(str) < d]
        X = v5.features(card, F, storage, history=h)
        P = v5.predict(X)
        g = P.groupby("race_id")[["p_win", "p_top2", "p_top3"]].sum()
        res = {"rows": int(len(P)), "races": int(len(g)), "nan": int(P[["p_win", "p_top2", "p_top3"]].isna().sum().sum()),
               "sum_win": [float(g["p_win"].min()), float(g["p_win"].max())], "sum_top2": [float(g["p_top2"].min()), float(g["p_top2"].max())], "sum_top3": [float(g["p_top3"].min()), float(g["p_top3"].max())]}
        m = P.merge(ref, on=KEY, suffixes=("", "__r"))
        if len(m) > 20:
            res["corr_with_bundle_2024"] = {t: float(np.corrcoef(m[f"p_{t}"], m[f"p_{t}__r"])[0, 1]) for t in ("win", "top2", "top3")}
        rep[day] = res
        print(day, json.dumps(res, ensure_ascii=False), flush=True)
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
