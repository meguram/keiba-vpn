"""導入可否の確認用: 2026 年の抜き取りの日を v6 バンドルで（前段 e・RB の履歴状態つき）算出し、確率と前段・補助ヘッドの出力を保存する（readiness-audit）。

  KEIBA_INFERENCE_ENGINE=full python -m src.scripts.data.readiness_v6_sample --days 20260124,20260214 --max-races 10 --out <dir> \
      --bundle <final_v6e/bundle_v6_2025> --stage <final_stage/v6_e_2025> --rb-state <state/rb_v6>

backfill_predictions と同じ手順（その日より前の履歴だけ＋同日で先に発走したレースの結果）で特徴量を作る。違いは、確率 P に加えて
前段（s_time_pred / s_time_gap / s_rank_pred）と補助ヘッド（mtl_*）の列を ``<out>/stage/<日>/<race_id>.parquet`` に保存する点。
v5 の確率は ``<out>/v5/<日>/<race_id>.parquet``（backfill と同じ列）。保存済みの v4 の確率は変更しない。
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import polars as pl

logger = logging.getLogger("scripts.readiness_v6_sample")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", required=True, help="カンマ区切り YYYYMMDD")
    ap.add_argument("--max-races", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bundle", default="")
    ap.add_argument("--stage", default="")
    ap.add_argument("--rb-state", default="", help="RB の履歴状態のフォルダ（build_rb_state build の出力）")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()
    from src.pipeline.inference import weight_triggered_predict as wtp
    from src.pipeline.inference.v3_serving import V3Serving, default_bundle_dir, default_stage_dir
    from src.scraper.storage import HybridStorage
    from src.scripts.data.backfill_predictions import race_days

    bundle = Path(a.bundle) if a.bundle else default_bundle_dir()
    stage = Path(a.stage) if a.stage else default_stage_dir(bundle)
    tables = Path(os.environ.get("KEIBA_TABLES_DIR") or "data/page_reference/tables")
    storage = HybridStorage()
    out = Path(a.out)
    days_req = [d.strip() for d in a.days.split(",") if d.strip()]
    days = race_days([2026], tables, min(days_req), max(days_req))
    days = {d: ids for d, ids in days.items() if d in days_req}

    ctx = wtp.InferenceContext()
    ctx.warm()
    assert ctx.raw_hist is not None, "KEIBA_INFERENCE_ENGINE=full で実行する"
    raw_full = ctx.raw_hist
    v3 = V3Serving(bundle, stage)
    v3.rb_state = Path(a.rb_state) if a.rb_state else None
    v3.warm()
    hist_full = v3.history.copy()
    for day, ids in days.items():
        t0 = time.time()
        ids = ids[: a.max_races]
        d = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        try:
            card = wtp.fill_card_meta(wtp.card_frame(storage, ids), [])
            ok_ids = (card.filter(pl.col("horse_number").fill_null(0) > 0)
                      .filter(pl.col("distance").cast(pl.Float64, strict=False) > 0)
                      .filter(pl.col("surface").cast(pl.Utf8).fill_null("") != "")["race_id"].unique().to_list())
            card = card.filter(pl.col("race_id").is_in(ok_ids))
            if card.height == 0:
                logger.warning("%s: 出馬表が揃わない", day)
                continue
            cut = datetime.date.fromisoformat(d) if raw_full.schema["date"] == pl.Date else d
            ctx.raw_hist = raw_full.filter(pl.col("date") < cut)
            F, _info = ctx.build(card)
            (out / "v6" / day).mkdir(parents=True, exist_ok=True)
            (out / "stage" / day).mkdir(parents=True, exist_ok=True)
            n_ok = 0
            for rid in sorted(ok_ids):
                cr = card.filter(pl.col("race_id") == rid)
                st = str(cr["start_time"][0])
                h = hist_full[(hist_full["date"].astype(str) < d)
                              | ((hist_full["date"].astype(str) == d) & (hist_full["start_time"].astype(str) < st) & (hist_full["race_id"].astype(str) != rid))]
                t1 = time.time()
                X = v3.features(cr, F.filter(pl.col("race_id") == rid), storage, history=h)
                P = v3.predict(X)
                P.to_parquet(out / "v6" / day / f"{rid}.parquet", index=False)
                keep = [c for c in X.columns if c.startswith(("s_time", "s_rank", "mtl_"))]
                key = X[[c for c in ("race_id", "horse_id") if c in X.columns]].reset_index(drop=True)
                pd.concat([key, X[keep].reset_index(drop=True)], axis=1).to_parquet(out / "stage" / day / f"{rid}.parquet", index=False)
                n_ok += 1
                logger.info("%s %s: %.1f 秒（特徴量＋推論）", day, rid, time.time() - t1)
            logger.info("%s: %d/%d レース %.0f 秒", day, n_ok, len(ids), time.time() - t0)
        except Exception:  # noqa: BLE001
            logger.exception("%s: 失敗（次の日へ）", day)
    (out / "done.json").write_text(json.dumps({"days": list(days), "finished": datetime.datetime.now().isoformat()}), encoding="utf-8")


if __name__ == "__main__":
    main()
