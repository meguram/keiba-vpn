"""血統キー（serving_parents）の修正前後で、同じレース・同じバンドルの予測がどれだけ変わるかを測る（読み取りのみ）。

KEIBA_INFERENCE_ENGINE=full python -m src.scripts.data.ped_fix_check --days 20250202,20250614 --max-races 6 --out <json> \
    --bundles <bundle_v4_2024>:<stage_v4_verify_2024>,<bundle_v6_2024>:<stage_v6_e_2024> [--rb-state <dir>]
各バンドルを KEIBA_PED_PARENT_KEYS=0（修正前＝血統名を空）と 1（修正後）で動かし、勝率・連対・複勝の相関・平均絶対差・LogLoss の差（確定結果に対する。診断用で選択には使わない）を出す。
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

KEY = ["race_id", "horse_id"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", required=True)
    ap.add_argument("--max-races", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bundles", required=True, help="バンドル:前段 をカンマ区切り")
    ap.add_argument("--rb-state", default="")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()
    from src.pipeline.inference import weight_triggered_predict as wtp
    from src.pipeline.inference.v3_serving import V3Serving
    from src.scraper.storage import HybridStorage
    from src.scripts.data.backfill_predictions import race_days

    tables = Path(os.environ.get("KEIBA_TABLES_DIR") or "data/page_reference/tables")
    storage = HybridStorage()
    days_req = [d.strip() for d in a.days.split(",") if d.strip()]
    days = race_days([2025], tables, min(days_req), max(days_req))
    days = {d: ids for d, ids in days.items() if d in days_req}
    ctx = wtp.InferenceContext()
    ctx.warm()
    raw_full = ctx.raw_hist
    work: list[tuple] = []                         # (day, rid, card_r, F_r)
    for day, ids in days.items():
        d = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        card = wtp.fill_card_meta(wtp.card_frame(storage, ids[: a.max_races]), [])
        ok_ids = (card.filter(pl.col("horse_number").fill_null(0) > 0).filter(pl.col("distance").cast(pl.Float64, strict=False) > 0)
                  .filter(pl.col("surface").cast(pl.Utf8).fill_null("") != "")["race_id"].unique().to_list())
        card = card.filter(pl.col("race_id").is_in(ok_ids))
        cut = datetime.date.fromisoformat(d) if raw_full.schema["date"] == pl.Date else d
        ctx.raw_hist = raw_full.filter(pl.col("date") < cut)
        F, _ = ctx.build(card)
        for rid in sorted(ok_ids):
            work.append((d, rid, card.filter(pl.col("race_id") == rid), F.filter(pl.col("race_id") == rid)))
    res = pd.read_parquet(tables / "2025/race_result_flat.parquet", columns=["race_id", "horse_id", "finish_position"])
    res["race_id"], res["horse_id"] = res["race_id"].astype(str), res["horse_id"].astype(str)
    out: dict = {}
    for spec in a.bundles.split(","):
        bdir, sdir = spec.split(":")
        preds = {}
        for flag in ("0", "1"):
            os.environ["KEIBA_PED_PARENT_KEYS"] = flag
            v3 = V3Serving(Path(bdir), Path(sdir))
            if a.rb_state:
                v3.rb_state = Path(a.rb_state)
            v3.warm()
            hist_full = v3.history.copy()
            ps = []
            for d, rid, cr, Fr in work:
                st = str(cr["start_time"][0])
                h = hist_full[(hist_full["date"].astype(str) < d) | ((hist_full["date"].astype(str) == d) & (hist_full["start_time"].astype(str) < st) & (hist_full["race_id"].astype(str) != rid))]
                ps.append(v3.predict(v3.features(cr, Fr, storage, history=h)))
            P = pd.concat(ps, ignore_index=True)
            P["race_id"], P["horse_id"] = P["race_id"].astype(str), P["horse_id"].astype(str)
            preds[flag] = P
            del v3
        m = preds["0"].merge(preds["1"], on=KEY, suffixes=("_0", "_1")).merge(res, on=KEY)
        fp = pd.to_numeric(m["finish_position"], errors="coerce")
        m = m[fp > 0]
        fp = fp[fp > 0]
        r = {"rows": int(len(m)), "races": int(m["race_id"].nunique())}
        for t, k in (("win", 1), ("top2", 2), ("top3", 3)):
            y = (fp.to_numpy() <= k).astype(float)
            x0, x1 = m[f"p_{t}_0"].to_numpy(float), m[f"p_{t}_1"].to_numpy(float)
            ll = lambda p: float(-np.mean(y * np.log(np.clip(p, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1))))  # noqa: E731
            r[t] = {"corr": float(np.corrcoef(x0, x1)[0, 1]), "mad_p": float(np.mean(np.abs(x0 - x1))), "LL_before": ll(x0), "LL_after": ll(x1), "delta_LL": ll(x1) - ll(x0)}
        out[Path(bdir).name] = r
        print(Path(bdir).name, json.dumps(r, ensure_ascii=False), flush=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
