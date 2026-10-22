"""v6 の推論経路（V3Serving.features）が作る列を、学習側の行列（fit_final_v6 と同じ build_matrices）と同じ行で突き合わせる（監査用・読み取りのみ）。

KEIBA_INFERENCE_ENGINE=full V6_STAGE=e V6_TAG=e V5_GROUPS=CY,BEST,RB python -m src.scripts.data.v6_parity --days 20250202,20250614 --max-races 6 --out <json> \
    --bundle <final_v6e/bundle_v6_2024> --stage <final_stage/v6_e_2024> --rb-state <rb_state>
列ごとに、欠損の位置の不一致率と、両方に値がある行での不一致率（|差| > 1e-3×(1+|学習側|)）を出す。前段 e の入力（追走難度 v2・ラップ予想・位置取り予想）は
2025 年の保存値が無いため、その由来の列（s_*, EASE2/lp/pf, それらの相対化）は比べても意味がなく別枠で出す。
"""

from __future__ import annotations

import os

if os.environ.get("V6_STAGE"):
    from src.scripts.data import v6_work

    v6_work.use(os.environ["V6_STAGE"])

import argparse  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

KEY = ["race_id", "horse_id"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", required=True)
    ap.add_argument("--max-races", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--rb-state", required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()
    from src.pipeline.inference import weight_triggered_predict as wtp
    from src.pipeline.inference.v3_serving import V3Serving
    from src.scraper.storage import HybridStorage
    from src.scripts.data import fit_final_v4 as f4
    from src.scripts.data import fit_final_v6 as f6
    from src.scripts.data.backfill_predictions import race_days

    f6.patch()
    df, meta, mats, A, _ = f4.build_matrices()
    names = [m[0] for m in f4.MEMBERS]
    X_all = pd.concat([mats[n][2] for n in names], axis=1)
    X_all = X_all.loc[:, ~X_all.columns.duplicated()]
    X_all.insert(0, "horse_id", meta["horse_id"].to_numpy())
    X_all.insert(0, "race_id", meta["race_id"].astype(str).to_numpy())

    tables = Path(os.environ.get("KEIBA_TABLES_DIR") or "data/page_reference/tables")
    storage = HybridStorage()
    days_req = [d.strip() for d in a.days.split(",") if d.strip()]
    days = race_days([2025], tables, min(days_req), max(days_req))
    days = {d: ids for d, ids in days.items() if d in days_req}
    ctx = wtp.InferenceContext()
    ctx.warm()
    raw_full = ctx.raw_hist
    v3 = V3Serving(Path(a.bundle), Path(a.stage))
    v3.rb_state = Path(a.rb_state)
    v3.warm()
    hist_full = v3.history.copy()
    parts = []
    for day, ids in days.items():
        ids = ids[: a.max_races]
        d = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        card = wtp.fill_card_meta(wtp.card_frame(storage, ids), [])
        ok_ids = (card.filter(pl.col("horse_number").fill_null(0) > 0).filter(pl.col("distance").cast(pl.Float64, strict=False) > 0)
                  .filter(pl.col("surface").cast(pl.Utf8).fill_null("") != "")["race_id"].unique().to_list())
        card = card.filter(pl.col("race_id").is_in(ok_ids))
        cut = datetime.date.fromisoformat(d) if raw_full.schema["date"] == pl.Date else d
        ctx.raw_hist = raw_full.filter(pl.col("date") < cut)
        F, _ = ctx.build(card)
        for rid in sorted(ok_ids):
            cr = card.filter(pl.col("race_id") == rid)
            st = str(cr["start_time"][0])
            h = hist_full[(hist_full["date"].astype(str) < d) | ((hist_full["date"].astype(str) == d) & (hist_full["start_time"].astype(str) < st) & (hist_full["race_id"].astype(str) != rid))]
            X = v3.features(cr, F.filter(pl.col("race_id") == rid), storage, history=h)
            parts.append(X)
    S = pd.concat(parts, ignore_index=True)
    S["race_id"], S["horse_id"] = S["race_id"].astype(str), S["horse_id"].astype(str)
    X_all["horse_id"] = X_all["horse_id"].astype(str)
    m = S.merge(X_all, on=KEY, suffixes=("_s", "_t"))
    stage_like = lambda c: c.startswith(("s_", "EASE2_", "lp_", "pf_", "z_hat", "td_")) or "_s_" in c or c.split("_r")[0].startswith("s_")  # noqa: E731
    rows = []
    for c in [x for x in S.columns if x not in KEY]:
        if f"{c}_t" not in m.columns:
            rows.append({"col": c, "note": "学習側に無い"})
            continue
        s, t = pd.to_numeric(m[f"{c}_s"], errors="coerce").to_numpy(float), pd.to_numeric(m[f"{c}_t"], errors="coerce").to_numpy(float)
        both = ~np.isnan(s) & ~np.isnan(t)
        rows.append({"col": c, "n": int(len(m)), "nan_mismatch": float((np.isnan(s) != np.isnan(t)).mean()),
                     "val_mismatch": float((np.abs(s - t)[both] > 1e-3 * (1 + np.abs(t[both]))).mean()) if both.any() else 0.0,
                     "corr": float(np.corrcoef(s[both], t[both])[0, 1]) if both.sum() > 5 and np.std(s[both]) > 0 and np.std(t[both]) > 0 else None,
                     "mad_over_sd": float(np.mean(np.abs(s - t)[both]) / (np.std(t[both]) + 1e-12)) if both.any() else 0.0,
                     "stage_like": bool(stage_like(c))})
    rep = pd.DataFrame(rows)
    big = rep[(rep["mad_over_sd"] > 0.2) | (rep["nan_mismatch"] > 0.05)]
    out = {"n_big_diff": int(len(big)), "big_cols": big["col"].tolist(), "rows": int(len(m)), "races": int(m["race_id"].nunique()), "columns": int(len(rep)),
           "worst": rep.sort_values(["mad_over_sd", "nan_mismatch"], ascending=False).head(80).to_dict("records"),
           "all": {r["col"]: {k: r.get(k) for k in ("nan_mismatch", "val_mismatch", "corr", "mad_over_sd")} for r in rows if "n" in r},
           "n_bad_nan_gt5pct": int((rep["nan_mismatch"] > 0.05).sum()), "n_bad_val_gt5pct": int((rep["val_mismatch"] > 0.05).sum()),
           "missing_in_train": rep[rep.get("note").notna()]["col"].tolist() if "note" in rep else []}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "worst"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
