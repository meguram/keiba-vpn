"""推論経路（v3_serving）で作った予測を、tp-06c の予測（学習表の特徴量から）と比べる。

KEIBA_INFERENCE_ENGINE=full python -m src.scripts.data.verify_v3_serving --days 20250302 20250614 --out /tmp/serve_v3_parity.json
  履歴は「その日より前」+「同日で発走時刻が前のレース」だけ。バンドルは 2020〜2024 で学習した bundle_2024、前段は verify_2024 を使う。
  比べる指標: 予測の差（平均絶対差・相関）、勝率 LogLoss（推論経路 vs tp-06c）、特徴量の欠損率。
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

from src.pipeline.inference import weight_triggered_predict as wtp
from src.pipeline.inference.v3_serving import V3Serving, WORK

TRAIN = WORK.parent / "training_pipeline_fixed/features/train_table.parquet"


def _ll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--bundle", default=str(WORK / "final_v3/bundle_2024"))
    ap.add_argument("--stage", default=str(WORK / "final_stage/verify_2024"))
    ap.add_argument("--max-races", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from src.scraper.storage import HybridStorage

    storage = HybridStorage()
    ctx = wtp.InferenceContext()
    ctx.warm()
    assert ctx.raw_hist is not None, "KEIBA_INFERENCE_ENGINE=full で実行する"
    raw_full = ctx.raw_hist
    v3 = V3Serving(a.bundle, a.stage)
    v3.warm()
    hist_full = v3.history.copy()
    ref = pd.read_parquet(WORK / "final_eval_v3/test2025_predictions.parquet")[["race_id", "horse_id", "p_win", "p_top2", "p_top3"]]
    ref["race_id"], ref["horse_id"] = ref["race_id"].astype(str), ref["horse_id"].astype(str)
    fin = pl.read_parquet(TRAIN, columns=["race_id", "horse_id", "finish_position"]).to_pandas()
    fin["race_id"], fin["horse_id"] = fin["race_id"].astype(str), fin["horse_id"].astype(str)

    rows, report = [], {"days": {}}
    for day in a.days:
        d = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        ids = sorted(ref.merge(fin, on=["race_id", "horse_id"])["race_id"].loc[lambda s: s.str[:4] == day[:4]].unique())
        flat = pl.read_parquet(Path("data/page_reference/tables") / day[:4] / "race_shutuba_flat.parquet", columns=["race_id", "date"]).filter(pl.col("date").cast(pl.Utf8) == d)
        ids = [r for r in sorted(flat["race_id"].unique().to_list()) if r in set(ids)]
        if a.max_races:
            ids = ids[: a.max_races]
        card = wtp.fill_card_meta(wtp.card_frame(storage, ids), [])
        cut = datetime.date.fromisoformat(d) if raw_full.schema["date"] == pl.Date else d
        ctx.raw_hist = raw_full.filter(pl.col("date") < cut)
        t0 = time.time()
        F, _info = ctx.build(card)
        for rid in ids:
            cr = card.filter(pl.col("race_id") == rid)
            st = str(cr["start_time"][0])
            h = hist_full[(hist_full["date"].astype(str) < d) | ((hist_full["date"].astype(str) == d) & (hist_full["start_time"].astype(str) < st) & (hist_full["race_id"].astype(str) != rid))]
            X = v3.features(cr, F.filter(pl.col("race_id") == rid), storage, history=h)
            P = v3.predict(X)
            nan_rate = float(X.drop(columns=["race_id", "horse_id"]).isna().mean().mean())
            P["nan_rate"] = nan_rate
            rows.append(P)
        report["days"][day] = {"races": len(ids), "sec": round(time.time() - t0, 1)}
    got = pd.concat(rows, ignore_index=True)
    m = got.merge(ref, on=["race_id", "horse_id"], suffixes=("", "_ref")).merge(fin, on=["race_id", "horse_id"])
    y = (m["finish_position"] == 1).to_numpy().astype(float)
    for t in ("win", "top2", "top3"):
        x, r = m[f"p_{t}"].to_numpy(), m[f"p_{t}_ref"].to_numpy()
        report[t] = {"rows": int(len(m)), "mean_abs_diff": float(np.mean(np.abs(x - r))), "max_abs_diff": float(np.max(np.abs(x - r))), "corr": float(np.corrcoef(x, r)[0, 1])}
    report["win_logloss"] = {"serving": _ll(m["p_win"].to_numpy(), y), "tp06c": _ll(m["p_win_ref"].to_numpy(), y)}
    report["mean_feature_nan_rate"] = float(got["nan_rate"].mean())
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
