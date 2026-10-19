"""tp-36 (5) 2026 年（未使用の hold-out）の v4 予測を、推論経路で『その日より前の履歴だけ』で作る。

KEIBA_INFERENCE_ENGINE=full python -m src.scripts.data.tp36_pred2026 --shard 0 --n-shards 4
bundle_v4_2025（2020〜2025 年で学習）と前段 v4_fixed を使う。履歴は『その日より前』だけ（1 日分をまとめて計算。前日事前計算と同じ運用）。
出力: data/local/modeling/betting/tp36/pred2026/<YYYYMMDD>.parquet（race_id, horse_id, p_win, p_top2, p_top3）
結果が確定している日（race_result_flat にある日）だけを対象にする。既にある日はスキップ（再開可）。
"""

from __future__ import annotations

import argparse
import datetime
import time
import traceback
from pathlib import Path

import pandas as pd
import polars as pl

from src.pipeline.inference import weight_triggered_predict as wtp
from src.pipeline.inference.v3_serving import V3Serving, WORK

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/local/modeling/betting/tp36/pred2026"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--bundle", default=str(WORK / "final_v4/bundle_v4_2025"))
    ap.add_argument("--stage", default=str(WORK / "final_stage/v4_fixed"))
    ap.add_argument("--year", default="2026")
    ap.add_argument("--max-days", type=int, default=0)
    ap.add_argument("--every", type=int, default=1, help="n 日に 1 日だけ処理する（サービング経路とバッチ特徴量のずれの点検用）")
    ap.add_argument("--out-dir", default=str(OUT))
    a = ap.parse_args()
    from src.scraper.storage import HybridStorage

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = HybridStorage()
    res = pl.read_parquet(ROOT / f"data/page_reference/tables/{a.year}/race_result_flat.parquet", columns=["race_id", "date"]).unique("race_id")
    days = sorted({str(x) for x in res["date"].to_list()})
    days = [d for i, d in enumerate(days) if i % a.n_shards == a.shard]
    days = days[:: a.every]
    if a.max_days:
        days = days[: a.max_days]
    ctx = wtp.InferenceContext()
    ctx.warm()
    assert ctx.raw_hist is not None, "KEIBA_INFERENCE_ENGINE=full で実行する"
    raw_full = ctx.raw_hist
    v4 = V3Serving(a.bundle, a.stage)
    v4.warm()
    hist_full = v4.history.copy()
    shut = pl.read_parquet(ROOT / f"data/page_reference/tables/{a.year}/race_shutuba_flat.parquet", columns=["race_id", "date"]).unique("race_id")
    for d in days:
        out = out_dir / f"{d.replace('-', '')}.parquet"
        if out.exists():
            continue
        ids = sorted(shut.filter(pl.col("date").cast(pl.Utf8) == d)["race_id"].to_list())
        ids = [r for r in ids if r in set(res.filter(pl.col("date").cast(pl.Utf8) == d)["race_id"].to_list())]
        if not ids:
            continue
        t0 = time.time()
        try:
            card = wtp.fill_card_meta(wtp.card_frame(storage, ids), [])
            cut = datetime.date.fromisoformat(d) if raw_full.schema["date"] == pl.Date else d
            ctx.raw_hist = raw_full.filter(pl.col("date") < cut)
            F, _ = ctx.build(card)
            h = hist_full[hist_full["date"].astype(str) < d]                    # その日より前の履歴だけ（前日事前計算と同じ）
            X = v4.features(card, F, storage, history=h)
            rows = [v4.predict(X)]
            if rows:
                pd.concat(rows, ignore_index=True).to_parquet(out, index=False)
            print(f"{d} races={len(ids)} {time.time() - t0:.0f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"{d} 失敗: {exc}\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
