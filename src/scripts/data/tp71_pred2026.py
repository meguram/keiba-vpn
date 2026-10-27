"""tp-71 2026 年（テストセット②）の v5 予測と特徴量を、推論経路で『その日より前の履歴だけ』で作る。

KEIBA_INFERENCE_ENGINE=full python -m src.scripts.data.tp71_pred2026 --shard 0 --n-shards 2 [--max-days 1]
bundle_v5_2025（2020〜2025 年で学習）・前段 v4_fixed・追い切り基準 tl_reference_v2 を使う（tp36_pred2026 の v5 版）。
出力: <out>/pred/<YYYYMMDD>.parquet（race_id, horse_id, p_win, p_top2, p_top3）と <out>/feat/<YYYYMMDD>.parquet（想定オッズ V8 の入力用に特徴量行列を float32 で保存）。
想定オッズの特徴量には市場オッズ・人気を含めない（tp71_odds が列名を検査する）。既にある日はスキップ（再開可）。
"""

from __future__ import annotations

import argparse
import datetime
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from src.pipeline.inference import weight_triggered_predict as wtp
from src.pipeline.inference.v3_serving import V3Serving, WORK, v5_paths

ROOT = Path(__file__).resolve().parents[3]
OUT = Path("/home/jovyan/work/tp71_out/pred2026")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--bundle", default=str(WORK / "final_v5/bundle_v5_2025"))
    ap.add_argument("--stage", default=str(WORK / "final_stage/v4_fixed"))
    ap.add_argument("--year", default="2026")
    ap.add_argument("--max-days", type=int, default=0)
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--reverse", action="store_true", help="日付を新しい順に処理する（別プロセスと真ん中で出会うまで並走させるため）")
    ap.add_argument("--v6", action="store_true", help="v6e（前段 e・RB の履歴状態つき）で算出する。bundle/stage は v6e の既定になる")
    ap.add_argument("--rb-state", default="/home/jovyan/work/readiness_out/rb_state")
    a = ap.parse_args()
    if a.v6:
        if a.bundle.endswith("bundle_v5_2025"):
            a.bundle = str(WORK / "final_v6e/bundle_v6_2025")
        if a.stage.endswith("v4_fixed"):
            a.stage = str(WORK / "final_stage/v6_e_2025")
    from src.scraper.storage import HybridStorage

    out_dir = Path(a.out_dir)
    (out_dir / "pred").mkdir(parents=True, exist_ok=True)
    (out_dir / "feat").mkdir(parents=True, exist_ok=True)
    storage = HybridStorage()
    res = pl.read_parquet(ROOT / f"data/page_reference/tables/{a.year}/race_result_flat.parquet", columns=["race_id", "date"]).unique("race_id")
    days = sorted({str(x) for x in res["date"].to_list()})
    days = [d for i, d in enumerate(days) if i % a.n_shards == a.shard]
    if a.reverse:
        days = days[::-1]
    if a.max_days:
        days = days[: a.max_days]
    ctx = wtp.InferenceContext()
    ctx.warm()
    assert ctx.raw_hist is not None, "KEIBA_INFERENCE_ENGINE=full で実行する"
    raw_full = ctx.raw_hist
    v5 = V3Serving(a.bundle, a.stage, tl_reference=v5_paths()["tl_reference"])
    if a.v6:
        v5.rb_state = Path(a.rb_state)
    v5.warm()
    hist_full = v5.history.copy()
    shut = pl.read_parquet(ROOT / f"data/page_reference/tables/{a.year}/race_shutuba_flat.parquet", columns=["race_id", "date"]).unique("race_id")
    for d in days:
        key = d.replace("-", "")
        out = out_dir / "pred" / f"{key}.parquet"
        if out.exists():
            continue
        day_ids = set(res.filter(pl.col("date").cast(pl.Utf8) == d)["race_id"].to_list())
        ids = sorted(r for r in shut.filter(pl.col("date").cast(pl.Utf8) == d)["race_id"].to_list() if r in day_ids)
        if not ids:
            continue
        t0 = time.time()
        try:
            card = wtp.fill_card_meta(wtp.card_frame(storage, ids), [])
            cut = datetime.date.fromisoformat(d) if raw_full.schema["date"] == pl.Date else d
            ctx.raw_hist = raw_full.filter(pl.col("date") < cut)
            F, _ = ctx.build(card)
            h = hist_full[hist_full["date"].astype(str) < d]
            X = v5.features(card, F, storage, history=h)
            P = v5.predict(X)
            Xf = X.reset_index(drop=True).copy()
            for c in Xf.columns:
                if c in ("race_id", "horse_id"):
                    continue
                Xf[c] = Xf[c].astype("float32") if pd.api.types.is_numeric_dtype(Xf[c]) and not pd.api.types.is_bool_dtype(Xf[c]) else Xf[c].astype(str)
            Xf.to_parquet(out_dir / "feat" / f"{key}.parquet", index=False)
            P.to_parquet(out, index=False)
            print(f"{d} races={len(ids)} rows={len(P)} {time.time() - t0:.0f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"{d} 失敗: {exc}\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
