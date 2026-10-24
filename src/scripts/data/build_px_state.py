"""v5 の ALL_BEST（serving_v5_px）の推論用の状態を作る: 祖先表・SVD の成分と、半年ごとのリッジ。

KEIBA_ENV=stg python -m src.scripts.data.build_px_state --static /home/jovyan/work/tp38_out/static --windows 2026-07-01
  状態の置き場: $KEIBA_INFERENCE_DIR/state/px_v5/（既定 data/local/inference/state/px_v5）。既にある静的な表は --force なしでは作り直さない。
  リッジは作り直し日 T0 より前の確定結果で学習する。新しい半年（1/1・7/1）に入る前に --windows <T0> で作っておく（無い場合は初回の予測で自動学習するが 1〜2 分かかる）。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from src.pipeline.features import serving_v5_px as sv

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", default=os.environ.get("V5_TP38_OUT", "/home/jovyan/work/tp38_out") + "/static")
    ap.add_argument("--windows", nargs="*", default=[], help="リッジを作る作り直し日（例 2026-07-01）。省略なら今日を含む半年")
    ap.add_argument("--train-table", default=str(ROOT / "data/local/modeling/training_pipeline_v4/features/train_table.parquet"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    if a.force or not (sv.STATE_DIR / "svd.npz").exists():
        info = sv.build_state(a.static, sv.STATE_DIR)
        print("静的な状態:", json.dumps(info), f"{time.time() - t0:.0f}s", flush=True)
    import pandas as pd

    from src.pipeline.features import serving_v4 as v4

    wins = a.windows or [sv.window_start(pd.Timestamp.today())]
    h = v4.V4History(None, a.train_table)
    h.warm()
    st = sv.PXState(sv.STATE_DIR)
    for w in wins:
        p = sv.fit_ridge_window(st, h.field_hist, w, sv.STATE_DIR)
        print("リッジ:", p.name, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
