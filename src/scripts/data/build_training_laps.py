"""GCS の horse_training（馬別の調教タイム）を読み、追い切りの 1 行 1 本の表にする（読み取りのみ。GCS へは書かない）。

python -m src.scripts.data.build_training_laps --out data/local/modeling/training_pipeline/features/training_laps.parquet
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from src.pipeline.features.training_laps import add_lap_features, record_to_rows
from src.scraper.storage import HybridStorage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-table", default="data/local/modeling/training_pipeline/features/train_table.parquet")
    ap.add_argument("--out", default="data/local/modeling/training_pipeline/features/training_laps.parquet")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    ids = sorted(pd.read_parquet(a.train_table, columns=["horse_id"])["horse_id"].astype(str).unique())
    if a.limit:
        ids = ids[: a.limit]
    st = HybridStorage(".")
    t0 = time.time()
    miss = 0
    rows: list[dict] = []

    def one(h: str):
        try:
            return h, st.load("horse_training", h)
        except Exception:
            return h, None

    with ThreadPoolExecutor(a.workers) as ex:
        for i, (h, rec) in enumerate(ex.map(one, ids), 1):
            r = record_to_rows(rec)
            if not r:
                miss += 1
            rows += r
            if i % 2000 == 0:
                print(f"{i}/{len(ids)} 本数 {len(rows)} 空 {miss} ({time.time() - t0:.0f}s)", flush=True)
    df = add_lap_features(pd.DataFrame(rows))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(f"馬 {len(ids)} / 調教なし {miss} / 追い切り {len(df)} 本 -> {a.out}")


if __name__ == "__main__":
    main()
