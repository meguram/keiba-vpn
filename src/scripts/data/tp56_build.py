"""tp-56 の特徴量表を作る（ホストで実行）。

python -m src.scripts.data.tp56_build [--years 2020 2021 ... 2026] [--out data/local/modeling/training_pipeline_v4/tp56/features.parquet]

入力: data/page_reference/tables/<年>/race_result_flat.parquet、備考フラグ slowreal/remarks_flags.parquet（不利系 trouble_rem）。
1 走 1 行（race_id, horse_id）の as-of 特徴量（src/pipeline/features/tp56_yosou.py）。全期間を時系列順に 1 回で作る。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.pipeline.features import tp56_yosou as ty

ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "data/page_reference/tables"
OUT = ROOT / "data/local/modeling/training_pipeline_v4/tp56"
FLAGS = ROOT / "data/local/modeling/training_pipeline_v4/slowreal/remarks_flags.parquet"
COLS = ["race_id", "date", "venue", "surface", "distance", "field_size", "horse_id", "horse_number", "bracket_number",
        "finish_position", "popularity", "passing_order", "last_3f", "pace"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=list(range(2020, 2027)))
    ap.add_argument("--out", type=Path, default=OUT / "features.parquet")
    a = ap.parse_args()
    parts = [pd.read_parquet(TABLES / str(y) / "race_result_flat.parquet", columns=COLS) for y in a.years if (TABLES / str(y) / "race_result_flat.parquet").exists()]
    runs = ty.prepare_runs(pd.concat(parts, ignore_index=True))
    flags = pd.read_parquet(FLAGS, columns=["race_id", "horse_id", "trouble_rem", "has_remarks"]) if FLAGS.exists() else None
    print("runs", runs.shape, "flags", None if flags is None else flags.shape, flush=True)
    f = ty.build_features(runs, flags)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    f.to_parquet(a.out, index=False)
    print("saved", a.out, f.shape, flush=True)
    print(f.describe().T[["count", "mean", "std", "min", "max"]].round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
