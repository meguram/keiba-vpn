"""track_bias 先読みリーク修正（2026-10-15）後のコードで、LayerA 母表と学習表を別ディレクトリに作り直す。

既存の layer_a.parquet / train_table.parquet は動いている notebook が読むので上書きしない。出力先は
data/local/modeling/training_pipeline_fixed/（layer_a.parquet, features/train_table.parquet, features/horse_history.parquet）。

python -m src.scripts.data.rebuild_train_table_fixed            # ホストで実行（読み取りのみ。GCS へは書かない）
"""

from __future__ import annotations

import gc
import json
import shutil
import time
from pathlib import Path

import numpy as np
import polars as pl

from src.pipeline.features.horse_history.builder import build_horse_history, feature_columns
from src.pipeline.features.horse_history.common import FIT_END
from src.pipeline.features.horse_history.repair import fetch_on_time, suspicious_race_ids
from src.pipeline.features.horse_history.common import load_runs
from src.pipeline.models.layer_a_dataset import build_layer_a_dataframe
from src.pipeline.models.memory_utils import downcast_frame
from src.pipeline.models.time_split import TRAINING_YEARS

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "data/local/modeling/training_pipeline"
OUT = ROOT / "data/local/modeling/training_pipeline_fixed"
TB = ["track_bias_winner_3f_same_day_prior", "track_bias_winner_3f_prev_day", "track_bias_winner_3f_diff", "track_bias_winner_3f_weighted",
      "sire_x_track_bias_diff", "dam_sire_x_track_bias_diff", "sire_x_track_bias_weighted", "dam_sire_x_track_bias_weighted"]


def main() -> None:
    t0 = time.time()
    (OUT / "features").mkdir(parents=True, exist_ok=True)
    la_path = OUT / "layer_a.parquet"
    full = build_layer_a_dataframe(years=list(TRAINING_YEARS), jra_only=True)
    full = downcast_frame(full, keep=["race_id", "horse_id", "horse_number", "date"])
    full.to_parquet(la_path, index=False)
    print(f"[{time.time() - t0:.0f}s] 母表 {full.shape}", flush=True)
    del full
    gc.collect()

    ot_src = SRC / "features/on_time_repair.parquet"
    ot_dst = OUT / "features/on_time_repair.parquet"
    if ot_src.exists():
        shutil.copy(ot_src, ot_dst)
    runs = load_runs()
    sus = suspicious_race_ids(runs)
    on_time = fetch_on_time(sus, ot_dst)
    F, P = build_horse_history(la_path, on_time=on_time)
    fam = feature_columns(F.columns)
    fam_cols = sorted({c for v in fam.values() for c in v})
    la = pl.read_parquet(la_path).drop([c for c in ("past_races", "training") if c in pl.read_parquet_schema(la_path)])
    T = la.join(F.select(["race_id", "horse_id"] + fam_cols), on=["race_id", "horse_id"], how="left")
    assert T.height == la.height
    T = T.with_columns(pl.col(pl.Float64).cast(pl.Float32))
    T.write_parquet(OUT / "features/train_table.parquet")
    (OUT / "features/feature_families.json").write_text(json.dumps(fam, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{time.time() - t0:.0f}s] 学習表 {T.shape}", flush=True)

    # 旧表との差（track_bias 系）
    old = pl.read_parquet(SRC / "features/train_table.parquet", columns=["race_id", "horse_id"] + [c for c in TB if c in pl.read_parquet_schema(SRC / "features/train_table.parquet")])
    j = T.select(["race_id", "horse_id"] + [c for c in TB if c in T.columns]).join(old, on=["race_id", "horse_id"], how="inner", suffix="_old")
    rep = {"rows_new": T.height, "rows_old": old.height, "joined": j.height}
    for c in TB:
        if c in T.columns and f"{c}_old" in j.columns:
            a, b = j[c].to_numpy().astype("float64"), j[f"{c}_old"].to_numpy().astype("float64")
            both = np.isfinite(a) & np.isfinite(b)
            rep[c] = {"changed_share": float(np.mean(~np.isclose(a[both], b[both]))) if both.any() else None,
                      "null_new": float(np.mean(~np.isfinite(a))), "null_old": float(np.mean(~np.isfinite(b))),
                      "mean_abs_diff": float(np.mean(np.abs(a[both] - b[both]))) if both.any() else None}
    (OUT / "rebuild_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1)[:2500])


if __name__ == "__main__":
    main()
