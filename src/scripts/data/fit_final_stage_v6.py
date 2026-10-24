"""v6 の前段 S3B（s_time_pred / s_time_gap / s_rank_pred。入力 = base 175 列 + tl_ 再取得 + CY + 追走難度 v2 + ラップ予想 + 位置取り予想）の最終モデルを学習して保存する。

python -m src.scripts.data.fit_final_stage_v6 --fit-end 2024      # 2024-12-22 より前の行で学習 → final_stage/v6_b_2024（bundle_v6_2024 と対。2025 年の行を s3x_b の OOF と比べる）
python -m src.scripts.data.fit_final_stage_v6 --fit-end 2025      # 2020〜2025 の全行で学習 → final_stage/v6_b_2025（2026 年は使わない）
学習の設定は StageModels.fit（= stage_models.oof_regression と同じ）。入力の追走難度 v2・ラップ予想は学習側では leave-one-year-out の予測、
推論側は本番の保存値を同じ列名に写して渡す（serving_v6.stage_input_columns が対応表）。出力: data/local/modeling/training_pipeline/final_stage/（git 管理外）
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features.serving_stage_rel import STAGE_COLS, STAGE_TARGETS, StageModels
from src.pipeline.models import stage_models as sm
from src.pipeline.models.compute import detect_compute
from src.scripts.data import tp07_base as tb
from src.scripts.data import v6_stage as vs

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "data/local/modeling/training_pipeline"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-end", type=int, choices=[2024, 2025], required=True)
    ap.add_argument("--variant", default="b")
    a = ap.parse_args()
    t0 = time.time()
    base = tb._base_cols()
    df = tb._load(vs.V4)
    labels = pd.read_parquet(tb.OLD / "oof/stage_labels.parquet")
    import json as _j

    fit_params = _j.loads((tb.OLD / "features/fit_params.json").read_text(encoding="utf-8"))
    T = sm.build_targets(df, labels, sm.pace_class_labels(labels, fit_params["pace_q"], fit_params["rpci_q"]))
    X = vs.make_inputs(a.variant, df, T, base)
    assert int(df["year"].max()) >= 2025
    tr = (df["date"] < pd.Timestamp("2024-12-22")).to_numpy() if a.fit_end == 2024 else (df["year"].to_numpy() <= 2025)
    threads = int(os.environ.get("V6_THREADS") or detect_compute(reserve_cores=6).threads)
    targets = {name: T[col].to_numpy()[tr] for name, col in STAGE_TARGETS.items()}
    print(f"学習 {int(tr.sum()):,} 行 / 列 {X.shape[1]} / threads {threads} / 変種 {a.variant}", flush=True)
    sms = StageModels.fit(X.loc[tr], targets, threads)
    out = OLD / "final_stage" / f"v6_{a.variant}_{a.fit_end}"
    sms.save(out, extra={"variant": a.variant, "fit_end": a.fit_end, "train_rows": int(tr.sum()), "seconds": round(time.time() - t0)})
    print(f"[{time.time() - t0:.0f}s] 保存: {out}", flush=True)
    s3x = pd.read_parquet(vs.OUT / f"s3x_{a.variant}.parquet")
    va = np.flatnonzero(df["year"].to_numpy() == 2025)
    pred = sms.predict(X.iloc[va], df["race_id"].to_numpy()[va])
    m = pd.concat([df[["race_id", "horse_id"]].iloc[va].reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
    for c in ("race_id", "horse_id"):
        m[c], s3x[c] = m[c].astype(str), s3x[c].astype(str)
    m = m.merge(s3x, on=["race_id", "horse_id"], suffixes=("", "_oof"))
    rep = {"fit_end": a.fit_end, "rows_2025": len(m)}
    for c in STAGE_COLS:
        x, y = m[c].to_numpy(float), m[c + "_oof"].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        rep[c] = {"corr_with_oof": float(np.corrcoef(x[ok], y[ok])[0, 1]), "mean_diff": float(np.mean(x[ok] - y[ok])), "max_abs_diff": float(np.max(np.abs(x[ok] - y[ok])))}
    (out / "parity_vs_oof_2025.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
