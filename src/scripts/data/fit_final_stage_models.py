"""前段モデル（P5 走破タイム差・P6 正規化着順。s_time_pred / s_time_gap / s_rank_pred）の最終モデルを学習して保存する（tp-07 の一部）。

python -m src.scripts.data.fit_final_stage_models --table fixed                       # 修正後の学習表（training_pipeline_fixed）で 2020〜2025 の全行
python -m src.scripts.data.fit_final_stage_models --table old                         # 旧表（v3 の学習に使った表）
python -m src.scripts.data.fit_final_stage_models --table old --train-before 2024-12-22 --out verify_2024   # tp-04 の 2025 年 OOF と照合する検証用
python -m src.scripts.data.fit_final_stage_models --table v4 --out v4_fixed                                    # v4 の新土台（training_pipeline_v4）で 2020〜2025 の全行
python -m src.scripts.data.fit_final_stage_models --table v4 --train-before 2024-12-22 --out v4_verify_2024   # v4 の検証用（bundle_v4_2024 と対）

tp-04 の採用は P5・P6 とも「前段なし」= 入力は tp-03 の base 175 列のみ（stage_report.json）。学習の設定は ``stage_models.oof_regression`` と同じ
（REG_PARAMS・300 ラウンド・seed 0）。出力: data/local/modeling/training_pipeline/final_stage/<名前>/（git 管理外。MLflow へは登録しない）。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features.serving_stage_rel import STAGE_COLS, STAGE_TARGETS, StageModels
from src.pipeline.models import ablation as ab
from src.pipeline.models import stage_models as sm
from src.pipeline.models.compute import detect_compute

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "data/local/modeling/training_pipeline"
TABLES = {"old": WORK, "fixed": ROOT / "data/local/modeling/training_pipeline_fixed", "v4": ROOT / "data/local/modeling/training_pipeline_v4"}


def load_base_and_targets(table: str) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    adopted = json.loads((WORK / "ablation_nospeed/adopted_features.json").read_text(encoding="utf-8"))
    base = list(adopted["b0_features"]) + list(adopted["features"])
    df = ab.load_table(TABLES[table] / "features/train_table.parquet")
    df["race_id"] = df["race_id"].astype(str)
    df["horse_id"] = df["horse_id"].astype(str)
    lp = TABLES[table] / "oof/stage_labels.parquet"
    labels = pd.read_parquet(lp if lp.exists() else WORK / "oof/stage_labels.parquet")           # 結果由来のラベル（学習専用。表のリーク修正とは無関係）
    fit_params = json.loads((WORK / "features/fit_params.json").read_text(encoding="utf-8"))
    race_cls = sm.pace_class_labels(labels, fit_params["pace_q"], fit_params["rpci_q"])
    T = sm.build_targets(df, labels, race_cls)
    return df, base, T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", choices=list(TABLES), default="fixed")
    ap.add_argument("--train-before", default=None, help="この日付より前の行だけで学習（検証用。省略時は全行）")
    ap.add_argument("--out", default=None, help="final_stage/ の下の名前（省略時は --table）")
    ap.add_argument("--compare", default=None, help="比較する別の保存先（final_stage/ の下の名前）。s_* の相関を出す")
    a = ap.parse_args()
    t0 = time.time()
    df, base, T = load_base_and_targets(a.table)
    assert int(df["year"].max()) <= 2025, "2026 年が混ざっています"
    tr = np.ones(len(df), bool) if a.train_before is None else (df["date"] < pd.Timestamp(a.train_before)).to_numpy()
    threads = detect_compute(reserve_cores=6).threads
    X = df[base]
    targets = {name: T[col].to_numpy()[tr] for name, col in STAGE_TARGETS.items()}
    print(f"学習 {int(tr.sum()):,} 行 / 列 {len(base)} / threads {threads} / 表 {a.table}", flush=True)
    sms = StageModels.fit(X.loc[tr], targets, threads)
    out = WORK / "final_stage" / (a.out or a.table)
    sms.save(out, extra={"table": a.table, "train_before": a.train_before, "train_rows": int(tr.sum()), "seconds": round(time.time() - t0)})
    print(f"[{time.time() - t0:.0f}s] 保存: {out}", flush=True)

    # 2025 年の行の予測を、tp-04 の OOF（2020〜2024 で学習したモデルの予測）と比べる
    op = TABLES[a.table] / "oof/stage_features_expanding.parquet"
    oof = pd.read_parquet(op if op.exists() else WORK / "oof/stage_features_expanding.parquet", columns=["race_id", "horse_id"] + STAGE_COLS)
    va = np.flatnonzero(df["year"].to_numpy() == 2025)
    pred = sms.predict(X.iloc[va], df["race_id"].to_numpy()[va])
    m = pd.concat([df[["race_id", "horse_id"]].iloc[va].reset_index(drop=True), pred.reset_index(drop=True)], axis=1).merge(
        oof, on=["race_id", "horse_id"], suffixes=("", "_oof"))
    rep: dict = {"table": a.table, "train_before": a.train_before, "rows_2025": len(m)}
    for c in STAGE_COLS:
        x, y = m[c].to_numpy(float), m[c + "_oof"].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        rep[c] = {"corr_with_oof": float(np.corrcoef(x[ok], y[ok])[0, 1]), "mean_diff": float(np.mean(x[ok] - y[ok])), "max_abs_diff": float(np.max(np.abs(x[ok] - y[ok]))),
                  "mean_pred": float(np.mean(x[ok])), "mean_oof": float(np.mean(y[ok]))}
    if a.compare:
        o2 = StageModels.load(WORK / "final_stage" / a.compare)
        p2 = o2.predict(X.iloc[va], df["race_id"].to_numpy()[va])
        rep["vs_" + a.compare] = {c: float(np.corrcoef(pred[c].to_numpy(float), p2[c].to_numpy(float))[0, 1]) for c in STAGE_COLS}
    (out / "parity_vs_oof_2025.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
