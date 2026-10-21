"""v6 の RB（めぐ指数/netkeiba 指数の履歴 16 列）の推論用の履歴状態を作る・学習側と照合する。

python -m src.scripts.data.build_rb_state build  [--runs <runs.parquet>] [--out <dir>]   # 1 走 1 行の履歴（クリップ後の値 x と下限つき xf）と定数（2020〜2021 年の分位点・尺度）
python -m src.scripts.data.build_rb_state verify [--runs ...] [--dates YYYY-MM-DD,...]   # 指定日の出走行について rb_features が tp65 の rob.parquet（学習側）と一致するか
runs.parquet は tp65_prep が作る（既定 <TP65_OUT>/runs.parquet = /home/jovyan/work/tp65_out）。出力の既定は data/local/inference/state/rb_v6/（本番の状態は KEIBA_V6_RB_STATE で指す）。
状態は日次で作り直す（前日までに確定した走を履歴に入れる。更新の運用は shadow の導入時に cron へ）。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features import serving_v6 as sv

ROOT = Path(__file__).resolve().parents[3]
TP65 = Path(os.environ.get("TP65_OUT", "/home/jovyan/work/tp65_out"))
DEFAULT_OUT = Path(os.environ.get("KEIBA_INFERENCE_DIR") or ROOT / "data/local/inference") / "state" / "rb_v6"


def cmd_build(a) -> None:
    runs = pd.read_parquet(a.runs).drop_duplicates(sv.KEY)
    params = sv.rb_params(runs)
    hist = sv.build_hist(runs, params)
    hist = hist[hist[["rb_mg_x", "rb_nk_x"]].notna().any(axis=1)]
    sv.save_state(a.out, hist, params, cut=str(pd.to_datetime(runs["date"]).max().date()))
    print(f"履歴 {len(hist):,} 走 → {a.out}（定数 {params}）")


def cmd_verify(a) -> None:
    runs = pd.read_parquet(a.runs).drop_duplicates(sv.KEY)
    params = sv.rb_params(runs)
    hist = sv.build_hist(runs, params)
    ref = pd.read_parquet(TP65 / "rob.parquet")
    runs["date"] = pd.to_datetime(runs["date"])
    days = [d for d in a.dates.split(",") if d] or [str(d.date()) for d in sorted(runs.loc[runs["date"].dt.year == 2025, "date"].drop_duplicates().sample(12, random_state=1))]
    worst = {}
    for d in days:
        t = runs[runs["date"] == pd.Timestamp(d)][sv.KEY + ["date"]]
        got = sv.rb_features(hist, t, params)
        m = ref.merge(got, on=sv.KEY, suffixes=("_t", "_s"))
        for c in sv.COLUMNS:
            x, y = m[f"{c}_t"].to_numpy(float), m[f"{c}_s"].to_numpy(float)
            nan_mismatch = int((np.isnan(x) != np.isnan(y)).sum())
            diff = float(np.nanmax(np.abs(x - y))) if np.isfinite(x - y).any() else 0.0
            w = worst.setdefault(c, {"rows": 0, "nan_mismatch": 0, "max_abs_diff": 0.0})
            w["rows"] += len(m)
            w["nan_mismatch"] += nan_mismatch
            w["max_abs_diff"] = max(w["max_abs_diff"], diff)
        print(d, len(t), "rows")
    ok = all(w["nan_mismatch"] == 0 and w["max_abs_diff"] < 1e-3 for w in worst.values())
    print({"ok": ok, "worst": worst})


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for n in ("build", "verify"):
        p = sp.add_parser(n)
        p.add_argument("--runs", default=str(TP65 / "runs.parquet"))
        if n == "build":
            p.add_argument("--out", default=str(DEFAULT_OUT))
        else:
            p.add_argument("--dates", default="")
    a = ap.parse_args()
    {"build": cmd_build, "verify": cmd_verify}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
