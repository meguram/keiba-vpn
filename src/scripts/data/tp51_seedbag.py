"""tp-51: V4 の既存メンバーを『別の seed でもう 3 本』学習して OOF を作る（seed 数を 3 → 6 にするバギングの効果の測定）。

python -m src.scripts.data.tp51_seedbag run NAME [--seeds 3,4,5]     # NAME: xgb_extras_ped / lgb_softmax / linear / lgb_extras_ped / lgb_extras
出力: <TP51_OUT>/members/<NAME>_sd345.parquet（valid 2022〜2024 の OOF。tp51_stack が『元の 3 seed + 追加の 3 seed』の logit 平均に置き換えて評価する）
列の作り方は tp27_eval.cmd_member と同じ（V4 の core）。重い処理は /tmp/heavy_run.sh 経由で起動する。
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd

os.environ.setdefault("TP40_BASE", "old")
from src.scripts.data import tp27_eval as t  # noqa: E402
from src.scripts.data import tp40_eval as e40  # noqa: E402,F401

OUT = Path(os.environ.get("TP51_OUT") or "/home/jovyan/work/tp51_out")
THREADS = int(os.environ.get("TP51_THREADS", 3))
KEY = ["race_id", "horse_id"]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cmd_run(a) -> None:
    from src.pipeline.models import zoo_runner
    from src.scripts.data import tp25e_eval as e

    p = OUT / "members" / f"{a.name}_sd345.parquet"
    if p.exists():
        _log(f"{p.name} 既存")
        return
    df, meta, folds, members, TLX = e.load_v3()
    recipe, params, X = members[a.name]
    X = pd.concat([X, TLX], axis=1)
    aux = pd.read_parquet(t.AUX_DIR / "aux_B.parquet")
    aux["race_id"], aux["horse_id"] = aux["race_id"].astype(str), aux["horse_id"].astype(str)
    m = df[KEY].merge(aux, on=KEY, how="left")
    A = m[[c for c in m.columns if c not in KEY]]
    A.index = df.index
    E = t._extra_for_ensemble(df, "core", t.NN_TOP if recipe == "nn" else None)
    X = pd.concat([X, A, E], axis=1)
    seeds = [int(s) for s in a.seeds.split(",")]
    t0 = time.time()
    res, oof = zoo_runner.run_oof(f"{a.name}_sd345", recipe, X, meta, folds, seeds, THREADS, params=dict(params), out_dir=None)
    p.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(p, index=False)
    _log(f"{a.name} seeds={seeds}: win {res['logloss_win'].mean():.5f} top2 {res['logloss_top2'].mean():.5f} top3 {res['logloss_top3'].mean():.5f} cols={X.shape[1]} ({time.time() - t0:.0f}s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("name")
    r.add_argument("--seeds", default="3,4,5")
    a = ap.parse_args()
    cmd_run(a)


if __name__ == "__main__":
    main()
