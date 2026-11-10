"""前段 S3（試行 7）の最終モデルを、2016〜2025 年の全行で 1 回学習して保存する（推論用）。

    REBUILD_2016=1 V6_EASE2=.../EASE2_t6.parquet V6_LAP=... V6_HORSE_EASE=... S3_EXTRA=.../pos_rel.parquet \
      python -m src.scripts.data.fit_final_stage_t7 --out <出力先>

作り方は docs/operations/retrain_trials/s3_try.py の試行 5・6 と同じ（L1・学習率 0.03・葉 63）。構成 a: 相対値 40 列・2000 本・乱数 3、構成 b: 相対値 80 列・2500 本・乱数 1。
相対値の元の列は、2022 年より前の行で学んだ 200 本のモデルの gain 順（試行と同じ）。入力は学習側の OOF の群（推論は同じ列名に本番の予測を写して渡す）。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

if os.environ.get("REBUILD_2016") == "1":
    from src.scripts.data import rebuild_groups_2016 as _rg

    _rg.redirect()

from src.pipeline.features.serving_stage_rel import STAGE_TARGETS, StageEnsembleT7, StageModels, stage_rel_columns
from src.pipeline.models import stage_models as sm
from src.pipeline.models.compute import detect_compute
from src.scripts.data import tp07_base as tb
from src.scripts.data import v6_stage as v6

MEMBERS = [{"name": "a", "rel": 40, "seeds": 3, "rounds": 2000}, {"name": "b", "rel": 80, "seeds": 1, "rounds": 2500}]
PARAMS = {"objective": "l1", "learning_rate": 0.03, "num_leaves": 63}


def log(m: str) -> None:
    print(f"[{time.strftime('%T')}] {m}", flush=True)


def extra_groups(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for f in filter(None, os.environ.get("S3_EXTRA", "").split(",")):
        g = pd.read_parquet(f)
        g["race_id"] = g["race_id"].astype(str)
        if "horse_id" in g.columns:
            parts.append(v6._merge(df, g))
        else:
            m = df[["race_id"]].astype(str).merge(g.drop_duplicates("race_id"), on="race_id", how="left").drop(columns=["race_id"])
            m.index = df.index
            parts.append(m)
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    df = tb._load(tb.V4)
    labels = pd.read_parquet(tb.OLD / "oof/stage_labels.parquet")
    fp = json.loads((tb.OLD / "features/fit_params.json").read_text(encoding="utf-8"))
    T = sm.build_targets(df, labels, sm.pace_class_labels(labels, fp["pace_q"], fp["rpci_q"]))
    X = v6.make_inputs("e", df, T, tb._base_cols())
    ex = extra_groups(df)
    if not len(ex.columns):
        raise SystemExit("S3_EXTRA（pos_rel.parquet）が未設定: 位置取りのメンバー内の関係列なしでは試行 7 と別物になる")
    if len(ex.columns):
        X = pd.concat([X, ex.select_dtypes(include=[np.number]).astype("float32")], axis=1)
    threads = int(os.environ.get("V6_THREADS") or detect_compute(reserve_cores=6).threads)
    yr = df["year"].astype(int).to_numpy()
    assert yr.max() >= 2025 and yr.min() <= 2016, "2016〜2025 の土台の表ではない（REBUILD_2016=1 が必要）"

    import lightgbm as lgb

    y0 = T["t_time_dev"].to_numpy(float)
    tr0 = (yr < 2022) & np.isfinite(y0)
    num = [c for c in X.columns if str(X[c].dtype) != "category"]
    m0 = lgb.train(dict(sm.REG_PARAMS, num_threads=threads), lgb.Dataset(X.loc[tr0, num], y0[tr0]), 200)
    order = [num[i] for i in np.argsort(-m0.feature_importance("gain"))]
    log(f"入力 {X.shape[1]} 列、相対値の元（上位）: {order[:10]}")

    tr = yr <= 2025
    rid = df["race_id"].astype(str).to_numpy()
    members, models = [], {}
    for mem in MEMBERS:
        src = order[: mem["rel"]]
        Xr = pd.concat([X, stage_rel_columns(X[src], rid)], axis=1)
        targets = {name: T[col].to_numpy(dtype=float)[tr] for name, col in STAGE_TARGETS.items()}
        models[mem["name"]] = []
        for s in range(mem["seeds"]):
            models[mem["name"]].append(StageModels.fit(Xr.loc[tr], targets, threads, seed=s, n_rounds=mem["rounds"], params=PARAMS))
            log(f"構成 {mem['name']} 乱数 {s} 完了（{Xr.shape[1]} 列）")
        members.append({**mem, "rel_src": src})
        del Xr
    ens = StageEnsembleT7(members, models)
    out = Path(a.out)
    ens.save(out, extra={"train_rows": int(tr.sum()), "fit_years": "2016-2025", "params": PARAMS, "seconds": round(time.time() - t0)})
    log(f"保存: {out}")

    va = np.flatnonzero(yr == 2025)
    pred = ens.predict(X.iloc[va], rid[va])
    rep = {}
    for col, tname in (("s_time_pred", "t_time_dev"), ("s_rank_pred", "t_rank_norm")):
        y = T[tname].to_numpy(float)[va]
        ok = np.isfinite(y)
        rep[tname + "_mae_2025_in_sample"] = round(float(np.abs(y[ok] - pred[col].to_numpy(float)[ok]).mean()), 5)
    (out / "sanity.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"学習内の 2025 年 MAE（過学習の目安。OOF の 0.69/0.207 より小さいのが普通）: {rep}")


if __name__ == "__main__":
    main()
