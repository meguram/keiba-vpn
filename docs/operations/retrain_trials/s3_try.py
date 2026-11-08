"""前段 S3（s_time_pred / s_rank_pred）の試行。v6_stage build e と同じ作り方（年ごとに広げる OOF）で、入力を差し替えて MAE を測る。

    PYTHONPATH=. python docs/operations/retrain_trials/s3_try.py measure        # 今の版（本番の s3x_e・旧 v4）の MAE だけ
    S3_TRY=1 PYTHONPATH=. python docs/operations/retrain_trials/s3_try.py run    # 試行（入力は環境変数）

主指標（§5.1）: 2022〜2024 年の MAE（t_time_dev / t_rank_norm）。合格線 ≤0.711 / ≤0.2108（旧版の値。e を測ってから置き直す）。
入力の環境変数: V6_EASE2・V6_HORSE_EASE・V6_LAP（v6_stage と同じ）、S3_EXTRA（追加の群 parquet、カンマ区切り。race_id[,horse_id] で結合）、
S3_PARAMS（k=v,...）、S3_SEEDS。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jovyan/work/keiba-vpn")
from src.scripts.data import tp07_base as tb  # noqa: E402
from src.scripts.data import v6_stage as v6  # noqa: E402

KEY = ["race_id", "horse_id"]
EVAL = [2022, 2023, 2024]
N = os.environ.get("S3_TRY", "x")
OUT = Path(f"/home/jovyan/work/retrain_out/s3_try{N}")


def log(m: str) -> None:
    print(f"[{time.strftime('%T')}] {m}", flush=True)


def setup():
    from src.pipeline.models import stage_models as sm

    df = tb._load(tb.V4)
    labels = pd.read_parquet(tb.OLD / "oof/stage_labels.parquet")
    fp = json.loads((tb.OLD / "features/fit_params.json").read_text(encoding="utf-8"))
    T = sm.build_targets(df, labels, sm.pace_class_labels(labels, fp["pace_q"], fp["rpci_q"]))
    return df, T


def mae(df, T, pred: dict[str, np.ndarray]) -> dict:
    yr = df["year"].astype(int).to_numpy()
    res = {}
    for col, tname in (("s_time_pred", "t_time_dev"), ("s_rank_pred", "t_rank_norm")):
        y = T[tname].to_numpy(float)
        p = np.asarray(pred[col], float)
        ok = np.isfinite(y) & np.isfinite(p) & np.isin(yr, EVAL)
        res[tname] = round(float(np.abs(y[ok] - p[ok]).mean()), 5)
        res[tname + "_by_year"] = {int(Y): round(float(np.abs(y[ok & (yr == Y)] - p[ok & (yr == Y)]).mean()), 5) for Y in EVAL}
        res[tname + "_n"] = int(ok.sum())
    return res


def aligned(df, path: Path) -> dict[str, np.ndarray]:
    s = pd.read_parquet(path)
    s["race_id"], s["horse_id"] = s["race_id"].astype(str), s["horse_id"].astype(str)
    m = df[KEY].astype(str).merge(s.drop_duplicates(KEY), on=KEY, how="left")
    return {c: m[c].to_numpy(float) for c in ("s_time_pred", "s_rank_pred")}


def measure() -> None:
    df, T = setup()
    res = {"prod_e": mae(df, T, aligned(df, tb.V4 / "v6/s3x_e.parquet")),
           "old_v4": mae(df, T, aligned(df, tb.V4 / "oof/stage_features_expanding.parquet"))}
    for k in ("s3x_e_2016",):
        p = Path("/home/jovyan/work/retrain_out/v6_2016/s3x_e.parquet")
        if p.exists():
            res[k] = mae(df, T, aligned(df, p))
    print(json.dumps(res, ensure_ascii=False, indent=1))


def extra_groups(df) -> pd.DataFrame:
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
        log(f"追加 {f}: {parts[-1].shape[1]} 列")
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)


def run() -> None:
    from src.pipeline.models import stage_models as sm

    OUT.mkdir(parents=True, exist_ok=True)
    df, T = setup()
    X = v6.make_inputs("e", df, T, tb._base_cols())
    ex = extra_groups(df)
    if len(ex.columns):
        X = pd.concat([X, ex.select_dtypes(include=[np.number]).astype("float32")], axis=1)
    rel = int(os.environ.get("S3_REL", "0"))
    if rel:
        import lightgbm as lgb
        yr0 = df["year"].astype(int).to_numpy(); y0 = T["t_time_dev"].to_numpy(float)
        tr0 = (yr0 < 2022) & np.isfinite(y0)
        num = [c for c in X.columns if str(X[c].dtype) != "category"]
        m0 = lgb.train(dict(sm.REG_PARAMS, num_threads=4), lgb.Dataset(X.loc[tr0, num], y0[tr0]), 200)
        top = [num[i] for i in np.argsort(-m0.feature_importance("gain"))[:rel]]
        F = X[top].astype("float32"); g = F.groupby(df["race_id"].astype(str).to_numpy())
        R = pd.concat([g.rank(pct=True).add_prefix("rk_"), (F - g.transform("mean")).add_prefix("dm_"), (F - g.transform("min")).add_prefix("dn_")], axis=1)
        R.index = X.index
        X = pd.concat([X, R], axis=1)
        log(f"レース内の相対値 {R.shape[1]} 列（上位: {top[:10]} …）")
    params = {}
    for kv in filter(None, os.environ.get("S3_PARAMS", "").split(",")):
        k, v = kv.split("=")
        params[k] = float(v) if "." in v else int(v)
    seeds = int(os.environ.get("S3_SEEDS", "1"))
    rounds = int(os.environ.get("S3_ROUNDS", str(sm.N_ROUNDS)))
    obj = os.environ.get("S3_OBJ", "")
    if obj:
        params["objective"] = obj
    years = sorted(int(y) for y in df["year"].unique() if int(y) <= 2025)
    log(f"列 {X.shape[1]}、params {params}、seeds {seeds}、rounds {rounds}")
    out = pd.DataFrame({"race_id": df["race_id"].astype(str).to_numpy(), "horse_id": df["horse_id"].astype(str).to_numpy()})
    for col, tname in (("s_time_pred", "t_time_dev"), ("s_rank_pred", "t_rank_norm")):
        ps = [sm.oof_regression(df, X, T[tname].to_numpy(dtype=float), years, "expanding", int(os.environ.get("V6_THREADS", "4")),
                                seed=s, n_rounds=rounds, params=params)["base"] for s in range(seeds)]
        out[col] = np.mean(ps, axis=0).astype("float32")
        log(f"{col} 完了")
    out["s_time_gap"] = (out["s_time_pred"] - out.groupby("race_id")["s_time_pred"].transform("min")).astype("float32")
    out.to_parquet(OUT / "s3x_e.parquet", index=False)
    res = {"cols": int(X.shape[1]), "env": {k: v for k, v in os.environ.items() if k.startswith(("S3_", "V6_"))},
           "mae": mae(df, T, {c: out[c].to_numpy() for c in ("s_time_pred", "s_rank_pred")})}
    (OUT / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log("MAE: " + json.dumps(res["mae"], ensure_ascii=False))


if __name__ == "__main__":
    {"measure": measure, "run": run}[sys.argv[1]]()
