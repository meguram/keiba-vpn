"""前段 S3（s_time_pred・s_rank_pred・s_time_gap）を、直した表（ラベル・群）で作り直す。改善サイクルの土台用（2026-11-16）。

    REBUILD_BASE=<土台> python -m src.scripts.data.model_cycles_stage run a [SEED]   # 16a の作り: レース内の相対値 40 列・3 seed・2,000 本（SEED で 1 seed ずつ並べられる）
    REBUILD_BASE=<土台> python -m src.scripts.data.model_cycles_stage run b    # 16b の作り: 相対値 80 列・1 seed・2,500 本
    REBUILD_BASE=<土台> python -m src.scripts.data.model_cycles_stage combine  # a と b の平均 ＋ 着順のレース内中心化（試行 7・16c と同じ）→ <土台>/v6/s3x_e.parquet

- 作り方は docs/operations/retrain_trials/s3_try.py の run（2016 年起点の 16a・16b・16c、README の「2016〜2025 年の土台での作り直し」）と同じ。
  入力は v6_stage.make_inputs("e") ＋ pos_rel（位置取りのメンバー内の関係）、L1・学習率 0.03・葉 63、年ごとに広げる OOF。
- 違い: 2016〜2019 年の行の追い切りの列（wc_*・tl_*）を空にする（生き残りの漏れの除去。相対値を作る前に当てる）。
- ラベルは <土台>/oof/stage_labels.parquet（data-audit が補修版の flat から作り直したもの）。
- 追走難度（EASE2_t6）・ラップ予想（LAP）・位置取り（pos_rel）は作り直していない（旧の 2016 年起点のもの）。
- 出力: <土台>/v6/s3x_e_{a,b}.parquet・s3x_e.parquet と、MAE（2022〜2024 年）の result_*.json。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["race_id", "horse_id"]
EVAL = [2022, 2023, 2024]
INPUTS = {
    "V6_EASE2": "/home/jovyan/work/retrain_out/tp54_2016/groups/EASE2_t6.parquet",
    "V6_LAP": "/home/jovyan/work/retrain_out/tp55_2016/groups/LAP.parquet",
    "V6_HORSE_EASE": "/home/jovyan/work/retrain_out/tp54_2016/horse_ease.parquet",
}
POS_REL = Path("/home/jovyan/work/retrain_out/pos_rel.parquet")
VARIANTS = {"a": {"rel": 40, "seeds": 3, "rounds": 2000}, "b": {"rel": 80, "seeds": 1, "rounds": 2500}}
PARAMS = {"learning_rate": 0.03, "num_leaves": 63, "objective": "l1"}
BLANK_BEFORE = 2020
BLANK_PREFIX = ("wc_", "tl_")
OLD_BASE = Path("/home/jovyan/work/retrain_out/base_2016")
for _k, _v in INPUTS.items():          # v6_stage は読み込んだ時点でこの 3 つを定数にするので、どの import よりも前に当てる
    os.environ.setdefault(_k, _v)


def log(m: str) -> None:
    print(f"[{time.strftime('%T')}] {m}", flush=True)


def _setup():
    if not os.environ.get("REBUILD_BASE"):
        raise SystemExit("REBUILD_BASE（重ね合わせの土台）を付ける")
    if Path(os.environ["REBUILD_BASE"]).resolve() == OLD_BASE.resolve():
        raise SystemExit("旧 base_2016 には書かない（重ね合わせの土台を指す）")
    from src.scripts.data import v6_stage as v6

    if str(v6.EASE2) != os.environ["V6_EASE2"] or str(v6.LAP) != os.environ["V6_LAP"] or str(v6.HORSE_EASE) != os.environ["V6_HORSE_EASE"]:
        raise SystemExit(f"前段の入力が環境変数と違う: {v6.EASE2} {v6.LAP} {v6.HORSE_EASE}")
    log(f"前段の入力: {v6.EASE2} / {v6.LAP} / {v6.HORSE_EASE}")
    from src.scripts.data import rebuild_groups_2016 as rg

    rg.redirect()
    from src.pipeline.models import stage_models as sm
    from src.scripts.data import tp07_base as tb

    df = tb._load(tb.V4)
    labels = pd.read_parquet(tb.OLD / "oof/stage_labels.parquet")
    fp = json.loads((tb.OLD / "features/fit_params.json").read_text(encoding="utf-8"))
    T = sm.build_targets(df, labels, sm.pace_class_labels(labels, fp["pace_q"], fp["rpci_q"]))
    return df, T, tb


def mae(df, T, pred: dict[str, np.ndarray]) -> dict:
    yr = df["year"].astype(int).to_numpy()
    res = {}
    for col, tname in (("s_time_pred", "t_time_dev"), ("s_rank_pred", "t_rank_norm")):
        y = T[tname].to_numpy(float)
        p = np.asarray(pred[col], float)
        ok = np.isfinite(y) & np.isfinite(p) & np.isin(yr, EVAL)
        res[tname] = round(float(np.abs(y[ok] - p[ok]).mean()), 5)
        res[tname + "_by_year"] = {int(Y): round(float(np.abs(y[ok & (yr == Y)] - p[ok & (yr == Y)]).mean()), 5) for Y in EVAL}
    return res


def run(variant: str, seed: int | None = None) -> None:
    """seed を渡すとその 1 seed だけを回し、s3x_e_<variant>_s<seed>.parquet に書く（16a の 3 seed を別々の処理で並べるため。combine で平均する）。"""
    import lightgbm as lgb

    from src.pipeline.models import stage_models as sm
    from src.scripts.data import v6_stage as v6

    cfg = VARIANTS[variant]
    df, T, tb = _setup()
    out_dir = Path(os.environ["REBUILD_BASE"]) / "v6"
    out_dir.mkdir(parents=True, exist_ok=True)
    X = v6.make_inputs("e", df, T, tb._base_cols())
    g = pd.read_parquet(POS_REL)
    g["race_id"], g["horse_id"] = g["race_id"].astype(str), g["horse_id"].astype(str)
    ex = v6._merge(df, g)
    X = pd.concat([X, ex.select_dtypes(include=[np.number]).astype("float32")], axis=1)
    yr = df["year"].astype(int).to_numpy()
    blank = [c for c in X.columns if c.startswith(BLANK_PREFIX)]
    if not blank:
        raise SystemExit("追い切りの列（wc_・tl_）が前段の入力に無い")
    X.loc[yr < BLANK_BEFORE, blank] = np.nan
    log(f"2016〜{BLANK_BEFORE - 1} 年の行の追い切りの列 {len(blank)} 列を空にした")
    y0 = T["t_time_dev"].to_numpy(float)
    tr0 = (yr < 2022) & np.isfinite(y0)
    num = [c for c in X.columns if str(X[c].dtype) != "category"]
    m0 = lgb.train(dict(sm.REG_PARAMS, num_threads=int(os.environ.get("V6_THREADS", "4"))), lgb.Dataset(X.loc[tr0, num], y0[tr0]), 200)
    top = [num[i] for i in np.argsort(-m0.feature_importance("gain"))[:cfg["rel"]]]
    F = X[top].astype("float32")
    grp = F.groupby(df["race_id"].astype(str).to_numpy())
    R = pd.concat([grp.rank(pct=True).add_prefix("rk_"), (F - grp.transform("mean")).add_prefix("dm_"), (F - grp.transform("min")).add_prefix("dn_")], axis=1)
    R.index = X.index
    X = pd.concat([X, R], axis=1)
    years = sorted(int(y) for y in df["year"].unique() if int(y) <= 2025)
    log(f"{variant}: 列 {X.shape[1]}、params {PARAMS}、seeds {cfg['seeds']}、rounds {cfg['rounds']}")
    out = pd.DataFrame({"race_id": df["race_id"].astype(str).to_numpy(), "horse_id": df["horse_id"].astype(str).to_numpy()})
    for col, tname in (("s_time_pred", "t_time_dev"), ("s_rank_pred", "t_rank_norm")):
        ps = [sm.oof_regression(df, X, T[tname].to_numpy(dtype=float), years, "expanding", int(os.environ.get("V6_THREADS", "4")),
                                seed=s, n_rounds=cfg["rounds"], params=dict(PARAMS))["base"] for s in ([seed] if seed is not None else range(cfg["seeds"]))]
        out[col] = np.mean(ps, axis=0).astype("float32")
        log(f"{variant} {col} 完了")
    out["s_time_gap"] = (out["s_time_pred"] - out.groupby("race_id")["s_time_pred"].transform("min")).astype("float32")
    tag = variant if seed is None else f"{variant}_s{seed}"
    out.to_parquet(out_dir / f"s3x_e_{tag}.parquet", index=False)
    res = {"variant": variant, "cfg": cfg, "cols": int(X.shape[1]), "blank_cols": len(blank), "mae": mae(df, T, {c: out[c].to_numpy() for c in ("s_time_pred", "s_rank_pred")})}
    (out_dir / f"result_{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log("MAE: " + json.dumps(res["mae"], ensure_ascii=False))


def combine() -> None:
    df, T, _tb = _setup()
    out_dir = Path(os.environ["REBUILD_BASE"]) / "v6"
    def load(v: str) -> pd.DataFrame:
        """variant の出力。seed ごとに分けて回したときは、seed の平均（run の seed 平均と同じ）。"""
        p_ = out_dir / f"s3x_e_{v}.parquet"
        if p_.exists():
            return pd.read_parquet(p_)
        parts = [pd.read_parquet(out_dir / f"s3x_e_{v}_s{k}.parquet") for k in range(VARIANTS[v]["seeds"])]
        o = parts[0][KEY].copy()
        for c in ("s_time_pred", "s_rank_pred"):
            o[c] = np.mean([q[c].to_numpy(float) for q in parts], axis=0).astype("float32")
        return o

    a, b = load("a"), load("b")
    if not (a[KEY].equals(b[KEY])):
        raise SystemExit("a と b の行がそろわない")
    out = a[KEY].copy()
    for c in ("s_time_pred", "s_rank_pred"):
        out[c] = ((a[c].to_numpy(float) + b[c].to_numpy(float)) / 2).astype("float32")
    out["s_rank_pred"] = (out["s_rank_pred"] - out.groupby("race_id")["s_rank_pred"].transform("mean") + 0.5).astype("float32")   # 正規化着順のレース内平均は 0.5
    out["s_time_gap"] = (out["s_time_pred"] - out.groupby("race_id")["s_time_pred"].transform("min")).astype("float32")
    out.to_parquet(out_dir / "s3x_e.parquet", index=False)
    m = df[KEY].astype(str).merge(out, on=KEY, how="left")
    res = {"how": "a と b の平均＋着順のレース内中心化（試行 7・16c と同じ）、2016〜、追い切りの漏れの除去あり",
           "mae": mae(df, T, {c: m[c].to_numpy(float) for c in ("s_time_pred", "s_rank_pred")})}
    (out_dir / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log("MAE: " + json.dumps(res["mae"], ensure_ascii=False))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
    else:
        combine()
