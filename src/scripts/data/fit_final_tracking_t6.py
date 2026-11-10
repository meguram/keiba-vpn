"""追走難度 v2（試行 6 の設定）の最終モデルを、2016〜2025 年の全行で 1 回学習して保存する（推論用）。

    TP54_OUT=.../tp54_2016 python -m src.scripts.data.fit_final_tracking_t6 --out <出力先>

作り方は docs/operations/retrain_trials/ease_oof_t6.py と同じ（drain = CORE + ラップ予想 LAP/LAPD + 位置取りの関係 + 過去の終いの落ち、full = 全特徴 + 同じ追加、demand = 要求のみ）。
ベースラインの係数は全行で当てはめる。ease_pct の分位点は、学習側の OOF の drain（EASE2_t6.parquet）から取る（前段 S3 が学習で見た分布に合わせる）。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

os.environ.setdefault("TP54_OUT", "/home/jovyan/work/retrain_out/tp54_2016")
from src.pipeline.models import tracking_difficulty_v2 as td2
from src.scripts.data import tp54_model as m

SEEDS = (0, 1, 2)
ROUNDS = 1500
LAP_GROUP = "/home/jovyan/work/retrain_out/tp55_2016/groups/LAP.parquet"
LAPD_GROUP = "/home/jovyan/work/retrain_out/tp55_2016/groups/LAPD.parquet"
POS_OOF = "/home/jovyan/work/keiba-vpn/data/local/modeling/pos_model/pos_oof.parquet"
EASE2_OOF = "/home/jovyan/work/retrain_out/tp54_2016/groups/EASE2_t6.parquet"
PP_COLS = ["pp_rank", "pp_ahead", "pp_front_n", "pp_gap_mean"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%T')}] {msg}", flush=True)


def build_inputs(h: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """ease_oof_t6 と同じ L（LAP）・D（LAPD と 2 つの掛け合わせ）・P（位置取りの関係）。pe は初角位置の予想。"""
    lap = pd.read_parquet(LAP_GROUP)
    lap["race_id"], lap["horse_id"] = lap["race_id"].astype(str), lap["horse_id"].astype(str)
    L = h[["race_id", "horse_id"]].merge(lap.drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left").drop(columns=["race_id", "horse_id"])
    L.index = h.index
    ld = pd.read_parquet(LAPD_GROUP)
    ld["race_id"] = ld["race_id"].astype(str)
    D = h[["race_id"]].merge(ld.drop_duplicates("race_id"), on="race_id", how="left").drop(columns=["race_id"])
    D.index = h.index
    pe = h["pf_early"].astype(float)
    D["ld_front_x_f3"] = (1 - pe) * D["ld_f3"]
    D["ld_rear_x_q"] = pe * D["ld_q"]
    po = pd.read_parquet(POS_OOF)
    po["race_id"], po["horse_id"] = po["race_id"].astype(str), po["horse_id"].astype(str)
    g = po.groupby("race_id")["pos_pred"]
    po["pp_rank"], po["pp_ahead"] = g.rank(pct=True), g.rank(method="min") - 1
    po["pp_front_n"], po["pp_gap_mean"] = g.transform(lambda x: (x < 0.15).sum()), po["pos_pred"] - g.transform("mean")
    P = h[["race_id", "horse_id"]].merge(po[["race_id", "horse_id", *PP_COLS]].drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left").drop(columns=["race_id", "horse_id"])
    P.index = h.index
    return L, D, P, pe


def past_drop_inputs(h: pd.DataFrame, y: pd.Series, pe: pd.Series) -> pd.DataFrame:
    """馬の過去の終いの落ち（y = 実測の上がり偏差 − ベースライン）の直近 1・3 走平均に、前にいる度合い・圧力・ペースを掛けたもの。"""
    gy = y.groupby(h["horse_id"], sort=False)
    fr = 1 - pe
    hy3 = pd.concat([gy.shift(k) for k in (1, 2, 3)], axis=1).mean(axis=1)
    return pd.DataFrame({"hy3_x_front": hy3 * fr, "hy1_x_front": gy.shift(1) * fr, "hy3_x_press": hy3 * h["pf_pressure"].astype(float),
                         "hy3_x_zhat": hy3 * h["z_hat"].astype(float)}, index=h.index)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    h = m.load()
    h["race_id"], h["horse_id"] = h["race_id"].astype(str), h["horse_id"].astype(str)
    assert pd.to_datetime(h["date"]).dt.year.max() <= 2025, "2026 年以降の行が混ざっている（学習は 2016〜2025 のみ）"
    X = td2.make_features(h)
    ok = h["pf_early"].notna()
    L, D, P, pe = build_inputs(h)
    coef = m.baseline_fit(h, ok)
    _, y = m.make_target(h, coef)
    Hi = past_drop_inputs(h, y, pe)
    tr = ok & y.notna()
    XC = pd.concat([X[m.CORE], L, D, P, Hi], axis=1)
    XB = pd.concat([X, L, D, P, Hi], axis=1)
    log(f"学習 {int(tr.sum()):,} 行、drain {XC.shape[1]} 列、full {XB.shape[1]} 列")
    B = dict(m.PARAMS, learning_rate=0.015, num_leaves=63, min_child_samples=150, num_threads=int(os.environ.get("TP54_THREADS", "6")))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, XX in (("drain", XC), ("full", XB)):
        for s in SEEDS:
            lgb.train(dict(B, seed=s), lgb.Dataset(XX[tr], y[tr].to_numpy()), num_boost_round=ROUNDS).save_model(str(out / f"{name}_s{s}.txt"))
            log(f"{name} 乱数 {s} 完了")
    m.fit(X.loc[tr, m.DEMAND_ONLY], y[tr].to_numpy()).save_model(str(out / "demand.txt"))
    oof = pd.read_parquet(EASE2_OOF)["EASE2_drain"]
    q = np.nanquantile(oof, np.linspace(0, 1, 101)).tolist()
    meta = {"version": "tp54-t6-final", "drain_features": list(XC.columns), "full_features": list(XB.columns), "demand_features": m.DEMAND_ONLY,
            "lap_columns": list(L.columns), "lapd_columns": list(D.columns), "pos_columns": PP_COLS, "past_drop_columns": list(Hi.columns),
            "quantiles": q, "y_clip": td2.Y_CLIP, "baseline_coef": [float(c) for c in coef], "seeds": list(SEEDS), "rounds": ROUNDS,
            "trained_years": [int(h["year"].min()), int(h["year"].max())], "n_train": int(tr.sum()), "seconds": round(time.time() - t0)}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    pos = out / "position_effect.json"
    src = m.OUT / "position_effect.json"
    if src.exists() and not pos.exists():
        pos.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    # 学習内の目安（OOF の drain と、最終モデルの drain の相関）
    pred = np.mean([lgb.Booster(model_file=str(out / f"drain_s{s}.txt")).predict(XC[tr]) for s in SEEDS], axis=0)
    ref = pd.read_parquet(EASE2_OOF).loc[tr.to_numpy(), "EASE2_drain"].to_numpy(float)
    okc = np.isfinite(ref)
    (out / "sanity.json").write_text(json.dumps({"corr_in_sample_vs_oof_drain": round(float(np.corrcoef(pred[okc], ref[okc])[0, 1]), 4)}), encoding="utf-8")
    log(f"保存: {out}")


if __name__ == "__main__":
    main()
