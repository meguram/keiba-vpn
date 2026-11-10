"""ラップ予想の推論用の最終モデル（2016〜2025 の全行で 1 回学習）。ホストで実行（重い処理は /tmp/heavy_run.sh 経由）。

    TP55_OUT=retrain_out/tp55_2016 python -m src.scripts.data.fit_final_lap --out retrain_out/base_2016/final_lap

作るもの:
- ``model_A_full.pkl`` / ``model_A_noten.pkl``: LAP 群（lp_*）用の分類モデル。学習側の LAP 群と同じ ``n_oof=2``、シナリオ A。
- ``lapd/<目的変数>_s<乱数>.txt``: LAPD 群（ld_*）用の直接予測。Huber（alpha 0.6）、乱数 3 本、1200 本。
  目的変数は前半 3F（ld_f3）・後半 3F（ld_l3）・前後半の差（ld_q）。作り方は ``ease_lapdirect_group.py`` と同じ。
- ``lapd_meta.json``: 特徴量の並びとカテゴリの水準（推論で同じコードに揃える）。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.pipeline.features import tp55_lapfeat as lf
from src.pipeline.models import lap_forecast as lfm

OUT_IN = Path(os.environ.get("TP55_OUT", "/home/jovyan/work/retrain_out/tp55_2016"))
THREADS = int(os.environ.get("TP55_THREADS", "4"))
SEEDS = (0, 1, 2)
ROUNDS = 1200
RP = dict(objective="huber", alpha=0.6, learning_rate=0.03, num_leaves=31, min_data_in_leaf=60, feature_fraction=0.8, bagging_fraction=0.8,
          bagging_freq=1, lambda_l2=1.0, verbose=-1)


def lapd_targets(df: pd.DataFrame) -> dict[str, np.ndarray]:
    par = lf.fit_par(pd.DataFrame({"venue": df["venue"], "surface": df["surface"], "distance": df["distance"], "direction": df["direction"],
                                   "laps": [json.loads(x) for x in df["laps_json"]]}))
    laps = [json.loads(s) for s in df["laps_json"]]
    L = lfm.laps_matrix(laps)
    dev = L - lfm.par_matrix(df, par)
    n = np.array([len(a) for a in laps])
    st = lfm.scenario_stats(dev, n)
    return {"ld_f3": np.array([dev[i, :3].sum() for i in range(len(df))]),
            "ld_l3": np.array([dev[i, n[i] - 3:n[i]].sum() for i in range(len(df))]),
            "ld_q": st[:, 2] - st[:, 0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "lapd").mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(OUT_IN / "races_feat.parquet")
    d["year"] = d["date"].str[:4].astype(int)
    d = d[d["year"] <= 2025].reset_index(drop=True)
    print("学習", len(d), "レース", d["year"].min(), "-", d["year"].max(), flush=True)

    for label, drop in (("A_full", ()), ("A_noten", ("ten",))):
        m = lfm.LapForecast(scenario="A", threads=THREADS, n_oof=2, drop_groups=drop).fit(d)
        m.save(out / f"model_{label}.pkl")
        print("LAP", label, "done", flush=True)

    num, cats = lfm.feature_columns()
    cols = num + cats
    X = d[cols].copy()
    levels = {}
    for c in cats:
        X[c] = X[c].astype("category")
        levels[c] = [str(v) for v in X[c].cat.categories]
    tg = lapd_targets(d)
    sanity = {}
    for name, t in tg.items():
        ok = ~np.isnan(t)
        pred = []
        for s in SEEDS:
            b = lgb.train(dict(RP, seed=s, num_threads=THREADS), lgb.Dataset(X.loc[ok], label=t[ok]), num_boost_round=ROUNDS)
            b.save_model(str(out / "lapd" / f"{name}_s{s}.txt"))
            pred.append(b.predict(X.loc[ok]))
            print("LAPD", name, "seed", s, flush=True)
        sanity[name] = {"n": int(ok.sum()), "in_sample_mae": float(np.mean(np.abs(np.mean(pred, axis=0) - t[ok])))}
    (out / "lapd_meta.json").write_text(json.dumps({"num": num, "cats": cats, "levels": levels, "seeds": list(SEEDS), "rounds": ROUNDS, "params": RP,
                                                    "targets": list(tg), "n_train": int(len(d))}, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "sanity.json").write_text(json.dumps(sanity, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", sanity, flush=True)


if __name__ == "__main__":
    main()
