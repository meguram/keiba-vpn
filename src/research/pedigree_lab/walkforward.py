"""段 2: walk-forward の OOF 予測（年 t を、t より前の年〔2017 年〜〕だけで学習したモデルで当てる。t = 2019〜2024）。

勝ち（和 1）と 3 着内（和 min(3,n)）。列の束は ablate.col_sets の名前。温度は t−1 年を検証にして当てる。
    nice -n 19 python -m src.research.pedigree_lab.walkforward --tag v2 --set main
出力: OUT/oof_<tag>_<set>.parquet（race_id・horse_id・year・p_win・p_top3）。費用 0 円。
"""
from __future__ import annotations

import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.research.pedigree_lab import ablate, data, model


def fit_year(X: pd.DataFrame, cols: list[str], target: str, t: int, rounds: int) -> np.ndarray:
    yr = X["date"].dt.year.to_numpy()
    tr, va, te = (yr >= 2017) & (yr < t - 1), yr == t - 1, yr == t
    y = X[target].to_numpy()
    rid = X["race_id"].to_numpy()
    total = 1.0 if target == "win" else np.minimum(3, X["n_run"].to_numpy())
    m = lgb.train(model.PARAMS, lgb.Dataset(X.loc[tr, cols], y[tr]), rounds,
                  valid_sets=[lgb.Dataset(X.loc[va, cols], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    pv = m.predict(X.loc[va, cols], num_iteration=m.best_iteration)
    tot_v = total if np.isscalar(total) else total[va]
    best = (1e9, 1.0)
    for temp in np.arange(0.6, 1.81, 0.05):
        q = model.race_norm(pv, rid[va], tot_v, temp)
        s = model.race_ll(y[va], q, rid[va]) if target == "win" else model.ll(y[va], q)
        best = min(best, (s, temp))
    # 学習を t−1 年まで広げ、同じ回数で当て直す
    m2 = lgb.train(model.PARAMS, lgb.Dataset(X.loc[tr | va, cols], y[tr | va]), max(m.best_iteration, 50))
    pt = m2.predict(X.loc[te, cols])
    return model.race_norm(pt, rid[te], total if np.isscalar(total) else total[te], best[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--set", default="main")
    ap.add_argument("--rounds", type=int, default=2000)
    a = ap.parse_args()
    X = pd.read_parquet(data.OUT / f"feat_{a.tag}.parquet")
    cols = ablate.col_sets(X)[a.set]
    yr = X["date"].dt.year.to_numpy()
    out = X.loc[yr >= 2019, ["race_id", "horse_id"]].copy()
    out["year"] = yr[yr >= 2019]
    for tg in ("win", "top3"):
        p = np.full(len(X), np.nan)
        for t in range(2019, 2025):
            s = time.time()
            p[yr == t] = fit_year(X, cols, tg, t, a.rounds)
            print(f"{tg} {t}: {time.time() - s:.0f}s", flush=True)
        out[f"p_{tg}"] = p[yr >= 2019]
    out.to_parquet(data.OUT / f"oof_{a.tag}_{a.set}.parquet", index=False)
    print("oof", out.shape)


if __name__ == "__main__":
    main()
