"""段 2: 開催（race_id の先頭 8 桁）単位の K 分割の OOF 予測（主の判定。ユーザーの指示: 年で分けない）。

2017〜2024 年の開催を K=5 に無作為に分ける（種は固定）。各折は「ほかの 4 折」で学習し、その中の 1 折分を検証にして
早止め・温度を決め、残りの 4 折で当て直して、自分の折を当てる。特徴量は as-of なので、学習に未来の開催が入っても
その行の特徴量には未来の結果は入らない（ただし木が同じ馬を覚える分は、log の「覚え込み」で別に測る）。
    OMP_NUM_THREADS=4 nice -n 19 python -m src.research.pedigree_lab.kfold --tag v2 --set main
出力: OUT/kf_<tag>_<set>.parquet（race_id・horse_id・fold・p_win・p_top3）。費用 0 円。
"""
from __future__ import annotations

import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.research.pedigree_lab import ablate, data, model

K = 5


def fold_of(meetings: pd.Series, seed: int = 20261126) -> np.ndarray:
    u = np.array(sorted(pd.unique(meetings)))
    perm = np.random.default_rng(seed).permutation(len(u))
    f = np.empty(len(u), dtype=int)
    f[perm] = np.arange(len(u)) % K
    return meetings.map(dict(zip(u, f))).to_numpy()


def oof(X: pd.DataFrame, cols: list[str], target: str, fold: np.ndarray, rounds: int = 2000) -> np.ndarray:
    y = X[target].to_numpy()
    rid = X["race_id"].to_numpy()
    total = np.ones(len(X)) if target == "win" else np.minimum(3, X["n_run"].to_numpy()).astype(float)
    p = np.full(len(X), np.nan)
    for k in range(K):
        te = fold == k
        va = fold == (k + 1) % K
        tr = ~te & ~va
        m = lgb.train(model.PARAMS, lgb.Dataset(X.loc[tr, cols], y[tr]), rounds,
                      valid_sets=[lgb.Dataset(X.loc[va, cols], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        pv = m.predict(X.loc[va, cols], num_iteration=m.best_iteration)
        best = (1e9, 1.0)
        for temp in np.arange(0.6, 1.81, 0.05):
            q = model.race_norm(pv, rid[va], total[va], temp)
            s = model.race_ll(y[va], q, rid[va]) if target == "win" else model.ll(y[va], q)
            best = min(best, (s, temp))
        m2 = lgb.train(model.PARAMS, lgb.Dataset(X.loc[tr | va, cols], y[tr | va]), max(m.best_iteration, 50))
        p[te] = model.race_norm(m2.predict(X.loc[te, cols]), rid[te], total[te], best[1])
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--set", default="main")
    a = ap.parse_args()
    X = pd.read_parquet(data.OUT / f"feat_{a.tag}.parquet")
    X = X[X["date"].dt.year >= 2017].reset_index(drop=True)
    cols = ablate.col_sets(X)[a.set]
    fold = fold_of(X["meeting"])
    out = X[["race_id", "horse_id"]].copy()
    out["fold"] = fold
    out["year"] = X["date"].dt.year.to_numpy()
    for tg in ("win", "top3"):
        s = time.time()
        out[f"p_{tg}"] = oof(X, cols, tg, fold)
        print(f"{tg}: {time.time() - s:.0f}s", flush=True)
    out.to_parquet(data.OUT / f"kf_{a.tag}_{a.set}.parquet", index=False)
    n = X["n_run"].to_numpy()
    m = X["win"].to_numpy() == 1
    print("win race_ll gain vs uniform", float(np.mean(np.log(out["p_win"].to_numpy()[m])) - np.mean(np.log(1 / n[m]))))


if __name__ == "__main__":
    main()
