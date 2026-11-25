"""段 2: 入れ子の OOF 予測（主の判定。ユーザーの指示: 年で分けない。advisor の入れ子の形）。

- 2017〜2024 年の行を `split`（data.split_meetings・開催の無作為・種 20261125）で分ける。
  開発 = split 0・1（85%）、保留 = split 2（15%。買い方の保留として 1 回だけ開ける）。
- 開発の中で K=5 分割の OOF。分け方は group で選ぶ:
  meeting（主。開催の単位）/ dd（覚え込みの確かめ。2 代母の単位。いとこまで同じ側）。
- 保留は、5 折のモデルの予測（生）の平均で当てる（OOF と同じ物差し。2 号の段 1 の R1）。
- 複勝（3 着内）は学習率 0.05・上限 5,000 回（第 3 版は 0.03・2,000 回で上限に当たった）。早止めが上限に当たった折の数を記録する。
- 確率は「生の予測 → レースの中で和を揃える」。温度は開発の OOF 全体で 1 つ当てる（1 母数）。
    OMP_NUM_THREADS=4 nice -n 19 python -m src.research.pedigree_lab.kfold --tag v3 --set pure [--group dd]
出力: OUT/kf_<tag>_<set>_<group>.parquet（race_id・horse_id・part〔dev/hold〕・fold・p_win・p_top3）。費用 0 円。
"""
from __future__ import annotations

import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.research.pedigree_lab import ablate, data, model

K = 5
OBJ = "binary"       # --obj softmax で勝ちをレースの中の softmax で当てる


def folds(X: pd.DataFrame, dev: np.ndarray, group: str, seed: int = 20261125) -> np.ndarray:
    f = np.full(len(X), -1)
    key = X.loc[dev, "meeting" if group == "meeting" else "dd"].astype(str)
    u = np.array(sorted(pd.unique(key)))
    perm = np.random.default_rng(seed).permutation(len(u))
    lab = np.empty(len(u), dtype=int)
    lab[perm] = np.arange(len(u)) % K
    f[dev] = key.map(dict(zip(u, lab))).to_numpy()
    return f


def softmax_obj(rid_codes: np.ndarray):
    """レースの中の softmax（条件付きロジット）の目的関数。grad = p − y、hess = p(1 − p)。rid_codes は学習の行の順のレースの番号。"""
    def f(preds, ds):
        y = ds.get_label()
        e = np.exp(preds - pd.Series(preds).groupby(rid_codes).transform("max").to_numpy())
        den = np.bincount(rid_codes, e)
        p = e / den[rid_codes]
        return p - y, np.maximum(p * (1 - p), 1e-6)
    return f


def fit_temp(raw: np.ndarray, y: np.ndarray, rid: np.ndarray, total: np.ndarray, target: str) -> float:
    best = (1e9, 1.0)
    for t in np.arange(0.6, 1.81, 0.05):
        q = model.race_norm(raw, rid, total, t)
        s = model.race_ll(y, q, rid) if target == "win" else model.ll(y, q)
        best = min(best, (s, t))
    return float(best[1])


SAVE_DIR = None      # main で設定: 折のモデルと温度を保存する場所（保留・2025・26・前向きに同じ予測器を当てるため）


def run(X: pd.DataFrame, cols: list[str], target: str, f: np.ndarray, dev: np.ndarray, hold: np.ndarray,
        rounds: int = 2000) -> tuple[np.ndarray, dict]:
    y = X[target].to_numpy()
    rid = X["race_id"].to_numpy()
    total = np.ones(len(X)) if target == "win" else np.minimum(3, X["n_run"].to_numpy()).astype(float)
    raw = np.full(len(X), np.nan)
    hraw = np.zeros(hold.sum())
    iters = []
    params = dict(model.PARAMS, learning_rate=0.05) if target != "win" else model.PARAMS
    rounds = 5000 if target != "win" else rounds
    for k in range(K):
        te, va = f == k, f == (k + 1) % K
        tr = dev & ~te & ~va
        if OBJ == "softmax" and target == "win":
            prm = dict(params, objective=softmax_obj(pd.factorize(rid[tr])[0]), metric="None")
            dva = lgb.Dataset(X.loc[va, cols], y[va])
            rv = pd.factorize(rid[va])[0]

            def feval(preds, ds, rv=rv):
                e = np.exp(preds - pd.Series(preds).groupby(rv).transform("max").to_numpy())
                pp = e / np.bincount(rv, e)[rv]
                yy = ds.get_label()
                return "race_ll", float(-np.mean(np.log(np.clip(pp[yy == 1], 1e-9, 1)))), False
            m = lgb.train(prm, lgb.Dataset(X.loc[tr, cols], y[tr]), rounds, valid_sets=[dva], feval=feval,
                          callbacks=[lgb.early_stopping(100, verbose=False)])
        else:
            m = lgb.train(params, lgb.Dataset(X.loc[tr, cols], y[tr]), rounds,
                          valid_sets=[lgb.Dataset(X.loc[va, cols], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        iters.append(m.best_iteration)
        raw[te] = m.predict(X.loc[te, cols], num_iteration=m.best_iteration)
        if SAVE_DIR is not None:
            m.save_model(str(SAVE_DIR / f"{target}_fold{k}.txt"), num_iteration=m.best_iteration)
        hraw += m.predict(X.loc[hold, cols], num_iteration=m.best_iteration) / K
    raw[hold] = hraw
    if OBJ == "softmax" and target == "win":
        # ロジット → レースの最大を引いて exp（race_norm は log を取るので、レースの中の差がそのまま残る）
        raw = np.exp(raw - pd.Series(raw).groupby(rid).transform("max").to_numpy()) * 0.999
    temp = fit_temp(raw[dev], y[dev], rid[dev], total[dev], target)
    if SAVE_DIR is not None:
        import json as _j
        (SAVE_DIR / f"{target}_meta.json").write_text(_j.dumps({"temp": temp, "cols": cols, "iters": iters, "obj": OBJ}, ensure_ascii=False))
    p = np.full(len(X), np.nan)
    for part in (dev, hold):
        p[part] = model.race_norm(raw[part], rid[part], total[part], temp)
    return p, {"iters": iters, "at_cap": int(sum(i >= rounds for i in iters)), "temp": round(temp, 2),
               "temp_at_edge": bool(temp <= 0.6 or temp >= 1.8)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v3")
    ap.add_argument("--set", default="pure")
    ap.add_argument("--group", default="meeting", choices=["meeting", "dd"])
    ap.add_argument("--obj", default="binary", choices=["binary", "softmax"])
    a = ap.parse_args()
    global OBJ, SAVE_DIR
    OBJ = a.obj
    SAVE_DIR = data.OUT / f"models_{a.tag}_{a.set}_{a.group}{'' if a.obj == 'binary' else '_sm'}"
    SAVE_DIR.mkdir(exist_ok=True)
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", f"kfold_{a.tag}_{a.set}_{a.group}",
                                   "開催・2 代母の単位の OOF（学習と開発の対数損失。critic の指摘 9 で台帳を足した）"))
    X = pd.read_parquet(data.OUT / f"feat_{a.tag}.parquet")
    X = X[(X["date"].dt.year >= 2017) & (X["date"].dt.year <= 2024)].reset_index(drop=True)
    cols = ablate.col_sets(X)[a.set]
    dev = X["split"].isin([0, 1]).to_numpy()
    hold = (X["split"] == 2).to_numpy()
    f = folds(X, dev, a.group)
    out = X[["race_id", "horse_id"]].copy()
    out["part"] = np.where(dev, "dev", "hold")
    out["fold"] = f
    for tg in ("win", "top3"):
        s = time.time()
        out[f"p_{tg}"], info = run(X, cols, tg, f, dev, hold)
        print(f"{a.set} {a.group} {tg}: {time.time() - s:.0f}s {info}", flush=True)
    suffix = "" if a.obj == "binary" else "_sm"
    out.to_parquet(data.OUT / f"kf_{a.tag}_{a.set}_{a.group}{suffix}.parquet", index=False)
    data.finish_all(str(data.OUT / f"kf_{a.tag}_{a.set}_{a.group}{suffix}.parquet"))
    n = X["n_run"].to_numpy()
    m = X["win"].to_numpy() == 1
    for nm, part in (("dev", dev),):        # 保留は表示しない（段 3 の最後に 1 回だけ開ける）
        mm = m & part
        print(f"{a.set} {a.group} {nm} win race_ll gain vs uniform",
              round(float(np.mean(np.log(out["p_win"].to_numpy()[mm])) - np.mean(np.log(1 / n[mm]))), 4))


if __name__ == "__main__":
    main()
