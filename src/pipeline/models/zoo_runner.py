"""tp-05b〜e 共通: 1 つのモデル（recipe + 特徴量 + パラメータ）を fold × seed で学習・予測し、OOF を保存して指標を返す。"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.pipeline.models import win_models as wm
from src.pipeline.models import zoo


def metrics_by_year(meta_idx: pd.DataFrame, P: dict, fin: np.ndarray, race: np.ndarray, model: str) -> pd.DataFrame:
    """年（fold）ごとの指標。meta_idx は year を持つ（P・fin・race と同じ並び）。"""
    rows = []
    for yr, g in meta_idx.groupby("year"):
        ii = g.index.to_numpy()
        fp = fin[ii]
        m = wm.evaluate_probs(race[ii], P["win"][ii], P["top2"][ii], P["top3"][ii], fp)
        m["auc_win"] = float(roc_auc_score((fp == 1), P["win"][ii]))
        m["auc_top2"] = float(roc_auc_score((fp <= 2), P["top2"][ii]))
        m["auc_top3"] = float(roc_auc_score((fp <= 3), P["top3"][ii]))
        m["brier_top2"] = float(((P["top2"][ii] - (fp <= 2)) ** 2).mean())
        m["brier_top3"] = float(((P["top3"][ii] - (fp <= 3)) ** 2).mean())
        rows.append({"model": model, "fold": f"valid_{int(yr)}", **m})
    return pd.DataFrame(rows)


def run_oof(name: str, recipe: str, X: pd.DataFrame, meta: pd.DataFrame, folds, seeds, threads: int, params: dict | None = None,
            out_dir: Path | None = None, gammas: tuple[float, float] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """戻り値: (fold ごとの指標, OOF 予測)。meta は race_id・horse_id・finish_position・year を持つ。
    OOF は seed 平均後の p_win・p_top2・p_top3（レース内の合計がそろった確率）。
    p_top2・p_top3 を直接出さないモデルは、OOF 全体で当てはめた γ（gammas を渡せば固定）の割引 Harville で導く。"""
    race = meta["race_id"].to_numpy()
    fin = meta["finish_position"].to_numpy()
    Y = {"win": (fin == 1).astype(int), "top2": (fin <= 2).astype(int), "top3": (fin <= 3).astype(int), "finish": fin}
    sums = {k: np.zeros(len(meta)) for k in ("win", "top2", "top3")}
    wins_only = np.zeros(len(meta))
    direct23 = None
    covered = np.zeros(len(meta), bool)
    t0 = time.time()
    params = dict(params or {})
    min_year = params.pop("min_train_year", None)       # この年より前の行は学習に使わない（助走期間の除外）
    halflife = params.pop("halflife_years", None)       # 新しい年ほど重くする（半減期。年単位）
    year = meta["year"].to_numpy()
    for f in folds:
        tr, va = f.train_index, f.valid_index
        if min_year is not None:
            tr = tr[year[tr] >= min_year]
        if len(tr) < 1000:      # 学習行が足りない fold（例: 2020〜2021 年を外した fold 2022）は飛ばす
            continue
        ytr = {k: v[tr] for k, v in Y.items()}
        p_f = dict(params)
        if halflife:
            age = (year[va].min() - 1) - year[tr]       # 検証年の前年を 0 とした経過年数
            p_f["weight"] = 0.5 ** (np.clip(age, 0, None) / float(halflife))
        for seed in seeds:
            out = zoo.fit_predict(recipe, X.iloc[tr], ytr, race[tr], X.iloc[va], race[va], threads, seed=seed, params=p_f)
            direct23 = "top2" in out
            if direct23:
                fp_ = zoo.finalize(race[va], out)
                for k in sums:
                    sums[k][va] += fp_[k]
            else:
                wins_only[va] += wm.normalize_to_total(race[va], out["win"], 1.0)
        covered[va] = True
    n = len(seeds)
    idx = np.flatnonzero(covered)
    if direct23:
        P = {k: sums[k][idx] / n for k in sums}
    else:
        win = wins_only[idx] / n
        g2, g3 = gammas or (zoo.fit_gamma(race[idx], win, fin[idx], 2), zoo.fit_gamma(race[idx], win, fin[idx], 3))
        p2, p3 = zoo.harville_topk(race[idx], win, g2, g3)
        P = {"win": win, "top2": wm.normalize_to_total(race[idx], p2, 2.0), "top3": wm.normalize_to_total(race[idx], p3, 3.0)}
    oof = meta.iloc[idx][["race_id", "horse_id", "year"]].copy().reset_index(drop=True)
    for k in P:
        oof[f"p_{k}"] = P[k]
    res = metrics_by_year(oof, P, fin[idx], race[idx], name)
    res.attrs["seconds"] = time.time() - t0
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        oof.to_parquet(out_dir / f"{name}.parquet", index=False)
    return res, oof
