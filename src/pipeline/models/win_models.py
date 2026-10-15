"""tp-05a の部品: 勝ち馬確率のモデル（二値・レース内 softmax・ランキング）の学習と、確率の整合・評価。

方式（モデリング戦略書 §3）:
  M1 二値: p_win / p_top2 / p_top3 を別々の二値モデルで学習し、レース内で Σp = 1 / 2 / 3 になるよう後で正規化する。
  M2 レース内 softmax: 馬ごとのスコアをレース内で softmax し、勝ち馬の対数尤度を最大化する（LightGBM のカスタム目的関数）。
  M3 ランキング: lambdarank / rank_xendcg のスコアを、レース内 softmax と温度で確率にする。
評価は p_win の LogLoss（主）、レース内 softmax の勝ち馬の負の対数尤度（race_nll）、Brier、ECE、1 位的中率、NDCG@3。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

EPS = 1e-12


# ── 確率の整合
def normalize_to_total(race_ids: np.ndarray, p: np.ndarray, total: float, iters: int = 50) -> np.ndarray:
    """レース内の合計が total になるよう、各馬の確率を反復比例（0〜1 に収めながら）でそろえる。頭数が total 以下のレースは全頭 1 にする。"""
    p = np.clip(np.asarray(p, dtype="float64"), EPS, 1 - EPS)
    out = p.copy()
    codes, uniq = pd.factorize(race_ids)
    order = np.argsort(codes, kind="stable")
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    for idx in np.split(order, bounds):
        n = len(idx)
        if n <= total:
            out[idx] = 1.0
            continue
        q = p[idx]
        for _ in range(iters):
            s = q.sum()
            if abs(s - total) < 1e-9:
                break
            q = np.clip(q * total / s, EPS, 1 - EPS)
        out[idx] = q
    return out


def softmax_by_race(race_ids: np.ndarray, score: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """レース内 softmax（合計 1）。"""
    d = pd.DataFrame({"r": race_ids, "s": np.asarray(score, dtype="float64") / temperature})
    d["m"] = d.groupby("r")["s"].transform("max")
    d["e"] = np.exp(d["s"] - d["m"])
    return (d["e"] / d.groupby("r")["e"].transform("sum")).to_numpy()


# ── 評価
def binary_logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def race_nll(race_ids: np.ndarray, p_win: np.ndarray, y_win: np.ndarray) -> float:
    """レース内で正規化した確率での、勝ち馬の負の対数尤度（勝ち馬がちょうど 1 頭のレースだけ）。"""
    d = pd.DataFrame({"r": race_ids, "p": np.clip(p_win, EPS, None), "y": y_win})
    d["z"] = d.groupby("r")["p"].transform("sum")
    w = d[d["y"] == 1]
    cnt = w.groupby("r").size()
    w = w[w["r"].isin(cnt[cnt == 1].index)]
    return float(-np.log(w["p"] / w["z"]).mean())


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """確率の分位で 15 ビンに分けた ECE（頻度の少ない高確率側も見られるよう等頻度ビン）。"""
    q = pd.qcut(pd.Series(p), q=bins, duplicates="drop")
    g = pd.DataFrame({"p": p, "y": y, "b": q.to_numpy()}).groupby("b", observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    return float((g["n"] * (g["p"] - g["y"]).abs()).sum() / g["n"].sum())


def top1_hit_rate(race_ids: np.ndarray, score: np.ndarray, y_win: np.ndarray) -> float:
    d = pd.DataFrame({"r": race_ids, "s": score, "y": y_win})
    idx = d.groupby("r")["s"].idxmax()
    return float(d.loc[idx, "y"].mean())


def ndcg_at_k(race_ids: np.ndarray, score: np.ndarray, finish_position: np.ndarray, k: int = 3) -> float:
    """関連度 = max(0, 4 − 着順)（1 着 3、2 着 2、3 着 1、それ以外 0）の NDCG@k のレース平均。"""
    d = pd.DataFrame({"r": race_ids, "s": score, "rel": np.clip(4 - finish_position, 0, None)})
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    vals = []
    for _, g in d.groupby("r"):
        top = g.sort_values("s", ascending=False)["rel"].to_numpy()[:k]
        ideal = np.sort(g["rel"].to_numpy())[::-1][:k]
        idcg = float((ideal * disc[: len(ideal)]).sum())
        if idcg > 0:
            vals.append(float((top * disc[: len(top)]).sum()) / idcg)
    return float(np.mean(vals))


def per_race_loss(race_ids: np.ndarray, p: np.ndarray, y: np.ndarray) -> pd.Series:
    """レースごとの二値 LogLoss の平均（レース単位の paired bootstrap 用）。"""
    p = np.clip(p, 1e-15, 1 - 1e-15)
    l = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return pd.Series(l).groupby(np.asarray(race_ids)).mean()


def paired_bootstrap_diff(loss_a: pd.Series, loss_b: pd.Series, n_boot: int = 2000, seed: int = 0) -> dict:
    """レース単位の paired bootstrap で、平均損失の差（b − a、負なら b が良い）の 95% 信頼区間を返す。"""
    both = pd.concat([loss_a.rename("a"), loss_b.rename("b")], axis=1).dropna()
    d = (both["b"] - both["a"]).to_numpy()
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return {"diff": float(d.mean()), "lo": float(np.quantile(means, 0.025)), "hi": float(np.quantile(means, 0.975)), "races": int(len(d))}


def evaluate_probs(race_ids: np.ndarray, p_win: np.ndarray, p_top2: np.ndarray | None, p_top3: np.ndarray | None, finish_position: np.ndarray) -> dict:
    """p_win を中心にした全指標。p_top2・p_top3 があればそれらの LogLoss も返す。"""
    y1 = (finish_position == 1).astype(float)
    out = {"logloss_win": binary_logloss(p_win, y1), "race_nll": race_nll(race_ids, p_win, y1), "brier_win": float(((p_win - y1) ** 2).mean()),
           "ece_win": expected_calibration_error(p_win, y1), "top1_hit": top1_hit_rate(race_ids, p_win, y1),
           "ndcg3": ndcg_at_k(race_ids, p_win, finish_position, 3)}
    if p_top2 is not None:
        out["logloss_top2"] = binary_logloss(p_top2, (finish_position <= 2).astype(float))
    if p_top3 is not None:
        out["logloss_top3"] = binary_logloss(p_top3, (finish_position <= 3).astype(float))
    return out


# ── 学習
BASE_PARAMS = {"learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 100, "bagging_fraction": 0.8, "bagging_freq": 1,
               "feature_fraction": 0.7, "lambda_l2": 5.0, "verbosity": -1, "deterministic": True, "force_row_wise": True}
N_ROUNDS = 300


def _groups(race_ids: np.ndarray) -> np.ndarray:
    """race_ids が連続して並んでいる前提で、各レースの頭数を返す（LightGBM の group 用）。"""
    _, counts = np.unique(pd.factorize(race_ids)[0], return_counts=True)
    ids = pd.factorize(race_ids)[0]
    # factorize は出現順なので、連続していれば出現順の頭数と一致する
    change = np.flatnonzero(np.diff(ids)) + 1
    sizes = np.diff(np.concatenate(([0], change, [len(ids)])))
    assert len(sizes) == len(counts), "レースの行が連続していません。race_id・horse_number で並べ替えてください"
    return sizes


def softmax_objective(race_ids: np.ndarray, y_win: np.ndarray):
    """レース内 softmax のカスタム目的関数（LightGBM）。勝ち馬がちょうど 1 頭でないレースは勾配 0（学習に使わない）。
    損失 = −log softmax(score)[勝ち馬]。勾配 = p − y、ヘシアン = p(1−p)（対角近似）。"""
    codes = pd.factorize(race_ids)[0]
    n_win = pd.Series(y_win).groupby(codes).transform("sum").to_numpy()
    use = (n_win == 1).astype(float)

    def fobj(preds, dataset):
        s = np.asarray(preds, dtype="float64")
        m = pd.Series(s).groupby(codes).transform("max").to_numpy()
        e = np.exp(s - m)
        p = e / pd.Series(e).groupby(codes).transform("sum").to_numpy()
        grad = (p - y_win) * use
        hess = np.maximum(p * (1 - p), 1e-6) * use + (1 - use) * 1e-6
        return grad, hess

    return fobj


def fit_predict(kind: str, X_tr: pd.DataFrame, y_tr: dict, race_tr: np.ndarray, X_va: pd.DataFrame, race_va: np.ndarray, threads: int,
                seed: int = 0, n_rounds: int = N_ROUNDS, params: dict | None = None, weight: np.ndarray | None = None) -> dict:
    """kind: 'binary'（win/top2/top3 を別々に）, 'softmax'（win のみ）, 'lambdarank', 'xendcg'。
    y_tr: {'win':..., 'top2':..., 'top3':..., 'finish':...}。戻り値: {'win': 検証のスコア/確率, ...}。"""
    import lightgbm as lgb

    p = dict(BASE_PARAMS, **(params or {}), seed=seed, num_threads=threads)
    cats = [c for c in X_tr.columns if str(X_tr[c].dtype) == "category"]
    out: dict = {}
    if kind == "binary":
        for tgt in ("win", "top2", "top3"):
            ds = lgb.Dataset(X_tr, label=y_tr[tgt], weight=weight, categorical_feature=cats or "auto")
            out[tgt] = lgb.train(dict(p, objective="binary"), ds, num_boost_round=n_rounds).predict(X_va, raw_score=False)
    elif kind == "softmax":
        ds = lgb.Dataset(X_tr, label=y_tr["win"], categorical_feature=cats or "auto")
        # LightGBM 4 系: カスタム目的関数は params["objective"] に関数として渡す
        b = lgb.train(dict(p, objective=softmax_objective(race_tr, np.asarray(y_tr["win"], dtype="float64"))), ds, num_boost_round=n_rounds)
        out["win_score"] = b.predict(X_va, raw_score=True)
    elif kind in ("lambdarank", "xendcg"):
        rel = np.clip(4 - np.asarray(y_tr["finish"]), 0, None).astype(int)   # 1 着 3、2 着 2、3 着 1
        ds = lgb.Dataset(X_tr, label=rel, group=_groups(race_tr), categorical_feature=cats or "auto")
        obj = "lambdarank" if kind == "lambdarank" else "rank_xendcg"
        extra = {"lambdarank_truncation_level": 5, "eval_at": [3]} if kind == "lambdarank" else {}
        out["win_score"] = lgb.train(dict(p, objective=obj, **extra), ds, num_boost_round=n_rounds).predict(X_va, raw_score=True)
    else:
        raise ValueError(kind)
    return out


# ── C1: レース内の相対化
def add_race_relative(df: pd.DataFrame, cols: Sequence[str], suffix: tuple[str, str, str] = ("_rz", "_rrank", "_rgap")) -> pd.DataFrame:
    """各列について、レース内の z、順位（降順 1 位 = 最大）、最大との差を足す（as-of の特徴量どうしの比較だけで、結果は使わない）。"""
    g = df.groupby("race_id", sort=False)
    out = {}
    for c in cols:
        x = df[c].astype("float64")
        mu, sd = g[c].transform("mean"), g[c].transform("std")
        out[c + suffix[0]] = ((x - mu) / sd.replace(0, np.nan)).astype("float32")
        out[c + suffix[1]] = g[c].rank(ascending=False, method="average").astype("float32")
        out[c + suffix[2]] = (x - g[c].transform("max")).astype("float32")
    return pd.DataFrame(out, index=df.index)
