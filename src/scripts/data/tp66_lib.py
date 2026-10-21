"""tp-66 めぐ指数の集約方式の比較: 過去走のめぐ指数から『今走の好走に値する度』を出す指標の定義（as-of）。

各出走について、その馬の**今走の日付より前**の有効な走（絶対指数 megu_index_v3a_abs_robust が ok かつ低信頼でない）だけを使う。
同じ日の走は使わない。基準（突破ライン）は評価年より前の年だけで作った表を使う。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.pipeline.inference.megu_mark import DECAY, DIST_SCALE, LOOKBACK, NS, SURF_MISMATCH, dist_tol, features_for, winsor_mean  # noqa: F401


def compute_features(df: pd.DataFrame, lines: dict[str, float | None], min_year: int) -> pd.DataFrame:
    """df: 列 horse_id, date, race_id, idx, ok, surf, distance, year。lines: race_id → 突破ライン（無ければ None）。
    min_year 以降の行だけ特徴量を作る（それ以前の行は履歴にだけ使う）。戻り値は df と同じ index。"""
    d = df.sort_values(["horse_id", "date", "race_id"], kind="stable")
    hid = d["horse_id"].to_numpy()
    dates = d["date"].to_numpy()
    idx = d["idx"].to_numpy(dtype=float)
    idxf = d["idx_fl"].to_numpy(dtype=float) if "idx_fl" in d else idx
    ok = d["ok"].to_numpy(dtype=bool)
    sf = d["surf"].map({"芝": 0, "ダ": 1}).fillna(-1).to_numpy(dtype=int)
    dist = pd.to_numeric(d["distance"], errors="coerce").to_numpy(dtype=float)
    year = d["year"].to_numpy()
    rid = d["race_id"].to_numpy()
    rows: list[dict | None] = [None] * len(d)
    hist: list[tuple] = []
    pending: list[tuple] = []
    pdate = None
    cur = None
    for i in range(len(d)):
        if hid[i] != cur:
            cur, hist, pending, pdate = hid[i], [], [], None
        if pdate is not None and dates[i] > pdate:
            hist.extend(pending)
            pending = []
        if year[i] >= min_year and hist:
            dd = dist[i] if dist[i] == dist[i] else None
            rows[i] = features_for(hist, int(sf[i]), dd, lines.get(rid[i]))
        if ok[i]:
            pending.append((float(idx[i]), int(sf[i]), float(dist[i]) if dist[i] == dist[i] else None, float(idxf[i])))
            pdate = dates[i]
        elif pdate is None:
            pdate = dates[i]
    out = pd.DataFrame([r if r is not None else {} for r in rows], index=d.index)
    return out.reindex(df.index)


def within_race_rank(df: pd.DataFrame, col: str, ascending: bool = False) -> pd.Series:
    """レース内の順位（1 が最良）。欠損は NaN。"""
    return df.groupby("race_id")[col].rank(ascending=ascending, method="first")


def within_race_z(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby("race_id")[col]
    sd = g.transform("std")
    return ((df[col] - g.transform("mean")) / sd.replace(0, np.nan)).astype(float)


def mark_top_k(df: pd.DataFrame, score: str, k: int, min_runners: int = 6, require: str | None = None) -> pd.Series:
    """レースごとに score の上位 k 頭に True。score が NaN の馬、出走数が min_runners 未満のレースは False。
    require があれば、その列が True の馬だけ（例 突破ライン超え）。"""
    s = df[score]
    ok = s.notna()
    if require is not None:
        ok &= df[require].fillna(False).astype(bool)
    # 同点は馬番の小さい方を上にする（行の並び順＝着順に引きずられないため、事前に分かる馬番で決める）
    hn = pd.to_numeric(df["horse_number"], errors="coerce").fillna(0.0) if "horse_number" in df else pd.Series(0.0, index=df.index)
    r = (s.where(ok) - hn * 1e-9).groupby(df["race_id"]).rank(ascending=False, method="first")
    n = df.groupby("race_id")["horse_id"].transform("count")
    return (r <= k) & (n >= min_runners)


def logloss(y: np.ndarray, p: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_logit(X: np.ndarray, y: np.ndarray, l2: float = 1e-3, iters: int = 50) -> np.ndarray:
    """切片つきロジスティック回帰（ニュートン法、軽い L2）。X に切片列は含めない。戻り値は [切片, 係数...]。"""
    A = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(A.shape[1])
    for _ in range(iters):
        p = sigmoid(A @ w)
        g = A.T @ (p - y) + l2 * np.r_[0.0, w[1:]]
        W = p * (1 - p)
        H = (A * W[:, None]).T @ A + l2 * np.diag(np.r_[0.0, np.ones(len(w) - 1)])
        step = np.linalg.solve(H + 1e-9 * np.eye(len(w)), g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def predict_logit(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return sigmoid(w[0] + X @ w[1:])


def race_bootstrap_delta(race_ids: np.ndarray, a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """行ごとの損失 a(新) − b(基準) の平均差の 95% 区間（レース単位の再標本化）。負なら新が良い。"""
    u, inv = np.unique(race_ids, return_inverse=True)
    sa = np.bincount(inv, weights=a - b)
    cnt = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    m = len(u)
    draws = rng.multinomial(m, np.full(m, 1.0 / m), size=n_boot)
    num = draws @ sa
    den = draws @ cnt
    d = num / den
    return float((sa.sum() / cnt.sum())), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def v3b_feature(df: pd.DataFrame, n_runs: int = 5) -> pd.Series:
    """v3b の集約値（tp-69: 勝ち馬−40点の下限・低信頼の重み 0.5・減衰 0.65 の加重 Huber）。
    各行について、今走の日付より前の有効な走（低信頼も含む。重みを下げるだけ）の直近 n_runs 走から作る。
    df: 列 horse_id, date, valid（有効な走）, low（低信頼）, idx_fl（下限つき指数）。"""
    from src.pipeline.megu_index import v3b

    pos = np.arange(len(df))
    v = df[df["valid"]].sort_values(["horse_id", "date"], kind="stable")
    vh = v["horse_id"].to_numpy()
    vd = v["date"].to_numpy().astype("datetime64[D]").astype(np.int64)
    vv = v["idx_fl"].to_numpy(dtype=float)
    vl = v["low"].to_numpy(dtype=bool)
    starts: dict = {}
    for i, h in enumerate(vh):
        starts.setdefault(h, i)
    order = sorted(starts.items(), key=lambda kv: kv[1])
    bounds = {h: (i0, order[j + 1][1] if j + 1 < len(order) else len(vh)) for j, (h, i0) in enumerate(order)}   # 馬ごとの範囲 [start, end)
    hid = df["horse_id"].to_numpy()
    dt = df["date"].to_numpy().astype("datetime64[D]").astype(np.int64)
    M = np.full((len(df), n_runs), np.nan)
    W = np.zeros((len(df), n_runs))
    base = v3b.run_weights(n_runs)
    for r in range(len(df)):
        b = bounds.get(hid[r])
        if b is None:
            continue
        c = b[0] + int(np.searchsorted(vd[b[0]:b[1]], dt[r], side="left"))
        lo = max(b[0], c - n_runs)
        k = c - lo
        if k <= 0:
            continue
        seg = slice(lo, c)
        M[r, :k] = vv[seg][::-1]
        W[r, :k] = base[:k] * np.where(vl[seg][::-1], v3b.LOW_CONF_WEIGHT, 1.0)
    out = v3b.weighted_huber(M, W, v3b.SCALE)
    return pd.Series(out, index=df.index)
