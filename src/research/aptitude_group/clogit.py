"""着順の段ごとの条件付きロジット（重み付き・Newton 法・開催の束ねのサンドイッチの分散）。

行は「段の選択肢の 1 頭」。set は選択の組（レース × 段）で、行は set の順に連続して並ぶこと。
"""
from __future__ import annotations

import numpy as np


def _set_sum(x: np.ndarray, starts: np.ndarray) -> np.ndarray:
    return np.add.reduceat(x, starts, axis=0)


def probs(X: np.ndarray, b: np.ndarray, starts: np.ndarray, sid: np.ndarray) -> np.ndarray:
    u = X @ b
    m = np.maximum.reduceat(u, starts)
    e = np.exp(u - m[sid])
    return e / _set_sum(e, starts)[sid]


def fit(X: np.ndarray, chosen: np.ndarray, sid: np.ndarray, w_set: np.ndarray | None = None, b0: np.ndarray | None = None,
        iters: int = 30, tol: float = 1e-9, ridge: float = 1e-6) -> tuple[np.ndarray, np.ndarray, float]:
    """返す: (係数, ヘッセ行列の逆〔正〕, 対数尤度)。sid は 0.. の連番で行の順に並ぶ。"""
    starts = np.r_[0, np.flatnonzero(np.diff(sid)) + 1]
    S = len(starts)
    w_set = np.ones(S) if w_set is None else w_set
    w = w_set[sid]
    b = np.zeros(X.shape[1]) if b0 is None else b0.copy()
    if not np.all(np.add.reduceat(chosen.astype(int), starts) == 1):
        raise ValueError("どの選択の組にも、選ばれた行がちょうど 1 つ要る")
    ci = np.flatnonzero(chosen)

    def loglik(bb):
        u = X @ bb
        mx = np.maximum.reduceat(u, starts)
        return float((w_set * (u[ci] - (mx + np.log(_set_sum(np.exp(u - mx[sid]), starts))))).sum())
    ll_old = loglik(b)
    for _ in range(iters):
        p = probs(X, b, starts, sid)
        xbar = _set_sum(p[:, None] * X, starts)
        g = (w_set[:, None] * (X[ci] - xbar)).sum(0)
        Hm = (X * (w * p)[:, None]).T @ X - (xbar * w_set[:, None]).T @ xbar + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(Hm, g)
        t = 1.0
        while True:                                  # 刻みを縮める（尤度が下がる一歩は取らない。critic の 9）
            ll = loglik(b + t * step)
            if ll >= ll_old - 1e-12 * max(1.0, abs(ll_old)):   # 相対の許し（丸めの誤差で警告が誤って出ないように・critic 4 回目）
                break
            t *= 0.5
            if t < 1e-4:                             # 上がる一歩が無い = 止まる（下がる一歩は取らない。critic 2 回目の 10）
                ll = ll_old
                break
        if ll == ll_old and t < 1e-4:
            import warnings
            warnings.warn("clogit.fit: 尤度の上がる一歩が見つからず止まった（critic 3 回目の 10）")
            break
        b = b + t * step
        done = abs(ll - ll_old) < tol * max(1.0, abs(ll))
        ll_old = ll
        if done:
            break
    else:
        import warnings
        warnings.warn("clogit.fit: 繰り返しの上限で止まった（収束していないかもしれない）")
    ll = ll_old
    p = probs(X, b, starts, sid)
    xbar = _set_sum(p[:, None] * X, starts)
    Hm = (X * (w * p)[:, None]).T @ X - (xbar * w_set[:, None]).T @ xbar + ridge * np.eye(X.shape[1])
    return b, np.linalg.inv(Hm), ll


def cluster_se(X: np.ndarray, chosen: np.ndarray, sid: np.ndarray, b: np.ndarray, Hinv: np.ndarray, cl_set: np.ndarray) -> np.ndarray:
    """開催の束ねのサンドイッチの標準誤差。cl_set は set ごとの束の番号。"""
    starts = np.r_[0, np.flatnonzero(np.diff(sid)) + 1]
    p = probs(X, b, starts, sid)
    xbar = _set_sum(p[:, None] * X, starts)
    sc = X[np.flatnonzero(chosen)] - xbar
    G = np.zeros((cl_set.max() + 1, X.shape[1]))
    np.add.at(G, cl_set, sc)
    V = Hinv @ (G.T @ G) @ Hinv
    return np.sqrt(np.diag(V))


def sample(X: np.ndarray, b: np.ndarray, sid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """各 set から 1 行を、当てはめた確率で引く。返すのは選ばれた行の番号（set の順）。"""
    starts = np.r_[0, np.flatnonzero(np.diff(sid)) + 1]
    p = probs(X, b, starts, sid)
    c = np.cumsum(p)
    base = np.r_[0.0, c[starts[1:] - 1]]
    r = base + rng.random(len(starts)) * (c[np.r_[starts[1:] - 1, len(c) - 1]] - base)
    idx = np.searchsorted(c, r, side="left")
    ends = np.r_[starts[1:] - 1, len(c) - 1]
    return np.clip(idx, starts, ends)
