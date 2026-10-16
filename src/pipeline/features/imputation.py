"""欠損値のモデル補完（低ランク近似による反復補完）と RankGauss 変換。tp-17 用。

どちらも fit は学習に使ってよい期間の行だけで行い、transform は任意の行に適用する（行ごとの計算なので先読みは入らない）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class SoftImputer:
    """列ごとに標準化 → 低ランク（rank）の主成分空間へ射影して欠損を埋める、を繰り返す（Soft-Impute 系）。"""

    def __init__(self, rank: int = 8, iters: int = 15, clip: float = 5.0):
        self.rank, self.iters, self.clip = rank, iters, clip

    def _z(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Z = (X - self.mu) / self.sd
        Z = np.clip(Z, -self.clip, self.clip)
        M = np.isnan(Z)
        return np.where(M, 0.0, Z), M

    def fit(self, X: np.ndarray) -> "SoftImputer":
        X = np.asarray(X, dtype="float64")
        self.mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
        self.mu = np.where(np.isfinite(self.mu), self.mu, 0.0)
        self.sd = np.where((sd > 0) & np.isfinite(sd), sd, 1.0)
        Z, M = self._z(X)
        k = min(self.rank, Z.shape[1] - 1)
        for _ in range(self.iters):
            _, S, Vt = np.linalg.svd(Z, full_matrices=False)
            V = Vt[:k].T
            R = (Z @ V) @ V.T
            Z = np.where(M, R, Z)
        self.V = V
        return self

    def transform(self, X: np.ndarray, iters: int = 10) -> np.ndarray:
        X = np.asarray(X, dtype="float64")
        Z, M = self._z(X)
        for _ in range(iters):
            R = (Z @ self.V) @ self.V.T
            Z = np.where(M, R, Z)
        out = Z * self.sd + self.mu
        return np.where(np.isnan(X), out, X)


def impute_frame(X: pd.DataFrame, fit_mask: np.ndarray, cols: list[str], rank: int = 8, flag_threshold: float = 0.02) -> pd.DataFrame:
    """cols を補完した表を返す（それ以外の列はそのまま）。欠損率が flag_threshold を超える列には、元の欠損を示す miss_<列> を足す。"""
    out = X.copy()
    A = X[cols].to_numpy(dtype="float64")
    imp = SoftImputer(rank=rank).fit(A[fit_mask])
    B = imp.transform(A)
    flags = {}
    for j, c in enumerate(cols):
        if np.isnan(A[:, j]).mean() > flag_threshold:
            flags[f"miss_{c}"] = np.isnan(A[:, j]).astype("float32")
    for j, c in enumerate(cols):
        out[c] = B[:, j].astype(X[c].dtype if str(X[c].dtype).startswith("float") else "float64")
    if flags:
        out = pd.concat([out, pd.DataFrame(flags, index=X.index)], axis=1)
    return out


def rank_gauss(X: pd.DataFrame, fit_mask: np.ndarray, cols: list[str], n_quantiles: int = 1000) -> pd.DataFrame:
    """fit 期間の分位点で正規分布に写す（外れ値は端に張り付く）。欠損は欠損のまま。"""
    from sklearn.preprocessing import QuantileTransformer

    out = X.copy()
    A = X[cols].to_numpy(dtype="float64")
    qt = QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal", subsample=10**9, random_state=0)
    qt.fit(A[fit_mask])
    B = qt.transform(A)
    for j, c in enumerate(cols):
        out[c] = B[:, j].astype("float32")
    return out
