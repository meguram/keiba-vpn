"""tp-58: 学習済みモデルの OOF 確率に対するスコア検定（1 列を足したときの上乗せの有無を再学習なしで見る）。

- win : レース内ソフトマックス（offset = log p_win）。g = Σ(y−p)z, I = Σ_r[Σ p z² − (Σ p z)²]
- top2/top3 : 二値ロジット（offset = logit p）。g = Σ(y−p)z, I = Σ p(1−p) z²
z = g/√I は帰無（上乗せなし）で標準正規に近い。期間ごと（発見期間・確認期間）に分けて返す。
行はレースごとに連続して並んでいること。
"""

from __future__ import annotations

import numpy as np


class ScoreTest:
    def __init__(self, y, p, races, periods: dict, kind: str):
        if kind not in ("win", "binary"):
            raise ValueError(kind)
        races = np.asarray(races)
        self.starts = np.flatnonzero(np.r_[True, races[1:] != races[:-1]])
        if len(np.unique(races)) != len(self.starts):
            raise ValueError("races must be contiguous")
        self.kind = kind
        y = np.asarray(y, dtype="float64")
        p = np.asarray(p, dtype="float64")
        if kind == "win":
            cnt = np.diff(np.r_[self.starts, len(p)])
            p = p / np.repeat(np.add.reduceat(p, self.starts), cnt)
        self.p, self.y = p, y
        self.names = list(periods)
        self.rowm = np.stack([np.asarray(periods[k], dtype="float64") for k in self.names], 1)
        self.racem = self.rowm[self.starts]
        self.res = ((y - p)[:, None] * self.rowm).astype("float32")
        w = p if kind == "win" else p * (1 - p)
        self.pw = (w[:, None] * self.rowm).astype("float32")
        self.p32 = p.astype("float32")

    def gi(self, Z):
        """Z: N×B → (g, I, rel) 各 P×B。rel は I / Σ p z²（レース内で動かない列を落とすための割合）。"""
        Z = np.asarray(Z, dtype="float64")
        g = self.res.T.astype("float64") @ Z
        tot = self.pw.T.astype("float64") @ (Z * Z)
        if self.kind == "win":
            M = np.add.reduceat(self.p[:, None] * Z, self.starts, axis=0)
            Zc = Z - np.repeat(M, np.diff(np.r_[self.starts, len(Z)]), axis=0)
            I = self.pw.T.astype("float64") @ (Zc * Zc)
        else:
            I = tot
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(tot > 0, I / tot, 0.0)
        return g, I, rel

    def z(self, Z):
        g, I, rel = self.gi(Z)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where((rel > 1e-8) & (I > 1e-12), g / np.sqrt(np.maximum(I, 1e-300)), 0.0)
        return {k: out[i] for i, k in enumerate(self.names)}

    def efficient_z(self, z, nuisance):
        """z (N,) を、nuisance (N×k) の線形項を含めたうえでの有効スコア z（期間ごと）。"""
        V = np.column_stack([np.asarray(z), np.asarray(nuisance)]).astype("float64")
        out = {}
        for i, k in enumerate(self.names):
            m = self.rowm[:, i]
            res = (self.y - self.p) * m
            g = V.T @ res
            w = (self.p if self.kind == "win" else self.p * (1 - self.p)) * m
            I = (V * w[:, None]).T @ V
            if self.kind == "win":
                M = np.add.reduceat(self.p[:, None] * V, self.starts, axis=0)
                rm = self.racem[:, i]
                I = I - (M * rm[:, None]).T @ M
            gz, Izz = g[0], I[0, 0]
            if V.shape[1] > 1:
                Inn = I[1:, 1:] + 1e-9 * np.eye(V.shape[1] - 1)
                sol = np.linalg.solve(Inn, I[1:, 0])
                gz = g[0] - sol @ g[1:]
                Izz = I[0, 0] - I[0, 1:] @ sol
            out[k] = float(gz / np.sqrt(Izz)) if Izz > 1e-12 else 0.0
        return out
