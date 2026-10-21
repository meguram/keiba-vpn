"""tp-58: 交互作用を表現できる NN メンバー（MLP 対照 / DCN-v2 クロス層 / FM 二次項 / DeepFM）。

出力は 3 本のロジット（勝ち・2着内・3着内）。勝ちはレース内ソフトマックス（リストワイズ）、2着内・3着内は二値 CE。
レースは (R×maxN) の行番号行列（-1 で詰め物）で扱う。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VARIANTS = ("mlp", "dcn", "fm2", "deepfm")


def race_index(race_codes: np.ndarray, max_n: int = 20):
    """行がレースごとに連続している前提で (R×max_n) の行番号行列（-1 は詰め物）と各レースの先頭行を返す。"""
    starts = np.flatnonzero(np.r_[True, race_codes[1:] != race_codes[:-1]])
    ends = np.r_[starts[1:], len(race_codes)]
    cnt = np.minimum(ends - starts, max_n)
    idx = np.full((len(starts), max_n), -1, dtype="int64")
    for r, (s, c) in enumerate(zip(starts, cnt)):
        idx[r, :c] = np.arange(s, s + c)
    return idx, starts


class CrossNet(nn.Module):
    """DCN-v2: x_{l+1} = x0 ⊙ (W_l x_l + b_l) + x_l"""

    def __init__(self, d: int, layers: int = 3):
        super().__init__()
        self.lin = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])

    def forward(self, x0):
        x = x0
        for w in self.lin:
            x = x0 * w(x) + x
        return x


class FMTerm(nn.Module):
    """各ヘッドごとの二次項 ½Σ_k[(xV)² − x²V²]。出力 (…×heads)。"""

    def __init__(self, d: int, k: int = 16, heads: int = 3):
        super().__init__()
        self.k, self.heads = k, heads
        self.V = nn.Parameter(torch.randn(d, k * heads) * 0.01)

    def forward(self, x):
        xv = x @ self.V
        x2v2 = (x * x) @ (self.V * self.V)
        t = 0.5 * (xv * xv - x2v2)
        return t.view(*t.shape[:-1], self.heads, self.k).sum(-1)


class Net(nn.Module):
    def __init__(self, d: int, variant: str, hidden=(512, 256), drop: float = 0.2, cross_layers: int = 3, fm_k: int = 16, in_drop: float = 0.0):
        super().__init__()
        self.in_drop = nn.Dropout(in_drop)
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.lin = nn.Linear(d, 3)
        self.cross = CrossNet(d, cross_layers) if variant == "dcn" else None
        self.fm = FMTerm(d, fm_k, 3) if variant in ("fm2", "deepfm") else None
        self.deep = None
        if variant in ("mlp", "dcn", "deepfm"):
            layers, prev = [], d
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(drop)]
                prev = h
            self.deep = nn.Sequential(*layers)
            self.out_deep = nn.Linear(prev, 3)
        if variant == "dcn":
            self.out_cross = nn.Linear(d, 3)

    def forward(self, x):
        x = self.in_drop(x)
        out = self.lin(x)
        if self.deep is not None:
            out = out + self.out_deep(self.deep(x))
        if self.cross is not None:
            out = out + self.out_cross(self.cross(x))
        if self.fm is not None:
            out = out + self.fm(x)
        return out


def listwise_loss(logits, idx, y_win, y_top2, y_top3, valid):
    """logits: (N×3) 全行。idx: (B×maxN) 行番号 / -1。y_*: (N,) float、valid: (N,) bool（着順が有効な行）。"""
    m = idx >= 0
    safe = idx.clamp(min=0)
    lg = logits[safe]
    ok = m & valid[safe]
    w = torch.where(ok, lg[..., 0], torch.full_like(lg[..., 0], -1e9))
    logp = F.log_softmax(w, dim=1)
    yw = y_win[safe] * ok
    has_win = yw.sum(1) > 0
    win = -(logp * yw).sum(1)[has_win].mean() if has_win.any() else lg.sum() * 0.0
    b2 = F.binary_cross_entropy_with_logits(lg[..., 1][ok], y_top2[safe][ok]) if ok.any() else lg.sum() * 0.0
    b3 = F.binary_cross_entropy_with_logits(lg[..., 2][ok], y_top3[safe][ok]) if ok.any() else lg.sum() * 0.0
    return win + 0.5 * b2 + 0.5 * b3
