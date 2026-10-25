"""全券種・縛りなしの『パターン探索』: 回収率 M 倍（既定 5 倍）に届く買い目の持ち方のうち、どれか 1 つ当たる確率が最大のものを選ぶ。

買い目 i = 券種×組。購入額 s_i = ceil(M·B·k / 想定オッズ_i)（100 円単位）なので、どの買い目が当たっても払戻 ≥ M·B。
パターン = 買い目の部分集合 S（Σ s_i ≤ B）。的中率は、上位 3 着の全順列（n(n-1)(n-2) 通り）の確率の和で、S が覆う順列の和集合（重なりは厳密に 1 回だけ数える）。
期待回収率 = Σ p_i·s_i·odds_i / Σ s_i が τ 以上のパターンだけを認める（想定オッズの誤差は k で保守化）。
探索は費用対効果 p_i/s_i の上位 N 件から、サイズ K 以下の部分集合をビームサーチ（幅 beam）で組み立てる。厳密な全列挙との差は検証側で確認する。
入力は配列のみ（データの読み書きはしない）。券種の番号: 0 単勝 / 1 複勝 / 2 馬連 / 3 ワイド / 4 馬単 / 5 3連複 / 6 3連単。組は 0 始まりの馬 index のタプル。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np

UNIT = 100


@dataclass(frozen=True)
class PatternParams:
    mult: float = 5.0
    k: float = 2.0
    tau: float = 1.0          # パターンの期待回収率の下限
    bet_ev_min: float = 0.9   # 買い目単位の p×odds の下限（候補の足切り）
    top_n: int = 50
    max_k: int = 5
    beam: int = 12
    budget: int = 10_000


class Perms:
    """頭数 n の上位 3 着の全順列と、券種×組ごとの『当たる順列』の真偽配列。"""

    def __init__(self, n: int) -> None:
        self.n = n
        idx = np.array(list(permutations(range(n), 3)), dtype=np.int16)
        self.a, self.b, self.c = idx[:, 0], idx[:, 1], idx[:, 2]
        self._has = {}
        self._first = {}

    def has(self, x: int) -> np.ndarray:
        if x not in self._has:
            self._has[x] = (self.a == x) | (self.b == x) | (self.c == x)
        return self._has[x]

    def mask(self, bt: int, combo: tuple[int, ...]) -> np.ndarray:
        a, b, c = self.a, self.b, self.c
        if bt == 0:
            return a == combo[0]
        if bt == 1:
            return self.has(combo[0]) if self.n >= 8 else ((a == combo[0]) | (b == combo[0]))
        x, y = combo[0], combo[1]
        if bt == 2:
            return ((a == x) & (b == y)) | ((a == y) & (b == x))
        if bt == 3:
            return self.has(x) & self.has(y)
        if bt == 4:
            return (a == x) & (b == y)
        z = combo[2]
        if bt == 5:
            return self.has(x) & self.has(y) & self.has(z)
        return (a == x) & (b == y) & (c == z)


def required_stake(odds: np.ndarray, prm: PatternParams) -> np.ndarray:
    o = np.asarray(odds, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = prm.mult * prm.budget * prm.k / o
    out = np.ceil(raw / UNIT - 1e-9) * UNIT
    return np.where(np.isfinite(out) & (o > 0), np.maximum(out, UNIT), np.inf)


def select_pattern(bt: np.ndarray, combo: list[tuple[int, ...]], p: np.ndarray, odds: np.ndarray, perm_p: np.ndarray,
                   perms: Perms, prm: PatternParams, exhaustive: bool = False) -> dict:
    """戻り値: {idx, stakes, hit_prob, ev}。条件を満たすパターンが無ければ idx は空。"""
    p = np.asarray(p, dtype=float)
    o = np.asarray(odds, dtype=float)
    stake = required_stake(o, prm)
    ok = np.isfinite(stake) & (stake <= prm.budget) & (p > 0) & (p * o >= prm.bet_ev_min)
    cand = np.flatnonzero(ok)
    if cand.size == 0:
        return {"idx": np.empty(0, dtype=int), "stakes": np.empty(0), "hit_prob": 0.0, "ev": None}
    cand = cand[np.argsort(-(p[cand] / stake[cand]), kind="stable")][: prm.top_n]
    masks = [perms.mask(int(bt[j]), combo[j]) for j in cand]
    pp = np.asarray(perm_p, dtype=float)
    best = {"idx": np.empty(0, dtype=int), "stakes": np.empty(0), "hit_prob": 0.0, "ev": None}
    if exhaustive:
        from itertools import combinations as comb
        for K in range(1, min(prm.max_k, 3) + 1):
            for sub in comb(range(len(cand)), K):
                sp = float(stake[cand[list(sub)]].sum())
                if sp > prm.budget:
                    continue
                ev = float(sum(p[cand[i]] * stake[cand[i]] * o[cand[i]] for i in sub) / sp)
                if ev < prm.tau:
                    continue
                u = masks[sub[0]].copy()
                for i in sub[1:]:
                    u |= masks[i]
                hp = float(pp[u].sum())
                if hp > best["hit_prob"]:
                    best = {"idx": cand[list(sub)], "stakes": stake[cand[list(sub)]], "hit_prob": hp, "ev": ev}
        return best
    # ビームサーチ: 状態 = (和集合マスク, 選んだ位置のタプル, 使った額, 期待払戻の合計)
    states = [(np.zeros(len(pp), dtype=bool), (), 0.0, 0.0)]
    for _ in range(prm.max_k):
        nxt = []
        for um, chosen, spent, enum in states:
            start = chosen[-1] + 1 if chosen else 0
            for i in range(start, len(cand)):
                j = cand[i]
                s2 = spent + stake[j]
                if s2 > prm.budget:
                    continue
                u2 = um | masks[i]
                hp = float(pp[u2].sum())
                nxt.append((hp, u2, chosen + (i,), s2, enum + p[j] * stake[j] * o[j]))
        if not nxt:
            break
        nxt.sort(key=lambda t: -t[0])
        states = []
        for hp, u2, ch, s2, en in nxt[: prm.beam]:
            states.append((u2, ch, s2, en))
            if en / s2 >= prm.tau and hp > best["hit_prob"]:
                sel = cand[list(ch)]
                best = {"idx": sel, "stakes": stake[sel], "hit_prob": hp, "ev": float(en / s2)}
    return best
