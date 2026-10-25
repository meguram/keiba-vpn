"""目標払戻つきの買い目配分（500% 設計・トリガミ回避）。

1レースの予算 B（既定 10,000 円）で、どれか 1 つの買い目が的中すれば、レース全体の払戻が M×B 以上になるように
各買い目の購入額を決める。買い目 i の必要購入額は s_i = ceil(M·B·k / 想定オッズ_i)（100 円単位、k は想定オッズの誤差に対する余裕）。
制約を満たす買い目の集合は Σ s_i ≤ B。M≥1 ならトリガミ（的中しても払戻が購入額を下回る）は起きない設計になる。
当たったときの払戻水準を固定するだけで、期待値（回収率 100% 超）を保証するものではない。

入力は配列（的中確率 p、想定オッズ odds）だけで、データの読み書きはしない。評価は src/scripts/data/portfolio_target_eval.py。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

UNIT = 100
DEFAULT_BUDGET = 10_000


@dataclass(frozen=True)
class TargetParams:
    mult: float = 5.0          # 目標倍率 M（払戻 ≥ M×予算）
    k: float = 1.0             # 想定オッズの誤差に対する余裕（必要購入額を k 倍）
    tau: float = 0.0           # 想定の期待値 p×odds の下限（0 なら的中確率だけで選ぶ）
    max_bets: int | None = None  # 買い目数の上限（1 なら 1 点買い）
    budget: int = DEFAULT_BUDGET


def required_stake(odds: np.ndarray, mult: float, budget: int, k: float = 1.0, unit: int = UNIT) -> np.ndarray:
    """目標払戻 M×B に届く最小の購入額（unit 円単位に切り上げ）。オッズが無効（<=1 以下で届かない/NaN）なら inf。"""
    o = np.asarray(odds, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = mult * budget * k / o
    out = np.ceil(raw / unit - 1e-9) * unit
    out = np.where(np.isfinite(out) & (o > 0), np.maximum(out, unit), np.inf)
    return out


def select_portfolio(p: np.ndarray, odds: np.ndarray, params: TargetParams, allowed: np.ndarray | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
    """買い目の選択。戻り値は (選んだ index, その購入額)。制約を満たす買い目が無ければ空（見送り）。

    目的は『どれかが当たる確率の最大化』を Σp で近似し、予算内で p / 購入額 の大きい順に詰める（貪欲法）。
    1 点買い（max_bets=1）は、予算内に収まる買い目のうち p が最大のもの。
    """
    p = np.asarray(p, dtype=float)
    o = np.asarray(odds, dtype=float)
    stake = required_stake(o, params.mult, params.budget, params.k)
    ok = np.isfinite(stake) & (stake <= params.budget) & np.isfinite(p) & (p > 0)
    if params.tau > 0:
        ok &= (p * o) >= params.tau
    if allowed is not None:
        ok &= allowed
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return np.empty(0, dtype=int), np.empty(0)
    if params.max_bets == 1:
        j = idx[np.argmax(p[idx])]
        return np.array([j]), stake[[j]]
    order = idx[np.argsort(-(p[idx] / stake[idx]), kind="stable")]
    chosen, spent = [], 0.0
    for j in order:
        if spent + stake[j] <= params.budget:
            chosen.append(j)
            spent += stake[j]
            if params.max_bets is not None and len(chosen) >= params.max_bets:
                break
    ch = np.array(chosen, dtype=int)
    return ch, stake[ch]


def settle(stakes: np.ndarray, pay100: np.ndarray, unit: int = UNIT) -> float:
    """払戻の合計。pay100 は 100 円あたりの払戻（外れは 0）。"""
    return float(np.sum(np.asarray(stakes, dtype=float) / unit * np.asarray(pay100, dtype=float)))


def race_outcome(stakes: np.ndarray, pay100: np.ndarray, mult: float, budget: int) -> dict[str, float | bool]:
    spent = float(np.sum(stakes))
    pay = settle(stakes, pay100)
    hit = pay > 0
    return {"spent": spent, "pay": pay, "hit": hit,
            "torigami": bool(hit and pay < spent),
            "reach": bool(hit and pay >= mult * budget - 1e-9)}


def kelly_stakes(p: np.ndarray, odds: np.ndarray, budget: int = DEFAULT_BUDGET, frac: float = 0.25, ev_min: float = 1.05,
                 unit: int = UNIT) -> tuple[np.ndarray, np.ndarray]:
    """従来型の近似（EV 閾値 + 分数ケリー）。ベースライン用。合計が予算を超えたら比例縮小。"""
    p = np.asarray(p, dtype=float)
    o = np.asarray(odds, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where((p * o >= ev_min) & (o > 1), (p * o - 1) / (o - 1), 0.0) * frac
    f = np.where(np.isfinite(f) & (f > 0), f, 0.0)
    if f.sum() > 1:
        f = f / f.sum()
    st = np.floor(f * budget / unit) * unit
    idx = np.flatnonzero(st >= unit)
    return idx, st[idx]


def equal_split(ev_scores: np.ndarray, n: int, budget: int = DEFAULT_BUDGET, thr: float = 1.1, unit: int = UNIT
                ) -> tuple[np.ndarray, np.ndarray]:
    """EV が閾値以上の上位 n 点に、予算を均等に配分（ベースライン）。"""
    e = np.asarray(ev_scores, dtype=float)
    idx = np.flatnonzero(np.isfinite(e) & (e >= thr))
    if idx.size == 0:
        return np.empty(0, dtype=int), np.empty(0)
    idx = idx[np.argsort(-e[idx], kind="stable")][:n]
    each = math.floor(budget / len(idx) / unit) * unit
    if each < unit:
        return np.empty(0, dtype=int), np.empty(0)
    return idx, np.full(len(idx), float(each))
