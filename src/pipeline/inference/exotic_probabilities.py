"""馬連・ワイド・馬単・3連複の的中確率を、勝率・連対率・複勝率から一貫して計算する。

設計の背景は docs/html/modeling/chapters/exotic_bet_probabilities.html。要点:

  1. 着順の生成モデルは Plackett–Luce（= Harville 1973）に、着順ステージごとの減衰指数 α_r を付けた
     Benter (1994) / Lo & Bacon-Shone 型の「割引 Harville」:

         P(i が 1 着) = λ_i / Σ_k λ_k                     （α_1 = 1。λ は勝率そのもの）
         P(j が 2 着 | i が 1 着) = λ_j^α2 / Σ_{k≠i} λ_k^α2
         P(k が 3 着 | i, j が 1-2 着) = λ_k^α3 / Σ_{k∉{i,j}} λ_k^α3

     α = 1 なら Harville。Harville は 2・3 着の確率を人気馬に寄せすぎる（穴馬の連対・複勝を過小評価する）
     ことが知られており、α < 1 にするとそれが緩む。
  2. α2, α3 は文献の定数ではなく、**その馬のモデルが出す連対率・複勝率に合うように 1 レースごとに当てはめる**
     （周辺確率の整合）。これで「勝率・連対率・複勝率」と「馬連・ワイド・3連複」が同じ分布から出る。
  3. 連対率・複勝率が無いときは α を既定値にする（``config/exotic_alpha.json`` があればそれ、無ければ Harville = 1.0）。
     既定値は学習済みモデルの実績から ``validate_exotic_probabilities predictions --write-config`` で作る。

出力は全て同じ着順分布から導く（総和の制約が自動で満たされる）:
    馬単 P(i→j)             = P2[i,j]
    馬連 P({i,j})           = P2[i,j] + P2[j,i]           （総和 1）
    3連複 P({i,j,k})        = 6 通りの順列の和              （総和 1）
    ワイド P({i,j}) = Σ_k 3連複 P({i,j,k})                （総和 3）
    連対率 / 複勝率の周辺確率も同じ分布から出す
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np

ALPHA_MIN, ALPHA_MAX = 0.2, 1.5
ALPHA_CONFIG = Path(__file__).resolve().parents[3] / "config" / "exotic_alpha.json"


@lru_cache(maxsize=1)
def load_default_alpha() -> tuple[float, float]:
    """連対率・複勝率が無いときに使う α2, α3 の既定値。

    ``config/exotic_alpha.json``（学習済みモデルの実績から ``validate_exotic_probabilities predictions --write-config``
    で作る）があればそれ、無ければ Harville（1.0, 1.0）。範囲外・壊れた値は無視して Harville。
    """
    try:
        d = json.loads(ALPHA_CONFIG.read_text(encoding="utf-8"))
        a2, a3 = float(d["alpha2"]), float(d["alpha3"])
        if ALPHA_MIN <= a2 <= ALPHA_MAX and ALPHA_MIN <= a3 <= ALPHA_MAX:
            return a2, a3
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return 1.0, 1.0
_EPS = 1e-12

from src.utils import cache_epoch as _cache_epoch  # noqa: E402

_cache_epoch.register("exotic_probabilities.alpha", lambda: load_default_alpha.cache_clear())


def _normalize_win(win: Sequence[float]) -> np.ndarray:
    w = np.clip(np.asarray(win, dtype=float), _EPS, None)
    return w / w.sum()


def normalize_marginal(values: Sequence[float], total: float) -> np.ndarray:
    """周辺確率（連対率: 総和 2 / 複勝率: 総和 3）を、各 ≤ 1 を保ったまま総和に合わせる（反復スケーリング）。"""
    v = np.clip(np.asarray(values, dtype=float), _EPS, 1.0)
    total = min(total, len(v) - 1e-9)
    for _ in range(50):
        v = np.clip(v * (total / v.sum()), _EPS, 1.0)
        if abs(v.sum() - total) < 1e-9:
            break
    return v


@dataclass
class RankDistribution:
    """割引 Harville の着順分布。n は出走頭数（最大 18 程度を想定）。"""

    win: np.ndarray
    alpha2: float = 1.0
    alpha3: float = 1.0
    lam2: np.ndarray | None = None  # 2 着目の強さ（馬ごと）。None なら win**alpha2
    lam3: np.ndarray | None = None  # 3 着目の強さ（馬ごと）。None なら win**alpha3
    _p2: np.ndarray = field(init=False, repr=False)  # [i, j] = i が 1 着, j が 2 着
    _p3: np.ndarray = field(init=False, repr=False)  # [i, j, k] = i, j, k の順

    def __post_init__(self) -> None:
        w = _normalize_win(self.win)
        self.win = w
        n = len(w)
        l2 = np.clip(np.asarray(self.lam2, dtype=float), _EPS, None) if self.lam2 is not None else w ** self.alpha2
        l3 = np.clip(np.asarray(self.lam3, dtype=float), _EPS, None) if self.lam3 is not None else w ** self.alpha3
        s2, s3 = l2.sum(), l3.sum()
        p2 = np.zeros((n, n))
        for i in range(n):
            d = s2 - l2[i]
            if d > _EPS:
                p2[i] = w[i] * l2 / d
            p2[i, i] = 0.0
        p3 = np.zeros((n, n, n)) if n >= 3 else np.zeros((n, n, 0))
        if n >= 3:
            den = s3 - l3[:, None] - l3[None, :]  # [i, j]
            with np.errstate(divide="ignore", invalid="ignore"):
                cond = np.where(den[:, :, None] > _EPS, l3[None, None, :] / den[:, :, None], 0.0)
            p3 = p2[:, :, None] * cond
            idx = np.arange(n)
            p3[idx, idx, :] = 0.0
            p3[idx, :, idx] = 0.0
            p3[:, idx, idx] = 0.0
        self._p2, self._p3 = p2, p3

    @property
    def n(self) -> int:
        return len(self.win)

    # ── 周辺確率 ──
    def top2(self) -> np.ndarray:
        return self.win + self._p2.sum(axis=0)

    def top3(self) -> np.ndarray:
        if self.n < 3:
            return np.ones(self.n)
        return self.top2() + self._p3.sum(axis=(0, 1))

    # ── 組合せ確率（馬番ではなく 0 始まりの出走順位置 index で引く） ──
    def umatan(self, i: int, j: int) -> float:
        return float(self._p2[i, j])

    def umaren(self, i: int, j: int) -> float:
        return float(self._p2[i, j] + self._p2[j, i])

    def sanrentan(self, i: int, j: int, k: int) -> float:
        """i が 1 着、j が 2 着、k が 3 着になる確率（3連単）。"""
        return float(self._p3[i, j, k]) if self.n >= 3 else 0.0

    def sanrenpuku(self, i: int, j: int, k: int) -> float:
        p = self._p3
        return float(p[i, j, k] + p[i, k, j] + p[j, i, k] + p[j, k, i] + p[k, i, j] + p[k, j, i])

    def wide(self, i: int, j: int) -> float:
        """i と j がともに 3 着以内に入る確率（= 3 頭目 k を問わない 3連複の和）。"""
        if self.n < 3:
            return 1.0
        return float(sum(self.sanrenpuku(i, j, k) for k in range(self.n) if k != i and k != j))

    def all_umaren(self) -> dict[tuple[int, int], float]:
        return {(i, j): self.umaren(i, j) for i, j in combinations(range(self.n), 2)}

    def all_wide(self) -> dict[tuple[int, int], float]:
        n = self.n
        if n < 3:
            return {(0, 1): 1.0} if n == 2 else {}
        trio = self._trio_sums()
        out = {}
        for i, j in combinations(range(n), 2):
            out[(i, j)] = float(sum(trio[tuple(sorted((i, j, k)))] for k in range(n) if k != i and k != j))
        return out

    def _trio_sums(self) -> dict[tuple[int, int, int], float]:
        return {t: self.sanrenpuku(*t) for t in combinations(range(self.n), 3)}

    def all_sanrenpuku(self) -> dict[tuple[int, int, int], float]:
        return self._trio_sums() if self.n >= 3 else {}


def _golden_min(f, lo: float, hi: float, tol: float = 1e-4) -> float:
    g = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return (a + b) / 2


@dataclass
class FitReport:
    alpha2: float
    alpha3: float
    top2_rmse_before: float | None = None  # α=1（Harville）のときの周辺確率誤差
    top2_rmse_after: float | None = None
    top3_rmse_before: float | None = None
    top3_rmse_after: float | None = None
    used_top2: bool = False
    used_top3: bool = False


def fit_rank_distribution(
    win: Sequence[float],
    top2: Sequence[float] | None = None,
    top3: Sequence[float] | None = None,
    *,
    default_alpha2: float | None = None,
    default_alpha3: float | None = None,
) -> tuple[RankDistribution, FitReport]:
    """勝率（必須）と連対率・複勝率（任意）に周辺確率が合うよう α2, α3 を当てはめた着順分布を返す。

    - top2 があれば α2 を連対率の二乗誤差最小で決める（α3 は 1 で仮置き。連対率は α3 に依存しない）
    - top3 があれば、その α2 のもとで α3 を複勝率の二乗誤差最小で決める
    - top2 が無く top3 だけあるときは α2 = α3 = α を複勝率に合わせる（2 着目と 3 着目のなだらかさを揃える近似）
    - どちらも無ければ既定値（Harville）
    """
    if default_alpha2 is None or default_alpha3 is None:
        d2, d3 = load_default_alpha()
        default_alpha2 = d2 if default_alpha2 is None else default_alpha2
        default_alpha3 = d3 if default_alpha3 is None else default_alpha3
    w = _normalize_win(win)
    n = len(w)
    rep = FitReport(alpha2=default_alpha2, alpha3=default_alpha3)
    if n < 3:
        return RankDistribution(w, default_alpha2, default_alpha3), rep

    t2 = normalize_marginal(top2, 2.0) if top2 is not None else None
    t3 = normalize_marginal(top3, 3.0) if top3 is not None else None
    base = RankDistribution(w, 1.0, 1.0)
    if t2 is not None:
        rep.top2_rmse_before = float(np.sqrt(np.mean((base.top2() - t2) ** 2)))
    if t3 is not None:
        rep.top3_rmse_before = float(np.sqrt(np.mean((base.top3() - t3) ** 2)))

    a2, a3 = default_alpha2, default_alpha3
    if t2 is not None:
        a2 = _golden_min(lambda a: float(np.sum((RankDistribution(w, a, 1.0).top2() - t2) ** 2)), ALPHA_MIN, ALPHA_MAX)
        rep.used_top2 = True
    if t3 is not None:
        if t2 is not None:
            a3 = _golden_min(lambda a: float(np.sum((RankDistribution(w, a2, a).top3() - t3) ** 2)), ALPHA_MIN, ALPHA_MAX)
        else:
            a = _golden_min(lambda a: float(np.sum((RankDistribution(w, a, a).top3() - t3) ** 2)), ALPHA_MIN, ALPHA_MAX)
            a2 = a3 = a
        rep.used_top3 = True

    dist = RankDistribution(w, a2, a3)
    rep.alpha2, rep.alpha3 = a2, a3
    if t2 is not None:
        rep.top2_rmse_after = float(np.sqrt(np.mean((dist.top2() - t2) ** 2)))
    if t3 is not None:
        rep.top3_rmse_after = float(np.sqrt(np.mean((dist.top3() - t3) ** 2)))
    return dist, rep


def fit_marginal_distribution(
    win: Sequence[float],
    top2: Sequence[float],
    top3: Sequence[float],
    *,
    iters: int = 60,
    tol: float = 1e-7,
) -> tuple[RankDistribution, FitReport]:
    """勝率・連対率・複勝率の 3 つの周辺確率に**馬ごとに**厳密に合わせた着順分布（周辺確率一致の Plackett–Luce）。

    割引 Harville（``fit_rank_distribution``）は 2・3 着目の強さを w^α の 1 パラメータで決めるため、
    馬ごとの連対率・複勝率のずれ（粘る馬・止まる馬）を表せない。ここでは 2 着目・3 着目の強さ λ2_j, λ3_j を馬ごとの自由度にし、
    P(j が 2 着以内)=連対率_j、P(j が 3 着以内)=複勝率_j となるよう乗法的な反復で求める（不動点: λ ← λ·(目標/現在)^η）。
    勝率は 1 着目で厳密に保たれる。収束しないレースでも、反復中で最も誤差の小さい解を返す。
    """
    w = _normalize_win(win)
    n = len(w)
    t2 = normalize_marginal(top2, 2.0)
    t3 = normalize_marginal(top3, 3.0)
    t2 = np.maximum(t2, w)  # 連対率 ≥ 勝率、複勝率 ≥ 連対率 の整合
    t3 = np.maximum(t3, t2)
    rep = FitReport(alpha2=1.0, alpha3=1.0, used_top2=True, used_top3=True)
    base = RankDistribution(w, 1.0, 1.0)
    rep.top2_rmse_before = float(np.sqrt(np.mean((base.top2() - t2) ** 2)))
    rep.top3_rmse_before = float(np.sqrt(np.mean((base.top3() - t3) ** 2)))
    if n < 3:
        return base, rep
    l2, l3 = w.copy(), w.copy()
    best = (np.inf, l2.copy(), l3.copy())
    eta = 0.7
    for _ in range(iters):
        d = RankDistribution(w, 1.0, 1.0, lam2=l2, lam3=l3)
        e2, e3 = d.top2(), d.top3()
        err = float(np.sqrt(np.mean((e2 - t2) ** 2) + np.mean((e3 - t3) ** 2)))
        if err < best[0]:
            best = (err, l2.copy(), l3.copy())
        if err < tol:
            break
        # 2 着目の強さは連対率で、3 着目の強さは複勝率で合わせる（2 着目の分布が決まってから 3 着目を直す）
        l2 = l2 * (np.maximum(t2 - w, _EPS) / np.maximum(e2 - w, _EPS)) ** eta
        l3 = l3 * (np.maximum(t3 - t2, _EPS) / np.maximum(e3 - e2, _EPS)) ** eta
        l2 = np.clip(l2 / l2.max(), 1e-9, None)
        l3 = np.clip(l3 / l3.max(), 1e-9, None)
    _, l2, l3 = best
    dist = RankDistribution(w, 1.0, 1.0, lam2=l2, lam3=l3)
    rep.top2_rmse_after = float(np.sqrt(np.mean((dist.top2() - t2) ** 2)))
    rep.top3_rmse_after = float(np.sqrt(np.mean((dist.top3() - t3) ** 2)))
    return dist, rep


DEFAULT_BLEND_THETA = 0.6


def load_blend_theta() -> float:
    """周辺確率一致（θ=1）と共通 α の Harville 型（θ=0）の混合比。``config/exotic_alpha.json`` の ``blend_theta``、無ければ 0.6。"""
    try:
        t = float(json.loads(ALPHA_CONFIG.read_text(encoding="utf-8")).get("blend_theta", DEFAULT_BLEND_THETA))
        return t if 0.0 <= t <= 1.0 else DEFAULT_BLEND_THETA
    except (OSError, ValueError, TypeError):
        return DEFAULT_BLEND_THETA


def fit_blend_distribution(
    win: Sequence[float],
    top2: Sequence[float],
    top3: Sequence[float],
    *,
    theta: float | None = None,
    alpha2: float | None = None,
    alpha3: float | None = None,
) -> tuple[RankDistribution, FitReport]:
    """周辺確率一致（``fit_marginal_distribution``）と共通 α の割引 Harville の、2・3 着目の強さの対数線形ブレンド。

    log λ_r = θ · log λ_r(周辺一致) + (1−θ) · α_r · log 勝率。θ=1 で周辺一致、θ=0 で共通 α の割引 Harville。
    2022〜2024 年の OOF 約 1 万レースの検証（docs/html/modeling/chapters/exotic_bet_probabilities.html §10）で、
    複勝・馬連・ワイド・3連複は周辺一致寄り、馬単・3連単は共通 α 寄りが良く、θ=0.5〜0.7 は全券種で割引 Harville
    （レースごと当てはめ）より対数スコアが良かった。単一の分布から全券種を出すため θ=0.6 を既定にしている。
    """
    d2, d3 = load_default_alpha()
    a2 = d2 if alpha2 is None else alpha2
    a3 = d3 if alpha3 is None else alpha3
    th = load_blend_theta() if theta is None else float(theta)
    base, rep = fit_marginal_distribution(win, top2, top3)
    if base.n < 3:
        return base, rep
    w = base.win
    lw = np.log(np.clip(w, _EPS, None))
    l2 = np.exp(th * np.log(np.clip(base.lam2, _EPS, None)) + (1 - th) * a2 * lw)
    l3 = np.exp(th * np.log(np.clip(base.lam3, _EPS, None)) + (1 - th) * a3 * lw)
    dist = RankDistribution(w, 1.0, 1.0, lam2=l2, lam3=l3)
    t2 = np.maximum(normalize_marginal(top2, 2.0), w)
    t3 = np.maximum(normalize_marginal(top3, 3.0), t2)
    rep = FitReport(alpha2=a2, alpha3=a3, used_top2=True, used_top3=True,
                    top2_rmse_before=rep.top2_rmse_before, top3_rmse_before=rep.top3_rmse_before,
                    top2_rmse_after=float(np.sqrt(np.mean((dist.top2() - t2) ** 2))),
                    top3_rmse_after=float(np.sqrt(np.mean((dist.top3() - t3) ** 2))))
    return dist, rep
