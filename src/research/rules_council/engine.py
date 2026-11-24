"""ルールの会議の共通の道具: 母集団の表・水準の符号・物差し（帯の比）・入れ替えの帰無・合成（Plackett-Luce）。

- 母集団は frame.py の全走の表を、発走前に分かる値（T-60 の再現オッズの範囲・クラス）で絞る。オッズ・人気は切り口にしない。
- 物差し: 単勝は「探す側の同じ確定オッズの帯（× クラス群）の 勝ち ÷ q」の比 ρ を掛けた見込み e = ρ·q。複勝は帯 × 頭数の群の
  「払戻のある 3 着内」の率 e3。市場比は 勝ち ÷ Σe（1 なら同じ帯の全部と同じ）。
- 水準は型ごとの整数の符号（欠けは -1）。数え上げは bincount（one-hot の行列は作らない）。
費用 0 円（手元の表だけ）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.research.rules_council import rc_frame as F

RACE_LEVEL_TYPES = ("H06", "H07", "H08", "H09", "H10")
ERAS = [(2016, 2018), (2019, 2021), (2022, 2024)]
NB = len(F.ODDS_EDGES) - 1
NFG = len(F.FIELD_EDGES) + 1


EXTRA_RACE_DAYS: set[str] = set()    # 土日以外の開催日（祝日の月曜など）を足すときは YYYY-MM-DD で


def stop_window() -> bool:
    """開催日（土日と EXTRA_RACE_DAYS）の 07:30〜17:00 JST は止める（統括役の決まり。重い計算を流さない）。"""
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone(timedelta(hours=9)))
    race_day = now.weekday() >= 5 or now.strftime("%Y-%m-%d") in EXTRA_RACE_DAYS
    return race_day and (7, 30) <= (now.hour, now.minute) < (17, 0)


def check_stop() -> None:
    if stop_window():
        raise SystemExit("開催日の 07:30〜17:00 なので止める（続きは同じ命令で流す）")


@dataclass
class Pop:
    d: pd.DataFrame                  # 母集団の行（full の部分）
    codes: np.ndarray                # 行 × 型 の水準の符号（-1 は欠け）
    types: list[str]
    levels: list[list[str]]          # 型ごとの水準の名前
    full_rows: np.ndarray            # full の中の行番号
    offs: np.ndarray = field(default=None)   # 型ごとの水準の通し番号の先頭

    def __post_init__(self):
        self.offs = np.r_[0, np.cumsum([len(x) for x in self.levels])]

    @property
    def n_levels(self) -> int:
        return int(self.offs[-1])

    def level_name(self, j: int) -> str:
        t = int(np.searchsorted(self.offs, j, side="right") - 1)
        return f"{self.types[t]}={self.levels[t][j - self.offs[t]]}"

    def flat(self) -> np.ndarray:
        """行 × 型 の通し番号（欠けは -1）。"""
        c = self.codes.astype(np.int64) + self.offs[:-1][None, :]
        return np.where(self.codes >= 0, c, -1)


def encode(L: pd.DataFrame) -> tuple[np.ndarray, list[list[str]]]:
    from src.research.graded import model as M
    no = set(M.NO_INFO)
    codes = np.full(L.shape, -1, np.int32)
    names = []
    for j, t in enumerate(L.columns):
        s = L[t].astype("string")
        u = sorted(v for v in s.dropna().unique() if v not in no)
        mp = {v: i for i, v in enumerate(u)}
        codes[:, j] = s.map(mp).fillna(-1).astype(int).to_numpy()
        names.append([str(v) for v in u])
    return codes, names


OUTCOME_COLS = ("fin", "win", "top3", "rt", "rf", "ret_tan", "ret_fuku", "pay_ok")


def load_pop(pre_lo: float, pre_hi: float, cls_min: int = 0, cls_max: int = 8, pre_seed: int = 0,
             hi_closed: bool = False, outcome_until: int | None = None, exclude_types: tuple[str, ...] = ()) -> Pop:
    """再現オッズ（種 1000+pre_seed）で [pre_lo, pre_hi) の走（hi_closed なら両端を含む）、クラス cls_min〜cls_max。
    outcome_until を与えると、その年より後の行の着順・払戻の列を読み込んだ直後に消す（NaN。前向きの期間を開かない）。
    exclude_types の型は水準から外す（発走前に分からない型など）。"""
    d, L = F.load()
    if outcome_until is not None:
        late = d["year"].astype(int).to_numpy() > outcome_until
        for c in OUTCOME_COLS:
            if c in d:
                d[c] = d[c].astype(float)
                d.loc[late, c] = np.nan
    if exclude_types:
        L = L.drop(columns=[t for t in exclude_types if t in L.columns])
    pre = np.load(F.OUTW / "full_pre.npy")[pre_seed]
    hi_ok = pre <= pre_hi if hi_closed else pre < pre_hi
    cls = d["cls"].to_numpy()
    m = (pre >= pre_lo) & hi_ok & (cls >= cls_min) & (cls <= cls_max)
    rows = np.flatnonzero(m)
    d = d.iloc[rows].reset_index(drop=True)
    d["pre"] = pre[rows]
    L = L.iloc[rows].reset_index(drop=True)
    codes, names = encode(L)
    yr = d["year"].astype(int).to_numpy()
    d["era"] = np.select([yr <= e[1] for e in ERAS], range(len(ERAS)), len(ERAS) - 1)
    d["cgrp"] = np.select([cls[rows] <= 1, cls[rows] == 2, cls[rows] == 3, cls[rows] == 4], [0, 1, 2, 3], 4)  # 新馬未勝利・1勝・2勝・3勝・OP以上
    d["e3g"] = d["obin"].to_numpy() * NFG + d["fgrp"].to_numpy()
    d["qo"] = d["q"] * d["odds_final"]
    d["ridx"] = pd.factorize(d["race_id"])[0]
    d["hidx"] = pd.factorize(d["horse_id"].astype(str))[0]
    d["kidx"] = pd.factorize(d["kaisai"])[0]
    return Pop(d, codes, list(L.columns), names, rows)


# ---------------------------------------------------------------- 開催の 70/30 の分割（統括役の指示 2026-11-24: 年で分けない）
SPLIT70_SEED = 20261124
SPLIT70_SHARE = 0.70


def search70(kaisai: np.ndarray) -> np.ndarray:
    """2016〜24 の全開催（full の 324）を文字列の順に並べ、default_rng(20261124).permutation の先頭 70% を探す側にする。行ごとの真偽。"""
    keys = np.array(sorted(pd.read_parquet(F.OUTW / "full.parquet", columns=["kaisai"])["kaisai"].unique()))
    perm = np.random.default_rng(SPLIT70_SEED).permutation(len(keys))
    s = set(keys[perm[: int(round(SPLIT70_SHARE * len(keys)))]])
    return np.array([k in s for k in kaisai], bool)


# ---------------------------------------------------------------- 物差し
def band_ratio(d: pd.DataFrame, s: np.ndarray, win: np.ndarray, by_cls: bool = True) -> np.ndarray:
    """探す側 s の 帯（× クラス群）の 勝ち ÷ q。空・小さい層は帯だけの比、さらに 1 に落とす。返り値は行ごとの ρ。"""
    q = d["q"].to_numpy()
    ob = d["obin"].to_numpy()
    rb = np.ones(NB)
    wb = np.bincount(ob[s], weights=win[s], minlength=NB)
    qb = np.bincount(ob[s], weights=q[s], minlength=NB)
    ok = qb >= 5
    rb[ok] = wb[ok] / qb[ok]
    rho = rb[ob]
    if by_cls:
        key = ob * 5 + d["cgrp"].to_numpy()
        w2 = np.bincount(key[s], weights=win[s], minlength=NB * 5)
        q2 = np.bincount(key[s], weights=q[s], minlength=NB * 5)
        ok2 = q2 >= 20
        r2 = np.where(ok2, w2 / np.maximum(q2, 1e-9), rb[np.arange(NB * 5) // 5])
        rho = r2[key]
    return rho


def place_rate(d: pd.DataFrame, s: np.ndarray, top: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """探す側 s の 帯 × 頭数の群 の 3 着内の率と、当たりのときの複勝の払戻の平均（行ごと）。"""
    g = d["e3g"].to_numpy()
    rf = d["rf"].to_numpy()
    ng = NB * NFG
    n = np.bincount(g[s], minlength=ng)
    k = np.bincount(g[s], weights=top[s], minlength=ng)
    pay = np.bincount(g[s], weights=rf[s] * (top[s] > 0), minlength=ng)
    ob = np.arange(ng) // NFG
    nb_ = np.bincount(ob, weights=n, minlength=NB)
    kb_ = np.bincount(ob, weights=k, minlength=NB)
    pb_ = np.bincount(ob, weights=pay, minlength=NB)
    rate = np.where(n >= 30, k / np.maximum(n, 1), kb_[ob] / np.maximum(nb_[ob], 1))
    avg = np.where(k >= 10, pay / np.maximum(k, 1), pb_[ob] / np.maximum(kb_[ob], 1))
    return rate[g], avg[g]


# ---------------------------------------------------------------- 入れ替えの帰無
def permute_codes(P: Pop, seed: int, strata: str = "cgrp_era_obin") -> np.ndarray:
    """馬の型は 層（クラス群 × 時代 × 確定オッズの帯）の中で行ごと、レース単位の型は クラス群 × 時代 の中でレースの束ごとに入れ替える。
    行ごとの水準の組（型をまたいだ相関）は保つ。"""
    rng = np.random.default_rng(seed)
    d = P.d
    key = d["cgrp"].astype(str) + "|" + d["era"].astype(str)
    if "obin" in strata:
        key = key + "|" + d["obin"].astype(str)
    if "vs" in strata:
        key = key + "|" + d["vs"].astype(str)
    perm = np.arange(len(d))
    for _, idx in key.groupby(key).indices.items():
        if len(idx) > 1:
            perm[idx] = idx[rng.permutation(len(idx))]
    rt = [j for j, t in enumerate(P.types) if t in RACE_LEVEL_TYPES]
    ht = [j for j in range(len(P.types)) if j not in rt]
    out = P.codes.copy()
    out[:, ht] = P.codes[perm][:, ht]
    if rt:
        ridx = d["ridx"].to_numpy()
        first = np.r_[0, np.flatnonzero(np.diff(ridx)) + 1]       # 行はレースの順に並んでいる
        assert np.array_equal(ridx[first], np.arange(len(first))), "行がレースの順に並んでいない"
        rk = (d["cgrp"].astype(str) + "|" + d["era"].astype(str)).to_numpy()[first]
        src = first.copy()
        for _, idx in pd.Series(rk).groupby(rk).indices.items():
            if len(idx) > 1:
                src[idx] = first[idx[rng.permutation(len(idx))]]
        out[:, rt] = P.codes[src[ridx]][:, rt]
    return out


# ---------------------------------------------------------------- 合成（着順を引き直す）
class Sim:
    """レースの全頭（full の行）の上で、母集団の行の勝ち・3 着内を引き直す。重みは 確定の q × 帯の比（母集団の行だけ）× 植えた倍率。"""

    def __init__(self, P: Pop, rho: np.ndarray):
        full = pd.read_parquet(F.OUTW / "full.parquet", columns=["race_id", "q", "field"])
        rc = pd.factorize(full["race_id"].to_numpy())[0]
        has = np.zeros(int(rc.max()) + 1, bool)
        has[rc[P.full_rows]] = True
        keep = np.flatnonzero(has[rc])
        self.rid = pd.factorize(full["race_id"].to_numpy()[keep])[0]
        pr = np.full(len(full), -1)
        pr[P.full_rows] = np.arange(len(P.full_rows))
        self.pr = pr[keep]
        self.is_p = self.pr >= 0
        self.base = full["q"].to_numpy(float)[keep].copy()
        self.base[self.is_p] *= rho[self.pr[self.is_p]]
        N = full["field"].to_numpy(float)[keep]
        self.T = np.where(N <= 7, 2, 3)
        self.n = len(P.d)

    def win_prob(self, mult: np.ndarray | None) -> np.ndarray:
        """母集団の行の勝つ確率（レースの中で重みを和 1 に直した値）。"""
        w = self.base.copy()
        if mult is not None:
            w[self.is_p] *= mult[self.pr[self.is_p]]
        tot = np.bincount(self.rid, weights=w)
        out = np.zeros(self.n)
        out[self.pr[self.is_p]] = (w / tot[self.rid])[self.is_p]
        return out

    def calibrate(self, rows: np.ndarray, odds: np.ndarray, target_roi: float, lo: float = 0.3, hi: float = 8.0) -> tuple[float, float]:
        """rows の行に掛ける倍率を、レースの中で直した後の期待の回収率 Σp·オッズ ÷ 走 が target_roi になるように二分法で決める。
        (倍率, 届いた期待の回収率) を返す。端で届かないときは届いた値が目標から外れる（呼ぶ側で記録する）。"""
        mult = np.ones(self.n)
        for _ in range(40):
            m = (lo + hi) / 2
            mult[rows] = m
            roi = float((self.win_prob(mult)[rows] * odds[rows]).mean())
            lo, hi = (m, hi) if roi < target_roi else (lo, m)
        m = (lo + hi) / 2
        mult[rows] = m
        return m, float((self.win_prob(mult)[rows] * odds[rows]).mean())

    def draw(self, mult: np.ndarray | None, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        w = self.base.copy()
        if mult is not None:
            w[self.is_p] *= mult[self.pr[self.is_p]]
        key = np.log(w) + rng.gumbel(size=len(w))
        order = np.lexsort((-key, self.rid))
        start = np.r_[0, np.flatnonzero(np.diff(self.rid[order])) + 1]
        grp = np.repeat(np.arange(len(start)), np.diff(np.r_[start, len(w)]))
        rank = np.empty(len(w), int)
        rank[order] = np.arange(len(w)) - start[grp]
        win, top = np.zeros(self.n), np.zeros(self.n)
        p = self.is_p
        win[self.pr[p]] = rank[p] == 0
        top[self.pr[p]] = rank[p] < self.T[p]
        return win, top


# ---------------------------------------------------------------- 開催で束ねたブートストラップ
def boot_ci(kidx: np.ndarray, num: np.ndarray, den: np.ndarray, n_boot: int = 2000, seed: int = 20261124,
            q: tuple[float, float] = (0.025, 0.975)) -> tuple[float, float, float]:
    """比 Σnum/Σden の点推定と、開催で束ねたブートストラップの区間。"""
    k = pd.factorize(kidx)[0]
    nk = int(k.max()) + 1 if len(k) else 0
    if nk == 0 or den.sum() <= 0:
        return float("nan"), float("nan"), float("nan")
    a = np.bincount(k, weights=num, minlength=nk)
    b = np.bincount(k, weights=den, minlength=nk)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nk, size=(n_boot, nk))
    r = a[idx].sum(1) / np.maximum(b[idx].sum(1), 1e-12)
    return float(a.sum() / b.sum()), float(np.quantile(r, q[0])), float(np.quantile(r, q[1]))


# ---------------------------------------------------------------- 深さ 1 の数え上げ（全体の単位）
def level_sums(flat: np.ndarray, rows: np.ndarray, vals: list[np.ndarray], nl: int) -> list[np.ndarray]:
    """行 rows の、水準ごとの vals の和（flat は 行 × 型 の通し番号、欠けは -1）。"""
    f = flat[rows]
    ok = f >= 0
    idx = f[ok]
    rr = np.broadcast_to(np.arange(len(rows))[:, None], f.shape)[ok]
    return [np.bincount(idx, weights=v[rows][rr], minlength=nl) for v in vals]


# ---------------------------------------------------------------- 深さ 1 の検定（全体の単位・単勝）
@dataclass
class LevelStats:
    W: np.ndarray       # 勝ち
    Eb: np.ndarray      # 帯の比の見込み Σρq
    V: np.ndarray       # Σρq(1−ρq)（排他を入れない近似。保守側）
    N: np.ndarray       # 走
    RT: np.ndarray      # 単勝の払戻の和（円/100）
    BE: np.ndarray      # Σ ρq × 確定オッズ（帯の比のときの払戻の見込み）


def level_stats(P: Pop, flat: np.ndarray, rows: np.ndarray, win: np.ndarray, rho: np.ndarray, rt: np.ndarray) -> LevelStats:
    d = P.d
    e = rho * d["q"].to_numpy()
    o = d["odds_final"].to_numpy(float)
    W, Eb, V, N, RT, BE = level_sums(flat, rows, [win.astype(float), e, e * (1 - e), np.ones(len(d)), rt, e * o], P.n_levels)
    return LevelStats(W, Eb, V, N, RT, BE)


def zscores(st: LevelStats, mult: np.ndarray | float = 1.0) -> np.ndarray:
    """帰無の倍率 mult（帯の比に対して）での z。mult = 損益分岐の倍率なら『100% を超えるか』の検定。"""
    m = np.asarray(mult, float)
    return (st.W - m * st.Eb) / np.sqrt(np.maximum(m * st.V, 1e-12))


def breakeven_mult(st: LevelStats) -> np.ndarray:
    """回収率 100% になる、帯の比に対する倍率（水準ごと）: Σρq ÷ Σρq·オッズ の逆…＝ N ÷ BE。"""
    return st.N / np.maximum(st.BE, 1e-12)


def level_cluster_ss(flat: np.ndarray, rows: np.ndarray, xs: list[np.ndarray], cluster: np.ndarray, nl: int) -> list[np.ndarray]:
    """水準ごとの Σ_c (Σ_{i∈c, 水準} x_i)²（c は馬などの束）。同じ馬の繰り返しを入れた分散（束ねた分散）に使う。"""
    f = flat[rows]
    ok = f >= 0
    lv = f[ok].astype(np.int64)
    rr = np.broadcast_to(np.arange(len(rows))[:, None], f.shape)[ok]
    cl = cluster[rows][rr].astype(np.int64)
    key = lv * (int(cluster.max()) + 1) + cl
    uk, inv = np.unique(key, return_inverse=True)
    lv_of = uk // (int(cluster.max()) + 1)
    out = []
    for x in xs:
        s = np.bincount(inv, weights=x[rows][rr], minlength=len(uk))
        out.append(np.bincount(lv_of, weights=s * s, minlength=nl))
    return out
