"""全レース版の条件の探索と、◎・消しの当てはめ（2016〜2024 年の中だけ）。

- 単位 = コース × クラスの群（良馬場）。母集団（主 = 単位、副 = 同じコースの全群）の走で、切り口 × 水準の AND を探す（graded.buy.search）。
- 買い: 複勝率 70% 以上・単勝か複勝の回収率 100% 超（buy._qual）・走数の下限（版で変える）。1〜2 走は「極小」の印。
- 消し: 走数の下限以上で 3 着以内が 0（[0-0-0-n]）。
- 同じ走の集まりになる条件は 1 つにまとめる（AND の書き方の違いで ◎ の点が増えないように。型の数が少ない → 型 → 水準の順で残す）。
- ◎ = 当てはまった買いの条件の数（または複勝率の和）が最大の馬（同点は馬番の小さい方）。当てはまる馬がいないレースは買わない。
- 消し = 消しの条件に当たった馬（◎ は除く）。レースで 3 頭以上のときだけ消す。
- 行 × 条件の密な行列は作らない（条件ごとに当たりを足していく）。条件が多すぎるときは、型の数の少ない順に上限で打ち切って記録する。
- 値は探した同じ走で数える当てはめの値（過学習してよい。ユーザーの指示）。消しの的中は当てはめでは定義上 100% になるので、
  採否には使わず、消しを付けたレースの割合だけを見る。
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.research.graded import buy as BY

CONF = (2022, 2024)
MAX_CELLS = 200_000      # 1 母集団で扱う条件の上限（型の数の少ない順に残す）


@dataclass(frozen=True)
class Params:
    depth: int = 1            # 組み合わせの深さ（1〜3。副の層は 2 まで）
    n_min: int = 2            # 買いの条件の走数の下限
    kill_n: int = 10          # 消しの条件の走数の下限
    kill_depth: int = 1       # 消しの組み合わせの深さ（1〜2）
    min_races: int = 20       # 主の層で探す単位の回数の下限。足りない単位は、同じコースの全群（副）で探す
    weight: str = "count"     # ◎ の点: count = 当てはまった条件の数、rate = 条件の複勝率の和
    dayof: bool = False       # True = 当日版（当日の値の型も使う）、False = 前日版
    groups: int = 3           # クラスの群の分け方: 3 = 新馬・未勝利／1勝・2勝／3勝・OP、4 = 新馬を分ける

    def name(self) -> str:
        return (f"{'当日版' if self.dayof else '前日版'}・群{self.groups}・深さ{self.depth}・走数{self.n_min}以上・消し{self.kill_n}走{self.kill_depth}段"
                f"・主{self.min_races}回・{self.weight}")


def y_matrix(d: pd.DataFrame) -> np.ndarray:
    """buy.OUT の列: n・k（3 着内）・w1・w2・w3・np（払戻の解けた走）・rt・rf・e3。回収率は上限なしの払戻（上限付きは metrics で並べる）。"""
    fin = pd.to_numeric(d["fin"], errors="coerce").to_numpy()
    pay = d["pay_ok"].fillna(False).astype(bool).to_numpy() if "pay_ok" in d.columns else np.zeros(len(d), bool)
    col = lambda c: pd.to_numeric(d[c], errors="coerce").fillna(0).to_numpy() if c in d.columns else np.zeros(len(d))   # noqa: E731
    field = pd.to_numeric(d["field"], errors="coerce").to_numpy()
    return np.column_stack([np.ones(len(d)), fin <= 3, fin == 1, fin == 2, fin == 3, pay, np.where(pay, col("ret_tan"), 0.0),
                            np.where(pay, col("ret_fuku"), 0.0), np.minimum(3.0 / field, 1), np.where(pay, col("ret_tan_c"), 0.0)]).astype(float)


def _mask(X: np.ndarray, ts, cs) -> np.ndarray:
    m = np.ones(len(X), bool)
    for t, c in zip(ts, cs):
        m &= X[:, t] == c
    return m


def search_kill(X: np.ndarray, Y: np.ndarray, n_min: int, depth: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """3 着以内が 0 の条件（走数 n_min 以上）。深さ 1〜2。"""
    out = []
    T = X.shape[1]
    m = (1 << BY.CODE_BITS) - 1
    for t in range(T):
        v = X[:, t] >= 0
        if not v.any():
            continue
        u, S, _ = BY._agg(X[v, t].astype(np.int64), Y[v])
        out += [((t,), (int(c),)) for c in u[(S[:, 0] >= n_min) & (S[:, 1] == 0)]]
    if depth < 2:
        return out
    for t1 in range(T - 1):
        v1 = X[:, t1] >= 0
        if not v1.any():
            continue
        t2s = np.arange(t1 + 1, T)
        X1 = X[v1]
        K = (t2s[None, :].astype(np.int64) << (2 * BY.CODE_BITS)) | (X1[:, [t1]].astype(np.int64) << BY.CODE_BITS) | X1[:, t2s].astype(np.int64)
        V = X1[:, t2s] >= 0
        if not V.any():
            continue
        rows = np.broadcast_to(np.arange(len(X1))[:, None], K.shape)[V]
        u, S, _ = BY._agg(K[V], Y[v1][rows])
        for key in u[(S[:, 0] >= n_min) & (S[:, 1] == 0)]:
            out.append(((t1, int(key >> (2 * BY.CODE_BITS))), (int((key >> BY.CODE_BITS) & m), int(key & m))))
    return out


def find_cells(Xp: np.ndarray, Yp: np.ndarray, p: Params, sub: bool = False) -> dict:
    """母集団で、買いと消しの条件を探し、同じ走の集まりになる条件を 1 つにまとめ、走数などを付ける。"""
    depth = min(p.depth, 2) if sub else p.depth
    found = sorted(BY.search(Xp, Yp, depth), key=lambda f: (len(f[0]), f[0], f[1]))
    capped = len(found) > MAX_CELLS
    found = found[:MAX_CELLS]
    seen, buy = set(), []
    for ts, cs in found:
        m = _mask(Xp, ts, cs)
        key = hashlib.sha1(np.flatnonzero(m).tobytes()).digest()
        if key in seen:
            continue
        seen.add(key)
        s = Yp[m].sum(0)
        n, k = int(s[0]), int(s[1])
        if n < p.n_min:
            continue
        buy.append({"ts": ts, "cs": cs, "n": n, "k": k, "rate": k / n, "roi_t": float(s[6] / s[5]) if s[5] else None,
                    "roi_f": float(s[7] / s[5]) if s[5] else None, "tiny": n <= 2})
    seen, kill = set(), []
    for ts, cs in sorted(search_kill(Xp, Yp, p.kill_n, p.kill_depth), key=lambda f: (len(f[0]), f[0], f[1])):
        m = _mask(Xp, ts, cs)
        key = hashlib.sha1(np.flatnonzero(m).tobytes()).digest()
        if key not in seen:
            seen.add(key)
            kill.append({"ts": ts, "cs": cs, "n": int(m.sum())})
    return {"buy": buy, "kill": kill, "capped": capped, "n_found": len(found)}


def apply_cells(X: np.ndarray, race: np.ndarray, hn: np.ndarray, cells: dict, weight: str) -> tuple[dict, np.ndarray]:
    """対象の行に条件を当てはめ、レースごとの ◎ の行（{race: 行}）と、消す行の真偽を返す（行 × 条件の行列は作らない）。"""
    sc = np.zeros(len(X))
    for c in cells["buy"]:
        sc += _mask(X, c["ts"], c["cs"]) * (1.0 if weight == "count" else c["rate"])
    kh = np.zeros(len(X), bool)
    for c in cells["kill"]:
        kh |= _mask(X, c["ts"], c["cs"])
    honmei, killed = {}, np.zeros(len(X), bool)
    for r in np.unique(race):
        idx = np.flatnonzero(race == r)
        s = sc[idx]
        if s.max() > 0:
            honmei[r] = int(idx[np.lexsort((hn[idx], -s))][0])
        kk = idx[kh[idx]]
        kk = kk[kk != honmei.get(r, -1)]
        if len(kk) >= 3:
            killed[kk] = True
    return honmei, killed


def metrics(Y: np.ndarray, race: np.ndarray, honmei_rows: np.ndarray, killed_rows: np.ndarray, races_all: int) -> dict:
    """◎ の勝率・回収率（払戻の解けた回で割る。上限なしと上限付き）・買った回の割合、消しを付けたレースの割合・1 レースの頭数。
    honmei_rows・killed_rows は行の番号、race は全行のレースの番号（Y と同じ長さ）。"""
    h = Y[honmei_rows] if len(honmei_rows) else np.zeros((0, Y.shape[1]))
    npay = h[:, 5].sum()
    kraces = len(set(race[killed_rows])) if len(killed_rows) else 0
    return {"races": int(races_all), "bet_races": int(len(h)), "coverage": float(len(h) / races_all) if races_all else None,
            "win": float(h[:, 2].mean()) if len(h) else None, "place": float(h[:, 1].mean()) if len(h) else None,
            "roi_t": float(h[:, 6].sum() / npay) if npay else None, "roi_f": float(h[:, 7].sum() / npay) if npay else None,
            "roi_t_capped": float(h[:, 9].sum() / npay) if npay else None,
            "killed": int(len(killed_rows)), "kill_races": int(kraces), "kill_coverage": float(kraces / races_all) if races_all else None,
            "kill_per_race": float(len(killed_rows) / kraces) if kraces else None,
            "kill_precision": float(1 - Y[killed_rows][:, 1].mean()) if len(killed_rows) else None}


def params_dict(p: Params) -> dict:
    return asdict(p) | {"name": p.name()}
