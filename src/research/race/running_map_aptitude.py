"""
走法マップのコース適性（コースごとに、走法マップのどのゾーンが好走しやすいか）。

全馬共通の 2 次元座標（running_map の基準）に、2020 年以降の平地の全出走を置き、コース（競馬場・芝ダ・距離）ごとに
「このゾーンにいた馬は、3 着以内に入る期待値（頭数から決まる確率）の何倍入ったか」を地図にする。

  - 24×24 の格子の各点で、ガウスカーネル（幅 0.5）で近くの馬を平滑化して 観測 / 期待 を数える。
  - 倍率 lift = (観測 + M·全コース共通の倍率) / (期待 + M)。サンプルの少ないゾーンは全コース共通の倍率へ寄せる（M=20）。
  - 有意性は「レース単位でまとめた」サンドウィッチ分散の z（同じレースの馬は独立でないため）。
    格子の点どうしは強く相関するので、BH 法（FDR 5%）で多重検定を補正する。期待が小さい点は検定しない。
  - 走法は各馬の「そのレースより前の走」だけから作るので、座標自体に結果の漏れは無いが、
    倍率と有意性は 2020 年以降の全レースで測った値（標本内）。ホールドアウト（馬単位 5 分割）では、
    倍率で補正した予測の対数損失が 1 走あたり 0.012 前後改善する（約 3 分の 2 は全コース共通の脚質の効果、約 3 分の 1 がコース固有）。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

GRID = 24
BANDWIDTH = 0.5
SHRINK = 20.0
MIN_RUNS = 300
MIN_EXPECTED = 5.0
FDR = 0.05
MIN_FIELD = 5
_CHUNK = 20000


def course_key(venue: Any, surface: Any, distance: Any) -> str:
    return f"{venue}|{surface}|{int(distance)}"


def extent_from(xy: np.ndarray) -> dict[str, Any]:
    """格子の範囲 = 全馬の座標の 1〜99 パーセンタイル。"""
    lo, hi = np.percentile(xy, 1, axis=0), np.percentile(xy, 99, axis=0)
    return {"n": GRID, "x": [round(float(lo[0]), 4), round(float(hi[0]), 4)], "y": [round(float(lo[1]), 4), round(float(hi[1]), 4)]}


def _grid(extent: dict[str, Any]) -> np.ndarray:
    gx, gy = np.meshgrid(np.linspace(*extent["x"], extent["n"]), np.linspace(*extent["y"], extent["n"]), indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], 1)


def _kernel(grid: np.ndarray, P: np.ndarray):
    for i in range(0, len(P), _CHUNK):
        d2 = ((grid[:, None, :] - P[i:i + _CHUNK][None, :, :]) ** 2).sum(2)
        yield i, np.exp(-d2 / (2 * BANDWIDTH ** 2))


def _obs_exp(grid: np.ndarray, P: np.ndarray, y: np.ndarray, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    o, x = np.zeros(len(grid)), np.zeros(len(grid))
    for i, W in _kernel(grid, P):
        o += W @ y[i:i + W.shape[1]]
        x += W @ e[i:i + W.shape[1]]
    return o, x


def _cluster_z(grid: np.ndarray, P: np.ndarray, resid: np.ndarray, race_codes: np.ndarray) -> np.ndarray:
    """各格子点の z（レースごとに残差を合算してから分散をとる）。"""
    nr = int(race_codes.max()) + 1
    tr = np.zeros((len(grid), nr))
    for i, W in _kernel(grid, P):
        c = race_codes[i:i + W.shape[1]]
        T = W * resid[i:i + W.shape[1]]
        for j in range(len(grid)):
            tr[j] += np.bincount(c, weights=T[j], minlength=nr)
    return tr.sum(1) / np.sqrt((tr ** 2).sum(1) + 1e-9)


def _bh(p: np.ndarray, q: float = FDR) -> np.ndarray:
    order = np.argsort(p)
    m = len(p)
    ok = p[order] <= q * (np.arange(1, m + 1) / m)
    k = int(np.max(np.where(ok)[0])) + 1 if ok.any() else 0
    sel = np.zeros(m, bool)
    sel[order[:k]] = True
    return sel


def prepare(tab: pd.DataFrame) -> pd.DataFrame:
    """race_id, venue, surface, distance, finish_position, x, y2 → 期待 e・結果 y を付ける。
    頭数は座標が無い馬も含めて数える（呼び出し側で座標の欠けた行も渡すこと）。"""
    t = tab.copy()
    n = t.groupby("race_id")["race_id"].transform("size")
    t["e"] = np.minimum(3, n) / n
    t["y"] = (t["finish_position"] <= 3).astype(float)
    t = t[(n >= MIN_FIELD) & t[["x", "y2"]].notna().all(axis=1)]
    return t.reset_index(drop=True)


def build(tab: pd.DataFrame, extent: dict[str, Any]) -> dict[str, Any]:
    """tab は prepare() 済み。コースごとの lift と有意性を返す。"""
    from scipy.stats import norm

    grid = _grid(extent)
    P_all = tab[["x", "y2"]].to_numpy()
    o, x = _obs_exp(grid, P_all, tab["y"].to_numpy(), tab["e"].to_numpy())
    glift = o / np.maximum(x, 1e-9)

    courses: dict[str, Any] = {}
    for (venue, surface, distance), g in tab.groupby(["venue", "surface", "distance"]):
        if len(g) < MIN_RUNS:
            continue
        g = g.reset_index(drop=True)
        P = g[["x", "y2"]].to_numpy()
        yv, ev = g["y"].to_numpy(), g["e"].to_numpy()
        co, cx = _obs_exp(grid, P, yv, ev)
        valid = cx > MIN_EXPECTED
        if valid.sum() < 20:
            continue
        z = _cluster_z(grid, P, yv - ev, pd.factorize(g["race_id"])[0])
        sel = np.zeros(len(grid), bool)
        sel[np.where(valid)[0][_bh(2 * norm.sf(np.abs(z[valid])))]] = True
        lift = (co + SHRINK * glift) / (cx + SHRINK)
        sig = np.where(~valid, 0, np.where(sel, 2, 1))
        courses[course_key(venue, surface, distance)] = {
            "n_runs": int(len(g)),
            "n_races": int(g["race_id"].nunique()),
            "lift": [round(float(v), 3) for v in lift],
            "sig": "".join(str(int(v)) for v in sig),
        }
    return {"extent": extent, "courses": courses}
