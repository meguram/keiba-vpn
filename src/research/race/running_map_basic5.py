"""走法マップの「基本 5 軸」: TS・TS持続力・ギアチェンジ・距離適性・芝ダート適性（純関数・ファイルは読まない）。

目的は「画面上の距離が近い = 5 軸が似ている」を見せること。5 軸の z（全馬の分布で正規化した値）から、
  ・馬どうしの距離（2 頭とも値がある軸だけの二乗平均の平方根）と、近い馬 K 頭
  ・レース内の 2 次元座標（主成分 / 多次元尺度法）と、座標が元の距離をどれだけ保つか（距離の順位相関・近い 5 頭の再現率）
  ・選んだ馬を中心にした放射配置（半径 = 5 軸の真の距離、角度 = 2 次元座標の向き）
を作る。距離適性は（適）（最も得点の高い距離）を、芝ダート適性は血統側のダート割合を座標にする。
血統側の距離の得点分布（父・母父の産駒の平均を、産駒の少ない種牡馬ほど全体へ寄せる）もここで作る。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

KEYS = ["ts", "sustain", "gear", "dist", "dirt"]
LABELS = {"ts": "TS", "sustain": "TS持続力", "gear": "ギアチェンジ", "dist": "距離適性", "dirt": "芝ダート適性"}
K = 5                   # 近い馬の数
MIN_COMMON = 3          # 距離を測るのに、2 頭とも値がある軸がこれ以上必要
BINS = np.arange(1000, 3700, 100)      # 距離の得点分布の階級（100m 刻み。範囲外は端に寄せる）
PED_WEIGHTS = (1.0, 0.5)               # 血統側の分布: 父・母父の重み（tp-85 と同じ）
PED_KAPPA = 10.0                       # 産駒がこの頭数いるのと同じ強さで、全体の分布へ寄せる（産駒 1 頭の種牡馬は 9 割が全体）


# ── 距離と近い馬 ───────────────────────────────────────────────────────────

def pair_dist(Z: np.ndarray, min_common: int = MIN_COMMON) -> np.ndarray:
    """馬どうしの距離（n×n）。2 頭とも値がある軸だけの二乗平均の平方根。共通の軸が min_common 未満の組は NaN。対角は 0。"""
    n = len(Z)
    D = np.full((n, n), np.nan)
    ok = ~np.isnan(Z)
    for i in range(n):
        both = ok[i][None, :] & ok
        cnt = both.sum(axis=1)
        diff = np.where(both, np.nan_to_num(Z[i][None, :]) - np.nan_to_num(Z), 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            d = np.sqrt((diff ** 2).sum(axis=1) / cnt)
        D[i] = np.where(cnt >= min_common, d, np.nan)
    np.fill_diagonal(D, 0.0)
    return D


def similar(D: np.ndarray, k: int = K) -> list[list[tuple[int, float]]]:
    """距離行列 D から、馬ごとに近い馬を近い順に k 頭（番号と距離）。測れない組は除く。"""
    out = []
    for i in range(len(D)):
        cand = [(j, float(D[i, j])) for j in range(len(D)) if j != i and not np.isnan(D[i, j])]
        out.append(sorted(cand, key=lambda t: (t[1], t[0]))[:k])
    return out


def axis_diffs(zi: np.ndarray, zj: np.ndarray) -> list[Optional[float]]:
    """軸ごとの差（相手 − 自分、z 単位）。どちらかが欠けた軸は None。"""
    return [None if np.isnan(a) or np.isnan(b) else round(float(b - a), 3) for a, b in zip(zi, zj)]


# ── 2 次元の配置 ───────────────────────────────────────────────────────────

def layout_pca(Z: np.ndarray) -> np.ndarray:
    """5 軸の主成分の上位 2 つ（欠けた軸は平均 0 に置く）。"""
    X = np.nan_to_num(Z - np.nanmean(Z, axis=0, keepdims=True))
    if len(X) < 3:
        return np.zeros((len(X), 2))
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    Y = X @ Vt[:2].T
    return np.pad(Y, ((0, 0), (0, 2 - Y.shape[1])))


def fill_missing(D: np.ndarray) -> np.ndarray:
    """距離が測れない組は、測れた組の平均で埋める（配置の計算が止まらないように）。"""
    D = D.copy()
    bad = np.isnan(D)
    if bad.any():
        off = ~np.eye(len(D), dtype=bool)
        D[bad] = float(np.nanmean(D[off])) if (~bad & off).any() else 1.0
    return D


def layout_mds(Z: np.ndarray, seed: int = 0) -> np.ndarray:
    """距離行列（pair_dist）を保つように 2 次元へ置く（SMACOF。初期値は主成分）。"""
    from sklearn.manifold import MDS

    n = len(Z)
    if n < 3:
        return np.zeros((n, 2))
    D = fill_missing(pair_dist(Z))
    D = (D + D.T) / 2
    m = MDS(n_components=2, dissimilarity="precomputed", n_init=1, random_state=seed, max_iter=500)
    return m.fit_transform(D, init=layout_pca(Z))


def radial(d_row: np.ndarray, Y: np.ndarray, center: int) -> np.ndarray:
    """選んだ馬 center を原点にした放射配置。半径 = 5 軸の真の距離 d_row（測れない相手は Y の距離を尺度合わせして代用）、
    角度 = 2 次元座標 Y での center からの向き。center からの距離は厳密になる（ほかの馬どうしの距離は近似）。"""
    d = Y - Y[center]
    ang = np.arctan2(d[:, 1], d[:, 0])
    dy = np.hypot(d[:, 0], d[:, 1])
    ok = ~np.isnan(d_row) & (dy > 1e-9)
    scale = float(np.median(d_row[ok] / dy[ok])) if ok.any() else 1.0      # 2 次元の距離を 5 軸の距離の尺度に合わせる
    r = np.where(np.isnan(d_row), dy * scale, d_row)
    return np.column_stack([r * np.cos(ang), r * np.sin(ang)])


def knn_recall(D: np.ndarray, Y: np.ndarray, k: int = K) -> Optional[float]:
    """5 軸で近い k 頭のうち、2 次元座標でも近い k 頭に入る割合の平均（馬ごと）。比べられる馬が足りなければ None。"""
    from scipy.spatial.distance import cdist

    D2 = cdist(Y, Y)
    hits = []
    for i in range(len(D)):
        idx = np.array([j for j in range(len(D)) if j != i and not np.isnan(D[i, j])])
        if len(idx) < k:
            continue
        a = set(idx[np.argsort(D[i, idx], kind="stable")[:k]])
        b = set(idx[np.argsort(D2[i, idx], kind="stable")[:k]])
        hits.append(len(a & b) / k)
    return float(np.mean(hits)) if hits else None


def dist_rho(D: np.ndarray, Y: np.ndarray) -> Optional[float]:
    """5 軸の距離と、2 次元座標の距離の順位相関（馬の組ごと）。"""
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr

    iu = np.triu_indices(len(D), 1)
    d5, d2 = D[iu], pdist(Y)
    ok = ~np.isnan(d5)
    if ok.sum() < 3 or np.ptp(d5[ok]) == 0 or np.ptp(d2[ok]) == 0:
        return None
    r = spearmanr(d5[ok], d2[ok])[0]
    return None if np.isnan(r) else float(r)


# ── 血統側の距離の得点分布 ─────────────────────────────────────────────────

def bin_of(dist100) -> np.ndarray:
    """距離（100m 刻みに丸めた m）の階級の番号。範囲外は両端に寄せる。"""
    return np.clip(np.round((np.asarray(dist100, float) - BINS[0]) / 100.0), 0, len(BINS) - 1).astype(int)


def horse_points(p: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """good_distance.profile_table（by_surface=False）の結果から、馬ごとの階級別の点数（馬数×階級数）と馬 ID の並び。"""
    hid = p["horse_id"].astype(str)
    ids = list(dict.fromkeys(hid))
    row = {h: i for i, h in enumerate(ids)}
    P = np.zeros((len(ids), len(BINS)))
    np.add.at(P, (hid.map(row).to_numpy(), bin_of(p["dist100"].to_numpy())), p["points"].to_numpy(float))
    return ids, P


def parent_tables(P: np.ndarray, parent: np.ndarray, kappa: float = PED_KAPPA) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """種牡馬（母父）ごとの産駒の分布。P = 馬ごとの階級別の点数、parent = 馬ごとの種牡馬の ID（不明は ""）。
    馬ごとに割合へ直し（走数の多い馬・勝った馬が支配しないように）、産駒の割合の合計 + kappa 頭分の全体の分布を、頭数 + kappa で割る
    （産駒が少ないほど全体へ寄る）。戻り値は ({種牡馬 ID: 分布}, 全体の分布)。"""
    tot = P.sum(axis=1, keepdims=True)
    R = np.divide(P, tot, out=np.zeros_like(P), where=tot > 0)
    has = tot[:, 0] > 0
    g = R[has].mean(axis=0)
    out: dict[str, np.ndarray] = {}
    keep = (parent != "") & has
    if keep.any():
        ids, inv = np.unique(parent[keep], return_inverse=True)
        S = np.zeros((len(ids), P.shape[1]))
        np.add.at(S, inv, R[keep])
        n = np.bincount(inv, minlength=len(ids))
        for pid, s_, c in zip(ids, S, n):
            out[str(pid)] = (s_ + kappa * g) / (c + kappa)
    return out, g


def ped_hist(sire: Optional[np.ndarray], dam_sire: Optional[np.ndarray], glob: np.ndarray,
             weights: tuple[float, float] = PED_WEIGHTS) -> np.ndarray:
    """父・母父の分布を重み 1・0.5 で混ぜた、血統から見た距離の得点分布。どちらも不明なら全体の分布。"""
    parts = [(w, h) for w, h in zip(weights, (sire, dam_sire)) if h is not None]
    if not parts:
        return glob.copy()
    return sum(w * h for w, h in parts) / sum(w for w, _ in parts)


def hist_summary(h: np.ndarray, tau: float = 0.10) -> dict[str, int]:
    """分布 h（階級別の割合）の（短）（長）（適）。レンジは割合 >= tau（無ければ最大の距離）、適は最大の距離。"""
    from src.research.race import good_distance as gd

    lo, hi = gd._range_from_rel(h, BINS, tau)
    return {"short_m": int(lo[0]), "long_m": int(hi[0]), "best_m": gd.best_distance(h, BINS)}


def hist_dict(h: np.ndarray, digits: int = 4) -> dict[int, float]:
    """画面用: {距離 m: 割合}（割合が 0 の階級は省く）。"""
    return {int(b): round(float(v), digits) for b, v in zip(BINS, h) if v >= 10 ** -digits}
