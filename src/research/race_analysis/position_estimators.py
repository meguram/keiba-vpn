"""過去の位置取り（初角の通過順位÷頭数。0 に近いほど前）から、その馬の「ふだんの位置」を出す集計の候補と、出遅れ疑いの判定。

候補（評価: src/scripts/data/pos_late_start_eval.py。数を見る前に固定）:
  M1 中央値
  M2 KDE（ガウス核・帯域はシルバーマンの規則・下限 BW_MIN）の頂上
  M3 M2 に新しい走ほど重い重み（DECAY^何走前）
  M4 出遅れ疑いの走を除いた加重平均（DECAY^何走前）
  M5 M4 の挽回の条件を外した版（ふだんより LATE_GAP 以上後ろの走を除く加重平均）
出遅れ疑い: 初角の位置が、その馬の他の走の中央値より LATE_GAP 以上後ろで、かつ挽回した走
（最後の角までに RECOVER 以上前へ上がった、または上がり 3F がメンバーの上位 1/3）。
走数が MIN_RUNS 未満の馬は、どの候補も中央値を使う。
"""

from __future__ import annotations

import math
import statistics

import numpy as np

DECAY = 0.85
BW_MIN = 0.05
LATE_GAP = 0.3
RECOVER = 0.15
LAST3F_TOP = 1 / 3
MIN_RUNS = 3
MIN_LATE_RUNS = 3            # 出遅れ疑いを判定するのに要る走数（自分以外の走が 2 走以上）
PEAK2_RATIO = 0.5            # 2 つ目の山: 主の山の密度のこの倍以上
PEAK2_SEP = 0.25             # 2 つ目の山: 主の山からこの距離以上
GRID = np.linspace(0.0, 1.0, 101)


def silverman_bw(xs: list[float], ws: list[float] | None = None) -> float:
    """シルバーマンの規則の帯域（0.9·min(sd, IQR/1.34)·n^-1/5。下限 BW_MIN）。
    重みの有無で基準をそろえる: sd は（重み付きの）母標準偏差、n は有効な走数 n_eff（重みなしなら走数）、IQR は重みを付けない。"""
    a = np.asarray(xs, dtype=float)
    w = np.ones(len(a)) if ws is None else np.asarray(ws, dtype=float)
    n = float(w.sum() ** 2 / (w * w).sum())
    m = float((w * a).sum() / w.sum())
    sd = math.sqrt(float((w * (a - m) ** 2).sum() / w.sum()))
    iqr = float(np.subtract(*np.percentile(a, [75, 25]))) if len(a) > 1 else 0.0
    spread = min(sd, iqr / 1.34) if iqr > 0 else sd
    return max(0.9 * spread * n ** -0.2, BW_MIN)


def kde_curve(xs: list[float], ws: list[float] | None = None) -> np.ndarray:
    """GRID 上の密度（重みは和 1 に直す）。"""
    a = np.asarray(xs, dtype=float)
    w = np.ones(len(a)) if ws is None else np.asarray(ws, dtype=float)
    w = w / w.sum()
    h = silverman_bw(xs, ws)
    z = (GRID[:, None] - a[None, :]) / h
    return (np.exp(-0.5 * z * z) * w[None, :]).sum(axis=1) / (h * math.sqrt(2 * math.pi))


def kde_peaks(xs: list[float], ws: list[float] | None = None) -> list[tuple[float, float]]:
    """山の頂上（位置, 密度）を密度の高い順に。主の山と、条件（PEAK2_RATIO・PEAK2_SEP）を満たす 2 つ目の山まで。"""
    d = kde_curve(xs, ws)
    idx = [i for i in range(len(d)) if (i == 0 or d[i] > d[i - 1]) and (i == len(d) - 1 or d[i] >= d[i + 1])]   # 平らな頂上は前の端
    idx.sort(key=lambda i: (-d[i], i))                                                                              # 同じ密度は前の位置
    main = idx[0]
    out = [(float(GRID[main]), float(d[main]))]
    for i in idx[1:]:
        if d[i] >= PEAK2_RATIO * d[main] and abs(GRID[i] - GRID[main]) >= PEAK2_SEP:
            out.append((float(GRID[i]), float(d[i])))
            break
    return out


def late_flags(first: list[float], last: list[float | None], l3f_pct: list[float | None], need_recover: bool = True) -> list[bool]:
    """出遅れ疑いの印（新しい順の走ごと）。last は最後の角の位置（角が 1 つだけなら None）、l3f_pct は上がり 3F のメンバー内の順位÷頭数。"""
    n = len(first)
    if n < MIN_LATE_RUNS:
        return [False] * n
    out = []
    for i in range(n):
        others = first[:i] + first[i + 1:]
        gap = first[i] - statistics.median(others)
        rec = (last[i] is not None and first[i] - last[i] >= RECOVER) or (l3f_pct[i] is not None and l3f_pct[i] <= LAST3F_TOP)
        out.append(gap >= LATE_GAP and (bool(rec) or not need_recover))
    return out


def estimate(first: list[float], how: str, late: list[bool] | None = None) -> float | None:
    """first は新しい順の位置取り。how は median / kde / kde_w / mean_nolate / mean_nofar（late に挽回の条件あり・なしの印を渡す）。走数が MIN_RUNS 未満なら中央値。"""
    if not first:
        return None
    if len(first) < MIN_RUNS or how == "median":
        return float(statistics.median(first))
    if how == "kde":
        return kde_peaks(first)[0][0]
    if how == "kde_w":
        return kde_peaks(first, [DECAY ** i for i in range(len(first))])[0][0]
    if how in ("mean_nolate", "mean_nofar"):
        late = late or [False] * len(first)
        used = [(i, r) for i, r in enumerate(first) if not late[i]] or list(enumerate(first))
        ws = [DECAY ** i for i, _ in used]
        return sum(w * r for w, (_, r) in zip(ws, used)) / sum(ws)
    raise ValueError(how)
