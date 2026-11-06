"""走法マップの Best5: 地図の位置（コース特注ゾーン・馬場特注ゾーンの色）から、適性が合う順に 5 頭を選ぶ。

方式は 2025・2026 年の外挿（地図は過去のレースだけで作り、後のレースで評価。計約 1,500 レース）で比べて決めた。
- コース / 馬場 / 両方は、人気を超えた好走でほぼ同じ（誤差の範囲）。3 着内の的中は「コース（平滑）」「両方（平滑）」が最良。
- 格子は隣り合う格子と混ぜて（平滑）から読む。1 位が当たると 2〜3 位も当たりやすい、という偏りは見つからなかったので近さの加点はしない。
- 馬場特注ゾーンが無いレースは、同点のまま並びの順で選ぶと人気馬に偏るので、コース→中立の順に決まった規則で選ぶ。
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

MODES = ("both", "course", "day")
DEFAULT_MODE = "both"
K = 5
SIGMA = 1.5          # 格子の平滑（格子幅の何個分か）
LIFT_CLIP = (0.2, 5.0)


def _smooth_lookup(lift: Optional[list], extent: dict, xy: np.ndarray, sigma: float = SIGMA) -> np.ndarray:
    """格子（extent と同じ並び i*n+j）を平滑してから、各馬の位置の値を読む。格子が無ければ全部 NaN。"""
    n_h = len(xy)
    if not lift or not extent:
        return np.full(n_h, np.nan)
    from scipy.ndimage import gaussian_filter

    n = int(extent["n"])
    g = np.asarray(lift, float)
    if g.size != n * n:
        return np.full(n_h, np.nan)
    g = gaussian_filter(g.reshape(n, n), sigma, mode="nearest") if sigma else g.reshape(n, n)
    (x0, x1), (y0, y1) = extent["x"], extent["y"]
    ix = np.clip(np.rint((xy[:, 0] - x0) / (x1 - x0) * (n - 1)), 0, n - 1).astype(int)
    iy = np.clip(np.rint((xy[:, 1] - y0) / (y1 - y0) * (n - 1)), 0, n - 1).astype(int)
    return g[ix, iy]


def _log(v: np.ndarray) -> np.ndarray:
    return np.log(np.clip(v, *LIFT_CLIP))


def best5(horses: list[dict], extent: Optional[dict], course_lift: Optional[list], day_lift: Optional[list]) -> Optional[dict[str, Any]]:
    """モードごとの Best5。作れないモードは入れない。どちらの格子も無ければ None。

    返り値: {"default": "both", "modes": {mode: {"fallback": None|"course"|"neutral", "picks": [{"rank","horse_number","horse_id","score"}]}}}
    同点は「コースの倍率（平滑なし）→ 馬番」の順で決める（並びの順や人気には頼らない）。"""
    if not horses or not extent or (not course_lift and not day_lift):
        return None
    xy = np.array([[h["x"], h["y"]] for h in horses], float)
    nums = np.array([h.get("horse_number") if h.get("horse_number") is not None else 10**6 for h in horses], float)
    c_s = _smooth_lookup(course_lift, extent, xy)
    c_raw = _smooth_lookup(course_lift, extent, xy, 0)
    d_s = _smooth_lookup(day_lift, extent, xy)
    have_c, have_d = bool(np.isfinite(c_s).any()), bool(np.isfinite(d_s).any())
    tie = np.nan_to_num(c_raw, nan=1.0)

    def one(score: np.ndarray) -> list[dict[str, Any]]:
        order = np.lexsort((nums, -tie, -np.round(score, 9)))[:K]
        return [{"rank": r + 1, "horse_number": horses[i].get("horse_number"), "horse_id": horses[i].get("horse_id"), "score": round(float(score[i]), 4)}
                for r, i in enumerate(order)]

    zero = np.zeros(len(horses))
    log_c = _log(np.where(np.isfinite(c_s), c_s, 1.0)) if have_c else zero
    log_d = _log(np.where(np.isfinite(d_s), d_s, 1.0)) if have_d else zero
    modes: dict[str, Any] = {}
    if have_c or have_d:
        modes["both"] = {"fallback": None if (have_c and have_d) else ("course" if have_c else "day"), "picks": one(log_c + log_d)}
    if have_c:
        modes["course"] = {"fallback": None, "picks": one(log_c)}
    if have_d:
        modes["day"] = {"fallback": None, "picks": one(log_d)}
    elif have_c:
        modes["day"] = {"fallback": "course", "picks": one(log_c)}
    return {"default": DEFAULT_MODE, "modes": modes} if modes else None
