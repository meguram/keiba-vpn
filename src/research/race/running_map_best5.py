"""走法マップの Best5: 地図の位置（コース特注ゾーン・馬場特注ゾーンの色）から、適性が合う順に 5 頭を選ぶ（1 パターンだけ）。

得点 = log(コース特注ゾーンの倍率) + W_DAY * log(馬場特注ゾーンの倍率)。倍率は隣り合う格子と混ぜて（平滑）から読む。
2025・2026 年の外挿（地図は過去のレースだけで作り、後のレースで評価。計 1,565 レース）で配分 W_DAY を調整した。
- 定数の配分は W_DAY=0〜4 でほぼ平らで、人気を除いた上乗せは 0.5 が最大、3 着内の的中も 0〜1 が最良（2 以上は下がる）。
- クラス・レース番号・当日のレース数・芝ダ・地図の水準で配分を変えても、年をまたいだ検証では定数より良くならなかった（採用しない）。
  当日のデータ量は、馬場特注ゾーンの倍率が少ないほど 1（中立）に寄る作りで、すでに効いている。
- 近さの加点・人気込みは効果が無かった。馬場特注ゾーンが無いレースは、コースだけで選ぶ。
同点は「コースの倍率（平滑なし）→ 馬番」で決め、並びの順（≒人気順）には頼らない。
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

W_DAY = 0.5          # 馬場特注ゾーンの配分（コースを 1 として）
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
    """{"w_day", "source": "both"|"course"|"day", "picks": [{"rank","horse_number","horse_id","score"}]}。どちらの格子も無ければ None。"""
    if not horses or not extent or (not course_lift and not day_lift):
        return None
    xy = np.array([[h["x"], h["y"]] for h in horses], float)
    nums = np.array([h.get("horse_number") if h.get("horse_number") is not None else 10**6 for h in horses], float)
    c_s = _smooth_lookup(course_lift, extent, xy)
    c_raw = _smooth_lookup(course_lift, extent, xy, 0)
    d_s = _smooth_lookup(day_lift, extent, xy)
    have_c, have_d = bool(np.isfinite(c_s).any()), bool(np.isfinite(d_s).any())
    if not (have_c or have_d):
        return None
    zero = np.zeros(len(horses))
    score = (_log(np.where(np.isfinite(c_s), c_s, 1.0)) if have_c else zero) + (W_DAY * _log(np.where(np.isfinite(d_s), d_s, 1.0)) if have_d else zero)
    order = np.lexsort((nums, -np.nan_to_num(c_raw, nan=1.0), -np.round(score, 9)))[:K]
    picks = [{"rank": r + 1, "horse_number": horses[i].get("horse_number"), "horse_id": horses[i].get("horse_id"), "score": round(float(score[i]), 4)}
             for r, i in enumerate(order)]
    return {"w_day": W_DAY, "source": "both" if (have_c and have_d) else ("course" if have_c else "day"), "picks": picks}
