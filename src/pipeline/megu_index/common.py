"""めぐ指数共通定義（AREA-11 準拠）。"""

from __future__ import annotations

# §6-2 パターンB ウェイト
WEIGHTS_B = [0.35, 0.25, 0.20, 0.12, 0.08]

MEGU_POINTS_PER_SEC = 10.0
MEGU_BASE = 50.0  # AREA-11 v3: 50 = 1勝クラス2着馬相当
MAJOR_DIST_CHANGE_M = 600


def dist_band(distance: int) -> str:
    """距離帯（AREA-11 §6-3）。"""
    d = int(distance)
    if d < 1500:
        return "sprint"
    if d < 1800:
        return "mile"
    if d < 2400:
        return "middle"
    return "long"


def megu_to_adjusted_time(megu_index: float, par_time_sec: float) -> float:
    """指数 → 補正後タイム（秒）。"""
    return float(par_time_sec) - (float(megu_index) - MEGU_BASE) / MEGU_POINTS_PER_SEC


def adjusted_time_to_megu(adjusted_time_sec: float, par_time_sec: float) -> float:
    """補正後タイム → 指数。"""
    return MEGU_BASE + (float(par_time_sec) - float(adjusted_time_sec)) * MEGU_POINTS_PER_SEC


def is_major_condition_change(
    surface_from: str,
    surface_to: str,
    distance_from: int,
    distance_to: int,
) -> bool:
    """芝↔ダートまたは距離差 ±600m 超。"""
    if surface_from != surface_to:
        return True
    return abs(int(distance_to) - int(distance_from)) >= MAJOR_DIST_CHANGE_M
