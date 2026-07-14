"""JRA ラップタイムから前半スプリット計測点を求めるユーティリティ。"""

from __future__ import annotations

import json
import math
from typing import Optional


def lap_segment_end_distances(distance: int, n_laps: int) -> list[int]:
    """
    各区間終端の累積距離（m）。

    - 距離が 200m 倍数: 全区間 200m
    - それ以外: 先頭 = distance % 200（100m / 150m 等）、以降 200m
    - 先頭 300m + 200m×k もラップ数が一致するときのみ採用
    - いずれも不一致なら等分割（フォールバック）
    """
    d = int(distance)
    n = int(n_laps)
    if d <= 0 or n <= 0:
        return []

    remainder = d % 200
    if remainder == 0:
        segs = [200] * (d // 200)
    else:
        segs = [remainder] + [200] * ((d - remainder) // 200)

    if len(segs) != n and d >= 300 and (d - 300) % 200 == 0:
        segs300 = [300] + [200] * ((d - 300) // 200)
        if len(segs300) == n:
            segs = segs300

    if len(segs) != n:
        seg_len = d / n
        cum = 0.0
        ends: list[int] = []
        for _ in range(n):
            cum += seg_len
            ends.append(int(round(cum)))
        return ends

    cum = 0
    ends: list[int] = []
    for seg in segs:
        cum += seg
        ends.append(cum)
    return ends


def parse_lap_times(lap_json, distance: int) -> dict[int, float]:
    """lap_times JSON → {累積距離m: 累積秒}"""
    if not lap_json or (isinstance(lap_json, float) and math.isnan(lap_json)):
        return {}
    try:
        laps = json.loads(lap_json) if isinstance(lap_json, str) else lap_json
        if not laps:
            return {}
        ends = lap_segment_end_distances(int(distance), len(laps))
        if len(ends) != len(laps):
            return {}
        result: dict[int, float] = {}
        cumtime = 0.0
        for t, end_m in zip(laps, ends):
            cumtime += float(t)
            result[int(end_m)] = round(cumtime, 2)
        return result
    except Exception:
        return {}


def select_split_point(distance: int, available_dists: list[int]) -> Optional[int]:
    """レース距離の中間（50%）に最も近いラップ計測点を選ぶ。同距離差は中間以下を優先。"""
    if not available_dists:
        return None
    target = float(distance) * 0.5
    return min(available_dists, key=lambda d: (abs(float(d) - target), 1 if d > target else 0))
