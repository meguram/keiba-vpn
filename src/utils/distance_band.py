"""日本競馬の距離区分（プロジェクト共通）。

区分（左閉右開）:
  短距離 … 〜1499m
  マイル … 1500m〜1799m
  中距離 … 1800m〜2399m  （ダービー・JC等2400mは中距離）
  長距離 … 2400m〜
"""

from __future__ import annotations

from typing import Any

# pd.cut(..., right=False) と同じ境界
DIST_BINS_M = [0, 1500, 1800, 2400, 99999]
DIST_LABELS_JA = ["短距離", "マイル", "中距離", "長距離"]
DIST_KEYS = ["sprint", "mile", "middle", "long"]


def distance_m(distance: Any, n_furlongs: int | None = None) -> int:
    """メートル距離。未設定時はハロン数×200m。"""
    try:
        d = int(distance)
    except (TypeError, ValueError):
        d = 0
    if d > 0:
        return d
    if n_furlongs and n_furlongs > 0:
        return n_furlongs * 200
    return 0


def distance_group_key(distance: Any, n_furlongs: int | None = None) -> str:
    """内部キー sprint / mile / middle / long を返す。"""
    d = distance_m(distance, n_furlongs)
    if d <= 0:
        return "mile"
    if d < 1500:
        return "sprint"
    if d < 1800:
        return "mile"
    if d < 2400:
        return "middle"
    return "long"


def distance_group_label_ja(key: str) -> str:
    mapping = dict(zip(DIST_KEYS, DIST_LABELS_JA, strict=True))
    return mapping.get(key, key)
