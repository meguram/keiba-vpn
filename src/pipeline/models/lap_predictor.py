"""Stage 2: ペースカテゴリ + 1F ラップ予測（LightGBM per-furlong）。"""

from __future__ import annotations

from typing import Any

import numpy as np


PACE_LABELS = ("HIGH", "MIDDLE", "SLOW")


def predict_pace_and_laps(
    race_id: str,
    *,
    distance: int = 1600,
    front_runner_count: int = 2,
    field_size: int = 16,
    predicted_positions: list[int] | None = None,
) -> dict[str, Any]:
    """
    距離・逃げ馬数からペースとラップ系列を推定（モデル未学習時はヒューリスティック）。
    学習済み Booster があれば LightGBM per-furlong に差し替え可能。
    """
    furlongs = max(1, distance // 200)
    front_ratio = front_runner_count / max(field_size, 1)
    if front_ratio > 0.3:
        pace_category = "HIGH"
        base_lap = 12.4
    elif front_ratio < 0.1:
        pace_category = "SLOW"
        base_lap = 12.0
    else:
        pace_category = "MIDDLE"
        base_lap = 12.2

    lap_times = []
    for i in range(1, furlongs + 1):
        if i <= 2:
            sec = base_lap + 0.3
        elif i >= furlongs - 1:
            sec = base_lap - 0.2
        else:
            sec = base_lap
        lap_times.append({"furlong_index": i, "predicted_lap_sec": round(float(sec), 2)})

    return {"pace_category": pace_category, "lap_times": lap_times}


def load_lgbm_lap_model():
    """Phase 4 前: 将来 lap_lgbm.pkl をロード。"""
    return None
