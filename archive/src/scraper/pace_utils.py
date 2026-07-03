"""ペース・ラップデータの補完ユーティリティ。"""

from __future__ import annotations

from typing import Any


def pace_has_first_half(pace: Any) -> bool:
    if pace is None:
        return False
    if isinstance(pace, dict):
        try:
            v = float(pace.get("first_half_3f"))
            return v > 0
        except (TypeError, ValueError):
            return False
    return False


def merge_race_result_pace(
    result: dict[str, Any] | None,
    lap: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """race_result に pace / lap_times を race_result_lap 由来で補完する。"""
    if result is None:
        return None
    out = dict(result)
    pace = dict(out.get("pace") or {})
    lap_pace = dict((lap or {}).get("pace") or {})
    if not pace_has_first_half(pace) and pace_has_first_half(lap_pace):
        for k, v in lap_pace.items():
            if v is not None:
                pace[k] = v
    out["pace"] = pace
    lap_times = list(out.get("lap_times") or [])
    if not lap_times and lap:
        lap_times = list(lap.get("lap_times") or [])
    if lap_times:
        out["lap_times"] = lap_times
    return out
