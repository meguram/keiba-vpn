"""
race.netkeiba.com 結果（race_result_on_time）JSON の正規化。

過去アーカイブページを「フルスキーマ」の基準とし、当日速報などで欠けるキーは None で埋める。
"""

from __future__ import annotations

from typing import Any

_RACE_LIVE_TOP_KEYS: tuple[str, ...] = (
    "race_id",
    "race_name",
    "date",
    "venue",
    "round",
    "surface",
    "distance",
    "direction",
    "weather",
    "track_condition",
    "start_time",
    "grade",
    "field_size",
)

# 1 頭分のエントリ（欠けを None）
_RACE_LIVE_ENTRY_KEYS: tuple[str, ...] = (
    "finish_position",
    "bracket_number",
    "horse_number",
    "horse_name",
    "horse_id",
    "sex_age",
    "jockey_weight",
    "jockey_name",
    "jockey_id",
    "finish_time",
    "time_sec",
    "margin",
    "passing_order",
    "last_3f",
    "odds",
    "popularity",
    "weight",
    "weight_change",
    "trainer_name",
    "trainer_id",
)


def normalize_race_live_result(
    data: dict[str, Any],
    *,
    profile: str,
) -> dict[str, Any]:
    """
    同一のキー集合を保証する。既に存在する値は上書きしない（欠キーのみ None）。

    ``profile`` は ``_meta.result_schema_profile`` にも書き込む。
    """
    for k in _RACE_LIVE_TOP_KEYS:
        if k not in data:
            data[k] = None
    data.setdefault("payoff", {})
    data.setdefault("lap_times", [])
    data.setdefault("pace", {})
    data.setdefault("corner_passing", [])
    data.setdefault("entries", [])
    for ent in data["entries"]:
        if not isinstance(ent, dict):
            continue
        for k in _RACE_LIVE_ENTRY_KEYS:
            if k not in ent:
                ent[k] = None
    meta = dict(data.get("_meta") or {})
    meta["result_schema_profile"] = profile
    data["_meta"] = meta
    return data
