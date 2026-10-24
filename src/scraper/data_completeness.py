"""
レース系 JSON に _meta.data_quality を付与する（完全性の目安・再取得判断用）。

モニターや将来の smart_skip が参照できるよう、保存直前に runner から呼ぶ。
"""

from __future__ import annotations

from typing import Any


def _ensure_meta(data: dict[str, Any]) -> dict[str, Any]:
    m = data.setdefault("_meta", {})
    if not isinstance(m, dict):
        data["_meta"] = {}
        m = data["_meta"]
    return m


def enrich_race_shutuba(data: dict[str, Any]) -> None:
    entries = data.get("entries") or []
    n = len(entries)
    with_waku = sum(1 for e in entries if int(e.get("bracket_number") or 0) > 0)
    with_uma = sum(1 for e in entries if int(e.get("horse_number") or 0) > 0)
    with_hid = sum(1 for e in entries if (e.get("horse_id") or "").strip())

    low = 0 < n < 4
    miss_waku = n > 0 and with_waku < max(1, int(n * 0.5))
    miss_uma = n > 0 and with_uma < max(1, int(n * 0.5))

    if n >= 4 and not miss_waku and not miss_uma and with_hid >= n * 0.8:
        level = "ok"
    elif n == 0:
        level = "empty"
    else:
        level = "partial"

    _ensure_meta(data)["data_quality"] = {
        "schema_version": 1,
        "category": "race_shutuba",
        "entry_count": n,
        "with_waku": with_waku,
        "with_umaban": with_uma,
        "with_horse_id": with_hid,
        "flags": {
            "low_entry_count": low,
            "missing_many_waku": miss_waku,
            "missing_many_umaban": miss_uma,
        },
        "level": level,
    }


def enrich_race_result(data: dict[str, Any]) -> None:
    entries = data.get("entries") or []
    n = len(entries)
    with_rank = sum(
        1
        for e in entries
        if (e.get("finish_position") is not None and e.get("finish_position") != "")
        or (e.get("rank") is not None and e.get("rank") != "")
    )
    with_hid = sum(1 for e in entries if (e.get("horse_id") or "").strip())

    low = 0 < n < 4
    miss_rank = n > 0 and with_rank < max(1, int(n * 0.5))

    if n >= 4 and not miss_rank and with_hid >= n * 0.8:
        level = "ok"
    elif n == 0:
        level = "empty"
    else:
        level = "partial"

    _ensure_meta(data)["data_quality"] = {
        "schema_version": 1,
        "category": "race_result",
        "entry_count": n,
        "with_finish_rank": with_rank,
        "flags": {
            "low_entry_count": low,
            "missing_many_finish": miss_rank,
        },
        "level": level,
    }


def enrich_race_shutuba_past(data: dict[str, Any]) -> None:
    entries = data.get("entries") or []
    n = len(entries)
    past_lens = [len(e.get("past_races") or []) for e in entries]
    min_past = min(past_lens) if past_lens else 0
    max_past = max(past_lens) if past_lens else 0
    with_hid = sum(1 for e in entries if (e.get("horse_id") or "").strip())

    sparse = n > 0 and max_past < 2 and min_past < 2
    low = 0 < n < 4

    if n >= 4 and min_past >= 1 and with_hid >= n * 0.8:
        level = "ok"
    elif n == 0:
        level = "empty"
    else:
        level = "partial"

    _ensure_meta(data)["data_quality"] = {
        "schema_version": 1,
        "category": "race_shutuba_past",
        "entry_count": n,
        "min_past_races_per_horse": min_past,
        "max_past_races_per_horse": max_past,
        "flags": {
            "low_entry_count": low,
            "sparse_past_columns": sparse,
        },
        "level": level,
    }


ENRICHERS = {
    "race_shutuba": enrich_race_shutuba,
    "race_result": enrich_race_result,
    "race_shutuba_past": enrich_race_shutuba_past,
}


def attach_data_quality(category: str, data: dict[str, Any] | None) -> None:
    if not data or not isinstance(data, dict):
        return
    fn = ENRICHERS.get(category)
    if fn:
        fn(data)
