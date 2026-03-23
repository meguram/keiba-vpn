"""
race_lists に存在する JRA レースのうち、コアデータ未取得のものを列挙する。

/api/check-scraped-status と同じ基準で「取得済み」を判定する。
"""

from __future__ import annotations

from typing import Any

JRA_PLACE_CODES = frozenset({"01", "02", "03", "04", "05", "06", "07", "08", "09", "10"})


def is_jra_race_id(race_id: str) -> bool:
    return len(race_id) >= 6 and race_id[4:6] in JRA_PLACE_CODES


def scrape_status_detail(storage: Any, race_id: str) -> dict[str, Any]:
    """1 レース分の取得状況（check-scraped-status と同一ロジック）。"""
    data_types: list[str] = []
    check_types = ["race_shutuba", "race_odds", "race_result"]

    for dtype in check_types:
        try:
            data = storage.load(dtype, race_id)
            if data:
                data_types.append(dtype)
        except Exception:
            pass

    if "race_shutuba" in data_types:
        try:
            shutuba = storage.load("race_shutuba", race_id)
            entries = shutuba.get("entries", [])
            if entries and len(entries) > 0:
                first_horse_id = entries[0].get("horse_id")
                if first_horse_id:
                    horse_data = storage.load("horse_result", first_horse_id)
                    if horse_data and "race_history" in horse_data:
                        data_types.append("horse_result")
        except Exception:
            pass

    return {
        "scraped": len(data_types) > 0,
        "data_types": data_types,
        "count": len(data_types),
    }


def find_missing_jra_races_for_queue(
    storage: Any,
    *,
    start_date: str | None,
    end_date: str | None,
    limit: int,
    skip_race_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    race_lists 上の JRA レースで、scrape_status_detail が未取得のものを最大 limit 件返す。

    skip_race_ids: すでにキュー待機・実行中の race_id（重複投入を減らす）
    """
    meta = {"dates_scanned": 0, "races_in_lists": 0, "already_scraped": 0, "skipped_in_queue": 0}
    dates = sorted(storage.list_keys("race_lists"))
    if start_date:
        dates = [d for d in dates if d >= start_date]
    if end_date:
        dates = [d for d in dates if d <= end_date]

    out: list[dict[str, Any]] = []

    for date in dates:
        meta["dates_scanned"] += 1
        rl = storage.load("race_lists", date)
        if not rl:
            continue
        for r in rl.get("races", []):
            rid = r.get("race_id")
            if not rid or not is_jra_race_id(rid):
                continue
            meta["races_in_lists"] += 1
            if rid in skip_race_ids:
                meta["skipped_in_queue"] += 1
                continue
            detail = scrape_status_detail(storage, rid)
            if detail["scraped"]:
                meta["already_scraped"] += 1
                continue
            kaishi = r.get("date") or date
            if isinstance(kaishi, int):
                kaishi = str(kaishi)
            out.append({
                "race_id": rid,
                "date": str(kaishi),
                "venue": r.get("venue", "") or r.get("place_name", ""),
                "round": r.get("round", 0),
                "race_name": r.get("race_name", ""),
            })
            if len(out) >= limit:
                return out, meta

    return out, meta
