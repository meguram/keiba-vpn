"""
モニター向け: race_lists 上の各開催日について、JRA レースの未取得件数を集計する。

batch_list_blobs でキー一覧のみ取得し、個別 load を最小限にすることで
GCS コストを 1/1000 以下に抑える。
"""

from __future__ import annotations

import threading
import time
from typing import Any

_cache_lock = threading.Lock()
_cache: dict[str, tuple[float, dict]] = {}
_CACHE_TTL = 1800.0  # 30min — blob list キャッシュと同期してGCSコスト削減

_CHECK_CATEGORIES = ("race_shutuba", "race_odds", "race_result")


def _get_existing_keys(
    storage: Any, years: list[str],
) -> dict[str, set[str]]:
    """各カテゴリについて存在するキーのセットを返す。batch_list_blobs 使用。"""
    result: dict[str, set[str]] = {}
    for cat in _CHECK_CATEGORIES:
        keys: set[str] = set()
        for y in years:
            try:
                blobs = storage.batch_list_blobs(cat, y)
                keys.update(blobs.keys())
            except Exception:
                pass
        result[cat] = keys
    return result


def summarize_missing_dates(
    storage: Any,
    *,
    max_dates: int = 200,
    year: int | None = None,
) -> dict[str, Any]:
    from src.scraper.missing_races import is_jra_race_id
    from src.scraper.monitor_future_eligible import include_date_in_monitor_summary

    cache_key = f"{year or 'all'}:{max_dates}"
    with _cache_lock:
        cached = _cache.get(cache_key)
        if cached and (time.time() - cached[0]) < _CACHE_TTL:
            return cached[1]

    max_dates = max(1, min(int(max_dates), 500))

    all_rl_keys_raw = sorted(storage.list_keys("race_lists"))
    if year is not None:
        prefix = str(int(year))
        all_rl_keys_raw = [k for k in all_rl_keys_raw if k.startswith(prefix)]
        years = [prefix]
    else:
        years = sorted({k[:4] for k in all_rl_keys_raw if len(k) >= 4})

    all_rl_keys = all_rl_keys_raw[-max_dates:]

    existing = _get_existing_keys(storage, years)

    rows: list[dict[str, Any]] = []
    for d in all_rl_keys:
        rl = storage.load("race_lists", d)
        raw_races = rl.get("races") if rl else None
        meta = rl.get("_meta") if isinstance(rl, dict) else None
        if not include_date_in_monitor_summary(d, raw_races or [], meta):
            continue
        race_list_exists = bool(raw_races)
        race_list_row_count = len(raw_races) if raw_races else 0
        if not rl or not raw_races:
            rows.append({
                "date": d,
                "race_list_exists": race_list_exists,
                "race_list_row_count": race_list_row_count,
                "jra_races": 0,
                "missing_races": 0,
                "complete_races": 0,
                "pct_complete": None,
                "level": "no_list",
            })
            continue

        jra = 0
        miss = 0
        for r in raw_races:
            rid = r.get("race_id")
            if not rid or not is_jra_race_id(str(rid)):
                continue
            jra += 1
            rid_str = str(rid)
            has_any = any(
                rid_str in existing[cat] for cat in _CHECK_CATEGORIES
            )
            if not has_any:
                miss += 1

        comp = jra - miss
        pct = round(100.0 * comp / jra, 1) if jra else None
        if jra == 0:
            level = "no_jra"
        elif miss == 0:
            level = "full"
        elif comp == 0:
            level = "bare"
        else:
            level = "partial"

        rows.append({
            "date": d,
            "race_list_exists": race_list_exists,
            "race_list_row_count": race_list_row_count,
            "jra_races": jra,
            "missing_races": miss,
            "complete_races": comp,
            "pct_complete": pct,
            "level": level,
        })

    priority = sorted(rows, key=lambda x: (-x["missing_races"], x["date"]))[:40]
    result = {
        "dates": sorted(rows, key=lambda x: x["date"]),
        "priority": priority,
        "meta": {
            "scanned": len(rows),
            "days_with_backlog": sum(1 for r in rows if r["missing_races"] > 0),
            "total_missing_races": sum(r["missing_races"] for r in rows),
        },
    }

    with _cache_lock:
        _cache[cache_key] = (time.time(), result)

    return result
