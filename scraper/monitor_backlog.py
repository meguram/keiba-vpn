"""
モニター向け: race_lists 上の各開催日について、JRA レースの「一式未取得」件数を集計する。

判定は missing_races.scrape_status_detail と同一（出馬表・オッズ・結果＋代表馬の horse_result）。
"""

from __future__ import annotations

from typing import Any


def summarize_missing_dates(
    storage: Any,
    *,
    max_dates: int = 200,
    year: int | None = None,
) -> dict[str, Any]:
    from scraper.missing_races import is_jra_race_id, scrape_status_detail
    from scraper.monitor_future_eligible import include_date_in_monitor_summary

    max_dates = max(1, min(int(max_dates), 500))
    keys = sorted(storage.list_keys("race_lists"))
    if year is not None:
        y = str(int(year))
        keys = [k for k in keys if k.startswith(y)]
        keys = keys[-max_dates:]
    else:
        keys = keys[-max_dates:]

    rows: list[dict[str, Any]] = []
    for d in keys:
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
            if not scrape_status_detail(storage, str(rid)).get("scraped"):
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
    return {
        "dates": sorted(rows, key=lambda x: x["date"]),
        "priority": priority,
        "meta": {
            "scanned": len(rows),
            "days_with_backlog": sum(1 for r in rows if r["missing_races"] > 0),
            "total_missing_races": sum(r["missing_races"] for r in rows),
        },
    }
