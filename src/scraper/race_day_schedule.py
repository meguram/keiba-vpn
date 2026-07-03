"""
開催日単位の発走時刻表（`race_day_schedule` カテゴリ）。

従来は `race_lists` + 各 `race_shutuba` を都度読んで合成していたが、
確定スナップショットを `data/page_reference/race_day_schedule/{YYYYMMDD}.json` に保存し、
存在すれば `_fetch_race_schedule_storage` がそれを優先する。
"""

from __future__ import annotations

import time
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

_JST = ZoneInfo("Asia/Tokyo")

_STORAGE_FN = Any


def _race_day_from_fmt(date_fmt: str) -> date:
    ys, ms, ds = int(date_fmt[:4]), int(date_fmt[4:6]), int(date_fmt[6:8])
    return date(ys, ms, ds)


def synthesize_race_day_schedule_payload(storage: _STORAGE_FN, date_fmt: str) -> dict[str, Any]:
    """race_lists + race_shutuba から slots を構築（post_time は ISO8601 文字列）。"""
    rl = storage.load("race_lists", date_fmt)
    if not rl:
        return {
            "date_fmt": date_fmt,
            "slots": [],
            "_meta": {
                "source": "synthesized",
                "note": "no_race_lists",
                "generated_at": time.time(),
            },
        }

    races = rl.get("races") or []
    try:
        race_day = _race_day_from_fmt(date_fmt)
    except (ValueError, IndexError):
        race_day = datetime.now(_JST).date()

    slots: list[dict[str, Any]] = []
    for race in races:
        rid = race.get("race_id")
        if not rid or not isinstance(rid, str):
            continue
        rid = rid.strip()
        if not rid:
            continue
        card = storage.load("race_shutuba", rid) or {}
        start_str = str(card.get("start_time") or "").strip()
        source = "shutuba"

        if not start_str:
            rnd = race.get("round", 0)
            if isinstance(rnd, str):
                rnd = int(rnd) if rnd.isdigit() else 0
            elif not isinstance(rnd, int):
                rnd = 0
            if rnd <= 6:
                h = 9 + (rnd * 30) // 60
                m = 45 + (rnd * 30) % 60
            else:
                h = 12 + ((rnd - 5) * 30) // 60
                m = ((rnd - 5) * 30) % 60
            start_str = f"{h:02d}:{m:02d}"
            source = "estimated_round"

        try:
            h, m = map(int, start_str.split(":"))
            post_dt = datetime.combine(
                race_day,
                datetime.min.time().replace(hour=h, minute=m),
                tzinfo=_JST,
            )
        except (ValueError, TypeError):
            continue

        slots.append(
            {
                "race_id": rid,
                "venue": race.get("venue", card.get("venue", "") if card else ""),
                "round": race.get("round", ""),
                "race_name": race.get(
                    "race_name", card.get("race_name", "") if card else ""
                ),
                "start_time_str": start_str,
                "post_time_iso": post_dt.isoformat(),
                "time_source": source,
            }
        )

    slots.sort(key=lambda x: x.get("post_time_iso") or "")
    return {
        "date_fmt": date_fmt,
        "iso_date": race_day.isoformat(),
        "slots": slots,
        "_meta": {
            "source": "synthesized",
            "built_from": "race_lists+race_shutuba",
            "generated_at": time.time(),
        },
    }


def schedule_payload_to_runtime_list(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """キュー・ランナー用: post_time を aware datetime に戻す。"""
    out: list[dict[str, Any]] = []
    for row in payload.get("slots") or []:
        rid = row.get("race_id")
        if not rid:
            continue
        iso = str(row.get("post_time_iso") or "").strip()
        if not iso:
            continue
        try:
            post_dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
            if post_dt.tzinfo is None:
                post_dt = post_dt.replace(tzinfo=_JST)
        except (ValueError, TypeError):
            continue
        out.append(
            {
                "race_id": rid,
                "venue": row.get("venue", ""),
                "round": row.get("round", ""),
                "race_name": row.get("race_name", ""),
                "post_time": post_dt,
                "start_time_str": str(row.get("start_time_str") or ""),
            }
        )
    out.sort(key=lambda x: x["post_time"])
    return out
