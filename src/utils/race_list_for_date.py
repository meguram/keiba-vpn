"""
開催日キー (YYYYMMDD) に対する race_lists の正規読み取り。

race_id は暦日ではなく「年 + 場コード + 開催回 + 日目 + R」形式。
ファイル名 / JSON の date フィールドが開催日（カレンダー日）の正本。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RACE_LIST_DIR = _PROJECT_ROOT / "data/page_reference/race_lists"

VENUE_CODE_TO_NAME = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}


def decode_race_id(race_id: str) -> dict[str, Any]:
    """race_id を分解（先頭8桁は暦日ではない）。"""
    if not race_id or len(race_id) < 12:
        return {}
    return {
        "year": race_id[:4],
        "venue_code": race_id[4:6],
        "venue": VENUE_CODE_TO_NAME.get(race_id[4:6], race_id[4:6]),
        "kaisai_round": int(race_id[6:8]),
        "kaisai_day": int(race_id[8:10]),
        "race_num": int(race_id[10:12]),
    }


def load_race_list_data(date: str) -> dict[str, Any] | None:
    path = RACE_LIST_DIR / f"{date}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("race_list load failed [%s]: %s", date, e)
        return None


def load_jra_race_ids_for_opening_date(
    date: str,
    *,
    require_monitor_eligible: bool = True,
) -> list[str]:
    """
    開催日キーに対応する JRA race_id 一覧（race_lists 正本）。

    date_coverage の stale race_ids は使わない。
    """
    from src.scraper.missing_races import is_jra_race_id
    from src.scraper.monitor_future_eligible import include_date_in_monitor_summary

    data = load_race_list_data(date)
    if not data:
        return []

    raw_races = data.get("races") or []
    meta = data.get("_meta") if isinstance(data.get("_meta"), dict) else {}

    if require_monitor_eligible and not include_date_in_monitor_summary(date, raw_races, meta):
        return []

    return [
        str(r["race_id"])
        for r in raw_races
        if isinstance(r, dict) and r.get("race_id") and is_jra_race_id(str(r["race_id"]))
    ]


def opening_date_kind(date: str) -> str:
    """
    開催日の種別:
      meeting   — JRA レースあり
      no_meeting — 非開催・プレースホルダ
      missing   — race_list ファイルなし
    """
    from src.scraper.monitor_future_eligible import include_date_in_monitor_summary

    data = load_race_list_data(date)
    if not data:
        return "missing"
    raw = data.get("races") or []
    meta = data.get("_meta") if isinstance(data.get("_meta"), dict) else {}
    if include_date_in_monitor_summary(date, raw, meta):
        return "meeting"
    return "no_meeting"


OPENING_KIND_LABELS: dict[str, str] = {
    "meeting": "開催日",
    "no_meeting": "非開催（対象外）",
    "missing": "一覧なし",
}


def opening_date_display(date: str) -> dict[str, Any]:
    """UI 向け: 開催日種別と品質チェック対象か。"""
    kind = opening_date_kind(date)
    return {
        "date": date,
        "kind": kind,
        "label": OPENING_KIND_LABELS.get(kind, kind),
        "quality_applicable": kind == "meeting",
        "monitor_data_applicable": kind == "meeting",
    }
