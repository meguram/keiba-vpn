"""SmartRC の出走表（runners）の『出走前の値』を、不変のスナップショットとして新カテゴリ ``smartrc_prerace`` に残す。

背景: smartrc_race は夕方（出走後）の再取得で上書きされ、保存データの取得は全件が出走後（104〜2293 日遅れ）だった。馬具（blinker）など
出走前に確定している項目は、出走前に取った値でないと『先読みなし』と検証できない。そこで前日夕方（eve）の取得時に、その時点の runners を
レースごとに 1 文書（phases.eve）として残す。既に同じ phase があれば上書きしない。出走後の取得（発走時刻を過ぎている）は残さない。

文書: {race_id, rcode, source, phases: {eve: {phase, snapshot_at, runners: [...全項目...]}}}
注意: runners の ``old_pr`` のように結果や予想印由来の疑いのある列も、生の記録として残すが、特徴量には使わない（NOT_FOR_FEATURES）。
KEIBA_EVE_SMARTRC_SNAPSHOT=1 のときだけ有効（既定は無効）。
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

logger = logging.getLogger("scraper.smartrc_prerace")

CATEGORY = "smartrc_prerace"
JST = ZoneInfo("Asia/Tokyo")
NOT_FOR_FEATURES = ("old_pr",)
FEATURE_KEYS = ("blinker", "handicap", "invite", "ustat", "tozai", "jcode", "jcode_before", "interval", "age", "sex", "futan", "weight", "vary")


def enabled() -> bool:
    return os.environ.get("KEIBA_EVE_SMARTRC_SNAPSHOT", "0").strip().lower() in ("1", "true", "yes")


def phase_for(date: str, start_time: str = "", now: datetime | None = None) -> str | None:
    """取得時刻から phase を決める。前日以前=eve、当日で発走の 30 分前まで=t60。それ以外（出走後・日付不明）は None（残さない）。"""
    now = now or datetime.now(JST)
    d = (date or "").replace("-", "")
    if len(d) != 8 or not d.isdigit():
        return None
    race_day = datetime.strptime(d, "%Y%m%d").replace(tzinfo=JST)
    if now.date() < race_day.date():
        return "eve"
    if now.date() > race_day.date():
        return None
    try:
        hh, mm = (int(x) for x in str(start_time).split(":")[:2])
    except ValueError:
        return None                                   # 当日で発走時刻が分からない場合は残さない（出走後かもしれない）
    start = race_day.replace(hour=hh, minute=mm)
    return "t60" if now <= start - timedelta(minutes=30) else None


def build_phase_doc(data: dict[str, Any], phase: str, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(JST)
    return {"phase": phase, "snapshot_at": now.isoformat(timespec="seconds"), "runners": [dict(r) for r in data.get("runners") or []]}


def merge_phase(existing: dict[str, Any] | None, race_id: str, data: dict[str, Any], phase: str, now: datetime | None = None) -> dict[str, Any] | None:
    """既に同じ phase があれば None（上書きしない）。無ければ追加した文書を返す。"""
    phases = dict((existing or {}).get("phases") or {})
    if phase in phases:
        return None
    phases[phase] = build_phase_doc(data, phase, now)
    return {"race_id": race_id, "rcode": data.get("rcode") or (existing or {}).get("rcode") or "", "source": "smartrc", "phases": phases}


def save_snapshot(storage: Any, race_id: str, data: dict[str, Any], date: str, start_time: str = "", now: datetime | None = None) -> str | None:
    """保存した phase 名を返す（残さなかった場合は None）。"""
    if not data or not data.get("runners"):
        return None
    phase = phase_for(date, start_time, now)
    if phase is None:
        return None
    existing = storage.load(CATEGORY, race_id, bypass_cache=True)
    doc = merge_phase(existing, race_id, data, phase, now)
    if doc is None:
        return None
    if storage.save(CATEGORY, race_id, doc) is False:
        logger.warning("smartrc_prerace の保存に失敗: %s", race_id)
        return None
    return phase
