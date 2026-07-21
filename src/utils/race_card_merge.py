"""race_shutuba と race_result のマージ（文字化け・欠損の補完）。"""

from __future__ import annotations

import re
from typing import Any

_CJK_RE = re.compile(r"[\u3040-\u9fff\u30a0-\u30ff]")
_SEX_AGE_RE = re.compile(r"^[牡牝セン]")

RACE_META_KEYS = (
    "race_name",
    "venue",
    "surface",
    "distance",
    "direction",
    "weather",
    "track_condition",
    "grade",
    "race_class",
    "date",
    "start_time",
    "entries_count",
)


def is_plausible_sex_age(value: str | None) -> bool:
    return bool(value and _SEX_AGE_RE.match(str(value).strip()))


def is_plausible_label(value: str | None, *, min_cjk: int = 1) -> bool:
    if not value or not str(value).strip():
        return False
    text = str(value).strip()
    if "\ufffd" in text:
        return False
    return len(_CJK_RE.findall(text)) >= min_cjk


def _cjk_count(value: str | None) -> int:
    if not value:
        return 0
    return len(_CJK_RE.findall(str(value)))


def pick_better_text(primary: str | None, fallback: str | None) -> str | None:
    primary_s = str(primary).strip() if primary else ""
    fallback_s = str(fallback).strip() if fallback else ""
    primary_ok = is_plausible_label(primary_s)
    fallback_ok = is_plausible_label(fallback_s)
    if fallback_ok and not primary_ok:
        return fallback_s
    if primary_ok and not fallback_ok:
        return primary_s
    if fallback_ok and primary_ok:
        if _cjk_count(fallback_s) > _cjk_count(primary_s):
            return fallback_s
        return primary_s
    return primary_s or fallback_s or None


def _entry_list(card: dict | None) -> list[dict[str, Any]]:
    if not card:
        return []
    return list(card.get("entries") or card.get("results") or [])


def merge_race_card(shutuba: dict | None, result: dict | None) -> dict | None:
    """shutuba をベースに race_result で欠損・文字化けを補完する。"""
    if not shutuba and not result:
        return None

    card = dict(shutuba) if shutuba else {}
    if not result:
        return card or None

    for key in RACE_META_KEYS:
        result_val = result.get(key)
        if result_val is None or result_val == "":
            continue
        if key == "distance":
            current = int(card.get("distance") or 0)
            if current <= 0 and int(result_val) > 0:
                card["distance"] = result_val
            continue
        if key == "surface":
            if not str(card.get("surface") or "").strip():
                card["surface"] = result_val
            continue
        if key in ("race_name", "venue", "grade", "race_class", "track_condition", "weather"):
            card[key] = pick_better_text(card.get(key), result_val) or card.get(key) or result_val
            continue
        if not card.get(key):
            card[key] = result_val

    result_by_horse = {
        str(entry.get("horse_id")): entry
        for entry in _entry_list(result)
        if entry.get("horse_id")
    }
    entries = list(card.get("entries") or [])
    if not entries and result_by_horse:
        card["entries"] = list(result_by_horse.values())
        return card

    for entry in entries:
        horse_id = str(entry.get("horse_id") or "")
        result_entry = result_by_horse.get(horse_id)
        if not result_entry:
            continue
        entry["horse_name"] = (
            pick_better_text(entry.get("horse_name"), result_entry.get("horse_name"))
            or entry.get("horse_name")
        )
        if not is_plausible_sex_age(entry.get("sex_age")):
            if is_plausible_sex_age(result_entry.get("sex_age")):
                entry["sex_age"] = result_entry["sex_age"]
    card["entries"] = entries
    return card


def patch_result_metadata_from_shutuba(result: dict | None, shutuba: dict | None) -> dict | None:
    """race_result の surface/distance 等を race_shutuba（いずれもスクレイプ正本）で補完。"""
    if not result:
        return result
    if not shutuba:
        return result
    out = dict(result)
    for key in RACE_META_KEYS:
        rv = out.get(key)
        sv = shutuba.get(key)
        if sv is None or sv == "":
            continue
        if key == "distance":
            if int(out.get("distance") or 0) <= 0 and int(sv) > 0:
                out["distance"] = sv
            continue
        if key == "surface":
            if str(out.get("surface") or "").strip() not in ("芝", "ダート") and sv in ("芝", "ダート"):
                out["surface"] = sv
            continue
        if not rv or rv == "":
            out[key] = sv
    return out


def load_merged_race_card(race_id: str) -> dict | None:
    """GCS から shutuba / result を読み、マージしたレースカードを返す。"""
    try:
        from src.scraper.storage import HybridStorage

        storage = HybridStorage()
        shutuba = storage.load("race_shutuba", race_id)
        result = storage.load("race_result", race_id)
        return merge_race_card(shutuba, result)
    except Exception:
        return None


def patch_race_object(race: Any, meta: dict[str, Any]) -> None:
    """ORM Race 相当のオブジェクトにメタデータを上書き補完する（読み取り専用 API 用）。"""
    if not meta:
        return

    distance = int(meta.get("distance") or 0)
    if distance > 0 and (not getattr(race, "distance", None) or int(race.distance) <= 0):
        race.distance = distance

    for attr in ("race_name", "venue", "surface", "direction", "track_condition", "grade", "race_class"):
        current = getattr(race, attr, None)
        incoming = meta.get(attr)
        if attr in ("race_name", "venue", "grade", "race_class", "track_condition"):
            patched = pick_better_text(current, incoming)
            if patched:
                setattr(race, attr, patched)
        elif incoming and not str(current or "").strip():
            setattr(race, attr, incoming)

    race_date = meta.get("date")
    if race_date and not getattr(race, "race_date", None):
        try:
            from datetime import date, datetime

            if isinstance(race_date, date):
                race.race_date = race_date
            elif len(str(race_date)) == 8 and str(race_date).isdigit():
                race.race_date = datetime.strptime(str(race_date), "%Y%m%d").date()
            else:
                race.race_date = date.fromisoformat(str(race_date))
        except Exception:
            pass


def race_meta_from_card(card: dict | None) -> dict[str, Any]:
    """マージ済みレースカードから API / DB 補完用メタを抽出。"""
    if not card:
        return {}
    return {
        "race_name": card.get("race_name"),
        "venue": card.get("venue"),
        "surface": card.get("surface"),
        "distance": int(card["distance"]) if card.get("distance") else None,
        "track_condition": card.get("track_condition"),
        "grade": card.get("grade"),
        "race_class": card.get("race_class"),
        "start_time": card.get("start_time"),
    }
