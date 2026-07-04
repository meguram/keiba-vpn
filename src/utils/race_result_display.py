"""レース結果ページ表示用の race_result 補完。"""

from __future__ import annotations

from typing import Any


def merge_race_result_entries_lap(
    entries: list[dict[str, Any]] | None,
    entries_lap: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """entries_lap の通過・上り3F を馬番でマージする。"""
    by_num: dict[int, dict[str, Any]] = {}
    for le in entries_lap or []:
        hn = le.get("horse_number")
        if hn is None:
            continue
        try:
            by_num[int(hn)] = le
        except (TypeError, ValueError):
            continue

    out: list[dict[str, Any]] = []
    for e in entries or []:
        row = dict(e)
        try:
            hn = int(row.get("horse_number") or row.get("number") or 0)
        except (TypeError, ValueError):
            hn = 0
        le = by_num.get(hn) if hn else None
        if le:
            if not row.get("passing_order") and le.get("passing_order"):
                row["passing_order"] = le["passing_order"]
            last3f = row.get("last_3f")
            if (last3f is None or last3f == 0) and le.get("last_3f"):
                row["last_3f"] = le["last_3f"]
        out.append(row)
    return out


# netkeiba 成績ページの券種名（日本語キーが正）
_PAYOFF_DISPLAY_ORDER = (
    "単勝",
    "複勝",
    "枠連",
    "馬連",
    "ワイド",
    "馬単",
    "三連複",
    "三連単",
)

_PAYOFF_KEY_ALIASES: dict[str, str] = {
    "tansho": "単勝",
    "fukusho": "複勝",
    "wakuren": "枠連",
    "umaren": "馬連",
    "wide": "ワイド",
    "umatan": "馬単",
    "trio": "三連複",
    "tierce": "三連単",
}


def normalize_payoff_for_display(payoff: Any) -> dict[str, Any] | None:
    """払戻 dict を表示用に正規化（日本語キー・表示順）。"""
    if not payoff or not isinstance(payoff, dict):
        return None
    merged: dict[str, Any] = {}
    for raw_key, value in payoff.items():
        if value is None or value == "":
            continue
        key = _PAYOFF_KEY_ALIASES.get(str(raw_key).strip(), str(raw_key).strip())
        if key not in merged:
            merged[key] = value
    if not merged:
        return None
    ordered: dict[str, Any] = {}
    for key in _PAYOFF_DISPLAY_ORDER:
        if key in merged:
            ordered[key] = merged[key]
    for key, value in merged.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def merge_race_result_payoff(
    race_result: dict[str, Any] | None,
    race_result_on_time: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """race_result / on_time のいずれかから payoff を拾う。"""
    raw: dict[str, Any] = {}
    for src in (race_result, race_result_on_time):
        if not src:
            continue
        pf = src.get("payoff")
        if isinstance(pf, dict):
            raw.update(pf)
    return normalize_payoff_for_display(raw) if raw else None


def prepare_race_result_display(
    race_result: dict[str, Any] | None,
    race_result_on_time: dict[str, Any] | None,
    race_result_lap: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """race_result / on_time / lap を統合し UI 向けに正規化する。"""
    from src.scraper.pace_utils import merge_race_result_pace

    base = race_result if (race_result and race_result.get("entries")) else None
    if not base and race_result_on_time and race_result_on_time.get("entries"):
        base = race_result_on_time
    if not base:
        return None

    out = merge_race_result_pace(dict(base), race_result_lap) or dict(base)
    out["entries"] = merge_race_result_entries_lap(
        out.get("entries"),
        (race_result_lap or {}).get("entries_lap"),
    )
    payoff = merge_race_result_payoff(race_result, race_result_on_time)
    if payoff:
        out["payoff"] = payoff
    from src.utils.lap_pattern import attach_lap_pattern_to_result

    attach_lap_pattern_to_result(out)
    out["_display_source"] = (
        "race_result"
        if race_result and race_result.get("entries")
        else "race_result_on_time"
    )
    return out
