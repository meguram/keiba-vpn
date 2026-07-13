"""
race_lists の充足判定と、段階取得時の上書き可否。

開催日ごとの JRA レースは会場あたり 12R が基本（1〜3 会場で 12 / 24 / 36）。
netkeiba 側の掲載タイミングにより途中経過では 12 の倍数に満たないことがある。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.scraper.missing_races import is_jra_race_id

RACES_PER_VENUE = 12


@dataclass(frozen=True)
class RaceListStats:
    jra_count: int
    venue_codes: tuple[str, ...]
    per_venue_counts: dict[str, int]
    per_venue_rounds: dict[str, tuple[int, ...]]
    is_complete: bool
    reason: str


def _race_round(race: dict[str, Any]) -> int | None:
    r = race.get("round")
    if r is not None:
        try:
            rv = int(r)
            return rv if 1 <= rv <= RACES_PER_VENUE else None
        except (TypeError, ValueError):
            pass
    rid = str(race.get("race_id") or "").strip()
    if len(rid) >= 12 and rid[10:12].isdigit():
        rv = int(rid[10:12])
        return rv if 1 <= rv <= RACES_PER_VENUE else None
    return None


def _venue_code(race: dict[str, Any]) -> str | None:
    rid = str(race.get("race_id") or "").strip()
    if len(rid) >= 6 and is_jra_race_id(rid):
        return rid[4:6]
    return None


def jra_races_from_list_data(data: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not data or not isinstance(data, dict):
        return []
    races = data.get("races")
    if not isinstance(races, list):
        return []
    out: list[dict[str, Any]] = []
    for r in races:
        if not isinstance(r, dict):
            continue
        rid = str(r.get("race_id") or "").strip()
        if rid and is_jra_race_id(rid):
            out.append(r)
    return out


def race_list_stats(data: dict[str, Any] | None) -> RaceListStats:
    if not data or not isinstance(data, dict):
        return RaceListStats(0, (), {}, {}, False, "データなし")

    meta = data.get("_meta") if isinstance(data.get("_meta"), dict) else {}
    if meta.get("note") == "no_race_scheduled":
        return RaceListStats(0, (), {}, {}, True, "開催なし")

    jra = jra_races_from_list_data(data)
    if not jra:
        return RaceListStats(0, (), {}, {}, False, "JRAレース0件")

    per_venue_rounds: dict[str, set[int]] = {}
    for race in jra:
        vc = _venue_code(race)
        rd = _race_round(race)
        if not vc or rd is None:
            continue
        per_venue_rounds.setdefault(vc, set()).add(rd)

    per_venue_counts = {vc: len(rds) for vc, rds in per_venue_rounds.items()}
    venue_codes = tuple(sorted(per_venue_rounds))
    jra_count = len(jra)

    if jra_count % RACES_PER_VENUE != 0:
        return RaceListStats(
            jra_count,
            venue_codes,
            per_venue_counts,
            {vc: tuple(sorted(rds)) for vc, rds in per_venue_rounds.items()},
            False,
            f"合計{jra_count}件（12の倍数でない）",
        )

    if not per_venue_rounds:
        return RaceListStats(
            jra_count,
            venue_codes,
            per_venue_counts,
            {},
            False,
            "会場別ラウンドを解釈できない",
        )

    for vc, rds in per_venue_rounds.items():
        if len(rds) != RACES_PER_VENUE:
            return RaceListStats(
                jra_count,
                venue_codes,
                per_venue_counts,
                {v: tuple(sorted(s)) for v, s in per_venue_rounds.items()},
                False,
                f"会場{vc}: {len(rds)}/{RACES_PER_VENUE}R",
            )
        if rds != set(range(1, RACES_PER_VENUE + 1)):
            return RaceListStats(
                jra_count,
                venue_codes,
                per_venue_counts,
                {v: tuple(sorted(s)) for v, s in per_venue_rounds.items()},
                False,
                f"会場{vc}: 1〜12Rが揃っていない",
            )

    return RaceListStats(
        jra_count,
        venue_codes,
        per_venue_counts,
        {vc: tuple(sorted(rds)) for vc, rds in per_venue_rounds.items()},
        True,
        f"完備（{len(venue_codes)}会場×{RACES_PER_VENUE}R={jra_count}件）",
    )


def is_race_list_complete(data: dict[str, Any] | None) -> bool:
    return race_list_stats(data).is_complete


def should_replace_race_list(
    existing: dict[str, Any] | None,
    new: dict[str, Any],
) -> bool:
    """
    段階取得中は件数が増える取得結果で上書きする。
    完備済みを不完全な取得で上書きしない。
  """
    ex_stats = race_list_stats(existing)
    new_stats = race_list_stats(new)

    if new_stats.jra_count == 0 and not (
        isinstance(new.get("_meta"), dict) and new["_meta"].get("note") == "no_race_scheduled"
    ):
        return ex_stats.jra_count == 0

    if ex_stats.jra_count == 0:
        return True

    if ex_stats.is_complete and not new_stats.is_complete:
        return False

    if new_stats.is_complete and not ex_stats.is_complete:
        return True

    if new_stats.is_complete and ex_stats.is_complete:
        return new_stats.jra_count >= ex_stats.jra_count

    # いずれも未完了: 件数が増えたときだけ上書き（同数は新しい方を採用）
    return new_stats.jra_count >= ex_stats.jra_count


def merge_race_list_payload(
    date: str,
    races: list[dict],
    *,
    source: str = "",
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"date": date, "races": races}
    meta: dict[str, Any] = dict(extra_meta or {})
    if source:
        meta["race_list_source"] = source
    stats = race_list_stats(payload)
    meta["race_list_jra_count"] = stats.jra_count
    meta["race_list_complete"] = stats.is_complete
    meta["race_list_complete_reason"] = stats.reason
    if stats.venue_codes:
        meta["race_list_venues"] = list(stats.venue_codes)
    payload["_meta"] = meta
    return payload
