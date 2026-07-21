"""
品質チェック向けレース分類 — 障害・埋込 lap 等の N/A ルール。

めぐ指数監査 (audit_gcs_race_result) とは別に、
モニター品質ヘルスではデータ特性に基づく除外を適用する。
"""

from __future__ import annotations

from typing import Any

from src.pipeline.megu_index.flat_metadata import is_obstacle_race_name

# presence: 障害レースでは独立 blob 不要（データ特性）
OBSTACLE_PRESENCE_NA_CATEGORIES: frozenset[str] = frozenset({
    "race_result_lap",
    "race_result_lap_times",
    "race_barometer",
    "race_index",
    "race_paddock",
})

# race_result.lap_times があれば別 blob 不要
LAP_EMBEDDED_SATISFIES_CATEGORY = "race_result_lap"


def race_name_from_cards(*cards: dict | None) -> str:
    for card in cards:
        if not card:
            continue
        name = card.get("race_name") or card.get("name") or ""
        if name:
            return str(name)
    return ""


def is_obstacle_race(rr: dict | None = None, shutuba: dict | None = None) -> bool:
    """race_result / shutuba から障害レース判定。"""
    name = race_name_from_cards(rr, shutuba)
    if is_obstacle_race_name(name):
        return True
    surf = (rr or {}).get("surface") or (shutuba or {}).get("surface") or ""
    # 障害は surface が「障害」等になることが多い
    if "障" in str(surf):
        return True
    return False


def audit_gcs_race_result_for_health(
    rr: dict | None,
    *,
    obstacle: bool | None = None,
) -> list[str]:
    """
    品質ヘルス用 GCS race_result 監査。
    障害レースは lap/meta 欠損を異常とみなさない。
    結果本体（入着+lap）が揃い surface のみ欠損の場合も N/A。
    """
    from src.scripts.data.megu_index_scrape_readiness_audit import (
        _finisher_count,
        audit_gcs_race_result,
    )

    if rr is None:
        return ["gcs_missing"]
    if obstacle is None:
        obstacle = is_obstacle_race(rr)
    if obstacle:
        return []

    gaps = list(audit_gcs_race_result(rr))
    entries = rr.get("entries") or rr.get("results") or []
    has_finishers = _finisher_count(entries) > 0

    # 入着・lap があればメタ欠損のみはデータ運用上 N/A（再スクレイプ優先度低）
    if has_finishers and rr.get("lap_times") and "gcs_bad_meta" in gaps:
        gaps = [g for g in gaps if g != "gcs_bad_meta"]
    if has_finishers and gaps == ["gcs_bad_meta"]:
        gaps = []

    return gaps


def apply_obstacle_presence_na(
    row: dict[str, bool | None],
    *,
    obstacle: bool,
) -> dict[str, bool | None]:
    if not obstacle:
        return row
    out = dict(row)
    for cat in OBSTACLE_PRESENCE_NA_CATEGORIES:
        if out.get(cat) is False:
            out[cat] = None
    return out


def apply_embedded_lap_presence_na(
    row: dict[str, bool | None],
    race_id: str,
    storage: Any,
    *,
    rr: dict | None = None,
) -> dict[str, bool | None]:
    """race_result.lap_times があれば race_result_lap 欠損を N/A に。"""
    out = dict(row)
    if out.get(LAP_EMBEDDED_SATISFIES_CATEGORY) is not False:
        return out
    if out.get("race_result") is not True:
        return out
    if rr is None:
        try:
            rr = storage.load("race_result", race_id)
        except Exception:
            rr = None
    if rr and rr.get("lap_times"):
        out[LAP_EMBEDDED_SATISFIES_CATEGORY] = None
    return out


def enrich_presence_row(
    row: dict[str, bool | None],
    race_id: str,
    storage: Any,
    *,
    rr: dict | None = None,
    shutuba: dict | None = None,
) -> dict[str, bool | None]:
    """presence マトリクス行に N/A ルールを適用。"""
    from src.scraper.date_coverage import apply_derived_category_na

    if rr is None and shutuba is None:
        try:
            rr = storage.load("race_result", race_id)
        except Exception:
            rr = None
        if not rr:
            try:
                shutuba = storage.load("race_shutuba", race_id)
            except Exception:
                shutuba = None

    obstacle = is_obstacle_race(rr, shutuba)
    out = apply_derived_category_na(row)
    out = apply_embedded_lap_presence_na(out, race_id, storage, rr=rr)
    out = apply_obstacle_presence_na(out, obstacle=obstacle)
    return out
