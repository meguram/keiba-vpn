"""
要件書 `docs/requirements/data/scrape_process.md` の netkeiba 表（1 行＝1 レコード）と
正本ストレージ（HybridStorage / HtmlArchive）の対応をコード上の単一ソースとする。

各行は `row_id` で識別し、`requirement_row_trace` カテゴリに 1 JSON ずつマテリアライズできる。

行固有カテゴリ（派生）: race_shutuba 等の canonical JSON から必要フィールドを抽出した
派生カテゴリが `src/scraper/row_data_extractor.py` で定義されており、
`src/scripts/scraping/migrate_row_data_to_unique_paths.py` で生成する。
各 `CanonicalRef.category` はこの派生カテゴリを指すため、行ごとに GCS パスが一意になる。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Scope = Literal["race", "horse", "date"]


@dataclass(frozen=True, slots=True)
class CanonicalRef:
    """HybridStorage の参照。"""

    category: str
    """storage.load の第1引数。"""
    key_field: Literal["race_id", "horse_id", "date_fmt"]
    role: str = "json"
    """同一ファイルを複数行で指すときの役割ラベル。"""


@dataclass(frozen=True, slots=True)
class HtmlArchiveRef:
    """HtmlArchive（生 HTML gzip）の参照。無い行は空リスト。"""

    category: str
    key_field: Literal["race_id", "horse_id", "date_fmt"]
    role: str = "html.gz"


@dataclass(frozen=True, slots=True)
class RequirementRowSpec:
    row_id: str
    title_ja: str
    scope: Scope
    canonical: tuple[CanonicalRef, ...]
    raw_html: tuple[HtmlArchiveRef, ...] = ()


def trace_storage_key(scope: Scope, primary_id: str, row_id: str) -> str:
    """GCS `others/requirement_row_trace/{key}.json` および load/save の key。"""
    return f"{scope}_{primary_id}_{row_id}"


def primary_id_for_scope(scope: Scope, race_id: str, horse_id: str, date_fmt: str) -> str:
    if scope == "race":
        return race_id
    if scope == "horse":
        return horse_id
    return date_fmt


def NETKEIBA_REQUIREMENT_ROWS() -> tuple[RequirementRowSpec, ...]:
    """MD 表「netkeiba」セクションの行と同等の集合（順序は MD に近い）。"""
    r = CanonicalRef
    h = HtmlArchiveRef
    return (
        RequirementRowSpec(
            "nk_shutuba_entries",
            "出馬表HTML",
            "race",
            (r("race_shutuba", "race_id", "entries_table"),),
            (h("race_shutuba", "race_id"),),
        ),
        RequirementRowSpec(
            "nk_shutuba_race_meta",
            "レース情報HTML",
            "race",
            (r("race_shutuba_meta", "race_id"),),
            (h("race_shutuba", "race_id"),),
        ),
        RequirementRowSpec(
            "nk_speed_index",
            "レースタイム指数HTML",
            "race",
            (r("race_index", "race_id"),),
            (h("race_index", "race_id"),),
        ),
        RequirementRowSpec(
            "nk_barometer",
            "レース調子偏差値HTML",
            "race",
            (r("race_barometer", "race_id"),),
            (),
        ),
        RequirementRowSpec(
            "nk_paddock",
            "レースパドックHTML",
            "race",
            (r("race_paddock", "race_id"),),
            (h("race_paddock", "race_id"),),
        ),
        RequirementRowSpec(
            "nk_odds",
            "レースオッズHTML",
            "race",
            (r("race_odds", "race_id"),),
            (),
        ),
        RequirementRowSpec(
            "nk_result_on_time",
            "レース結果HTML",
            "race",
            (r("race_result_on_time", "race_id"),),
            (h("race_result_on_time", "race_id"),),
        ),
        RequirementRowSpec(
            "nk_payoff_html",
            "レース払戻HTML",
            "race",
            (r("race_result_on_time_payoff", "race_id"),),
            (h("race_result_on_time", "race_id"),),
        ),
        RequirementRowSpec(
            "nk_lap_html",
            "レースラップHTML",
            "race",
            (r("race_result_on_time_lap", "race_id"),),
            (h("race_result_on_time", "race_id"),),
        ),
        RequirementRowSpec(
            "nk_corner_html",
            "レース通過順位HTML",
            "race",
            (r("race_result_on_time_corner", "race_id"),),
            (h("race_result_on_time", "race_id"),),
        ),
        RequirementRowSpec(
            "nk_per_horse_lap_html",
            "レース個別ラップHTML",
            "race",
            (
                r("race_result_on_time", "race_id", "result_page"),
                r("race_result_lap", "race_id", "per_horse_lap"),
            ),
            (h("race_result", "race_id", "db_result_page_html"),),
        ),
        RequirementRowSpec(
            "nk_horse_profile",
            "馬プロフィール",
            "horse",
            (r("horse_profile", "horse_id"),),
            (h("horse_profile", "horse_id"),),
        ),
        RequirementRowSpec(
            "nk_horse_history",
            "馬過去成績",
            "horse",
            (r("horse_race_history", "horse_id"),),
            (h("horse_result_html", "horse_id"),),
        ),
        RequirementRowSpec(
            "nk_horse_pedigree",
            "馬血統データ",
            "horse",
            (r("horse_pedigree_5gen", "horse_id"),),
            (h("horse_ped", "horse_id"),),
        ),
        RequirementRowSpec(
            "nk_horse_training",
            "馬調教",
            "horse",
            (r("horse_training", "horse_id"),),
            (h("horse_training", "horse_id", "paged"),),
        ),
        RequirementRowSpec(
            "nk_race_list",
            "レースID一覧",
            "date",
            (r("race_lists", "date_fmt"),),
            (h("race_lists", "date_fmt"),),
        ),
        RequirementRowSpec(
            "nk_race_day_schedule",
            "レース発走時間",
            "date",
            (r("race_day_schedule", "date_fmt"),),
            (),
        ),
        RequirementRowSpec(
            "nk_db_race_result",
            "レース結果DB",
            "race",
            (r("race_result", "race_id"),),
            (h("race_result", "race_id"),),
        ),
        RequirementRowSpec(
            "nk_db_race_info",
            "レース情報DB",
            "race",
            (r("race_result_meta", "race_id"),),
            (h("race_result", "race_id"),),
        ),
        RequirementRowSpec(
            "nk_db_payoff",
            "レース払戻DB",
            "race",
            (r("race_result_payoff", "race_id"),),
            (h("race_result", "race_id"),),
        ),
        RequirementRowSpec(
            "nk_db_track",
            "レース馬場情報DB",
            "race",
            (r("race_result_track", "race_id"),),
            (h("race_result", "race_id"),),
        ),
        RequirementRowSpec(
            "nk_db_corner",
            "レース通過順位DB",
            "race",
            (r("race_result_corner", "race_id"),),
            (h("race_result", "race_id"),),
        ),
        RequirementRowSpec(
            "nk_db_lap",
            "レースラップDB",
            "race",
            (r("race_result_lap_times", "race_id"),),
            (h("race_result", "race_id"),),
        ),
        RequirementRowSpec(
            "nk_db_per_horse_lap",
            "レース個別ラップDB",
            "race",
            (r("race_result_lap", "race_id"),),
            (h("race_result", "race_id"),),
        ),
    )


def row_ids_netkeiba() -> frozenset[str]:
    return frozenset(s.row_id for s in NETKEIBA_REQUIREMENT_ROWS())


def resolve_storage_key(
    key_field: Literal["race_id", "horse_id", "date_fmt"],
    *,
    race_id: str,
    horse_id: str,
    date_fmt: str,
) -> str:
    if key_field == "race_id":
        return race_id
    if key_field == "horse_id":
        return horse_id
    return date_fmt


def build_trace_payload(
    spec: RequirementRowSpec,
    *,
    race_id: str,
    horse_id: str,
    date_fmt: str,
    presence: dict[tuple[str, str], bool] | None = None,
) -> dict[str, object]:
    """`requirement_row_trace` に save する dict（_meta は HybridStorage.save が付与）。"""
    pid = primary_id_for_scope(spec.scope, race_id, horse_id, date_fmt)
    tk = trace_storage_key(spec.scope, pid, spec.row_id)
    canonical: list[dict[str, object]] = []
    for c in spec.canonical:
        k = resolve_storage_key(
            c.key_field, race_id=race_id, horse_id=horse_id, date_fmt=date_fmt
        )
        item: dict[str, object] = {
            "category": c.category,
            "key": k,
            "role": c.role,
        }
        if presence is not None:
            item["present"] = bool(presence.get((c.category, k)))
        canonical.append(item)
    raw_html: list[dict[str, object]] = []
    for h in spec.raw_html:
        k = resolve_storage_key(
            h.key_field, race_id=race_id, horse_id=horse_id, date_fmt=date_fmt
        )
        if h.role == "paged" and h.key_field == "horse_id":
            k = f"{horse_id}_p1"
        item: dict[str, object] = {"category": h.category, "key": k, "role": h.role}
        if h.role == "paged":
            item["note"] = "複数ページ時は {horse_id}_p2 … も参照（HtmlArchive）"
        if presence is not None:
            item["present"] = bool(presence.get((h.category, k)))
        raw_html.append(item)
    return {
        "row_id": spec.row_id,
        "title_ja": spec.title_ja,
        "trace_key": tk,
        "scope": spec.scope,
        "primary_id": pid,
        "canonical": canonical,
        "raw_html": raw_html,
    }


def probe_canonical_presence(
    storage: object,
    spec: RequirementRowSpec,
    race_id: str,
    horse_id: str,
    date_fmt: str,
) -> dict[tuple[str, str], bool]:
    """HybridStorage.load の有無（truthy）で canonical の存在を記録。"""
    out: dict[tuple[str, str], bool] = {}
    for c in spec.canonical:
        k = resolve_storage_key(
            c.key_field, race_id=race_id, horse_id=horse_id, date_fmt=date_fmt
        )
        try:
            data = storage.load(c.category, k)
        except Exception:
            data = None
        out[(c.category, k)] = bool(data)
    return out
