"""
レースパフォーマンス算出モジュール。

目的:
  - 各馬の「そのレースでどれだけ強い走りをしたか」を run_performance として数値化
  - 各レースの「相手レベル」を pre-race / retrospective の2本で数値化
  - 結果を data/features 配下に再利用しやすい形で保存

出力:
  - data/features/race_performance/{year}.parquet
  - data/features/snapshots/race_performance_YYYY_YYYY.parquet
  - data/features/race_horse_tbl/<YYYY>/*.parquet (FeatureStore 登録・年別)
  - data/meta/modeling/race_performance_build_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.pipeline.features.feature_layout import BLOCK_RACE_HORSE_TBL, MERGE_KEYS_RACE_HORSE_TBL
from src.pipeline.features.feature_store import FeatureStore
from src.pipeline.features.id_value_policy import sanitize_netkeiba_string_id
from src.scraper.local_tables import available_years as local_available_years, load_all_races_grouped
from src.utils.keiba_logging import script_basic_config

logger = logging.getLogger(__name__)


SECONDS_PER_POINT = {
    "sprint": 0.08,
    "mile": 0.10,
    "intermediate": 0.12,
    "long": 0.14,
    "extended": 0.16,
}

KG_TO_SECONDS = {
    "sprint": 0.14,
    "mile": 0.17,
    "intermediate": 0.20,
    "long": 0.24,
    "extended": 0.26,
}

CLASS_PRIOR = {
    "G": 5.0,
    "OP": 3.5,
    "3勝": 2.5,
    "2勝": 1.5,
    "1勝": 0.5,
    "未勝利": -0.3,
    "新馬": -0.8,
    "障害": 0.0,
    "other": 0.0,
    "unknown": 0.0,
}

OUTPUT_DIR = Path("data/features/race_performance")
RACE_STORE_DIR = OUTPUT_DIR / "races"
SUMMARY_PATH = Path("data/meta/modeling/race_performance_build_summary.json")

FEATURE_COLUMNS = [
    "run_performance_raw",
    "run_performance_final",
    "run_performance_final_std",
    "run_performance_final_pct",
    "time_figure",
    "margin_adjusted_figure",
    "weight_adjusted_figure",
    "pace_adjusted_figure",
    "reference_weight_equivalent",
    "normalized_weight_carried",
    "current_weight_advantage_points",
    "horse_pre_rating_neutral",
    "horse_pre_rating_pre_race",
    "field_relative_pre_rating",
    "field_top3_gap",
    "field_strength_contrib",
    "race_level_pre_race",
    "race_level_retrospective",
    "track_variant",
    "track_variant_online",
    "race_uncertainty",
]


def _default_years(base_dir: str | Path = ".") -> list[str]:
    years = local_available_years(base_dir)
    return years or ["2020", "2021", "2022", "2023", "2024", "2025"]


@dataclass
class RaceMeta:
    race_id: str
    race_date: datetime
    venue_code: str
    venue: str
    surface: str
    distance: int
    grade: str
    race_class: str
    class_group: str
    track_condition: str
    field_size: int
    baseline_key: str
    top3_mean_time: float
    winner_time: float
    second_gap: float
    pace_shape_class: str
    grind_index: float
    burst_index: float
    track_variant: float = 0.0
    track_variant_online: float = 0.0
    baseline_mean: float = 0.0
    baseline_std: float = 1.0
    race_uncertainty: float = 0.0


def _parse_date(raw: str) -> datetime:
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d"):
        try:
            return datetime.strptime((raw or "").strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"unsupported date format: {raw!r}")


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def _distance_band(distance: int) -> str:
    if distance <= 1400:
        return "sprint"
    if distance <= 1800:
        return "mile"
    if distance <= 2200:
        return "intermediate"
    if distance <= 2800:
        return "long"
    return "extended"


def _venue_era(venue_code: str, race_date: datetime) -> str:
    if venue_code == "08":
        return "kyoto_post_20230422" if race_date >= datetime(2023, 4, 22) else "kyoto_pre_20230422"
    return f"{venue_code}_default"


def _class_group(grade: str, race_class: str) -> str:
    s = f"{grade or ''} {race_class or ''}".strip()
    if not s:
        return "unknown"
    if "G1" in s or "GⅠ" in s or "G2" in s or "GⅡ" in s or "G3" in s or "GⅢ" in s:
        return "G"
    if "L" in s or "OP" in s or "オープン" in s or "リステッド" in s:
        return "OP"
    if "3勝" in s or "1600万" in s:
        return "3勝"
    if "2勝" in s or "1000万" in s:
        return "2勝"
    if "1勝" in s or "500万" in s:
        return "1勝"
    if "未勝利" in s:
        return "未勝利"
    if "新馬" in s:
        return "新馬"
    if "障" in s:
        return "障害"
    return "other"


def _parse_lap_times(raw: Any) -> list[float]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        out: list[float] = []
        for v in raw:
            try:
                fv = float(v)
                if math.isfinite(fv) and fv > 0:
                    out.append(fv)
            except (TypeError, ValueError):
                continue
        return out
    if isinstance(raw, str):
        s = raw.replace("－", "-").replace("—", "-")
        nums = []
        for token in s.split("-"):
            try:
                fv = float(token.strip())
                if math.isfinite(fv) and fv > 0:
                    nums.append(fv)
            except (TypeError, ValueError):
                continue
        return nums
    return []


def _parse_pace(raw: Any) -> tuple[float | None, float | None]:
    if isinstance(raw, dict):
        fh = raw.get("first_half_3f")
        sh = raw.get("second_half_3f")
        try:
            fhv = float(fh) if fh is not None else None
            shv = float(sh) if sh is not None else None
            return fhv, shv
        except (TypeError, ValueError):
            return None, None
    return None, None


def _pace_shape(lap_times: list[float], pace: Any) -> tuple[str, float, float]:
    fh, sh = _parse_pace(pace)
    if fh is None or sh is None:
        if len(lap_times) >= 6:
            fh = sum(lap_times[:3])
            sh = sum(lap_times[-3:])
        else:
            return "unknown", 0.5, 0.5

    total = (fh or 0.0) + (sh or 0.0)
    if total <= 0:
        return "unknown", 0.5, 0.5

    delta = (sh - fh) / total
    grind = max(0.0, min(1.0, 0.5 + 2.5 * delta))
    burst = max(0.0, min(1.0, 0.5 - 2.0 * delta))

    if grind >= 0.6:
        shape = "grind"
    elif burst >= 0.6:
        shape = "burst"
    else:
        shape = "balanced"
    return shape, round(grind, 4), round(burst, 4)


def _norm_weight(jockey_weight: float, sex_age: str) -> float:
    sex = (sex_age or "")[:1]
    allowance = 2.0 if sex == "牝" else 0.0
    return jockey_weight + allowance


def _parse_age(sex_age: str) -> int:
    if not sex_age:
        return 0
    digits = "".join(ch for ch in str(sex_age) if ch.isdigit())
    try:
        return int(digits)
    except ValueError:
        return 0


def _race_standard_eq_weight(entries: list[dict[str, Any]]) -> float:
    ages = sorted(a for a in (_parse_age(str(e.get("sex_age", "") or "")) for e in entries) if a > 0)
    if not ages:
        return 58.0
    n = len(ages)
    med_age = ages[n // 2] if n % 2 else (ages[n // 2 - 1] + ages[n // 2]) / 2
    if med_age <= 2:
        return 55.0
    if med_age <= 3:
        return 57.0
    return 58.0


def _weighted_history_mean(values: list[float], limit: int = 5) -> float:
    if not values:
        return 0.0
    recent = values[-limit:]
    weights = [0.70 ** i for i in range(len(recent) - 1, -1, -1)]
    total = sum(weights)
    return sum(v * w for v, w in zip(recent, weights)) / max(total, 1e-9)


def _horse_pre_rating(
    history: list[dict[str, Any]],
    surface: str,
    distance_band: str,
    class_group: str,
) -> float:
    if not history:
        return CLASS_PRIOR.get(class_group, 0.0)

    overall = _weighted_history_mean([float(h["perf"]) for h in history], limit=6)
    same_surface_vals = [float(h["perf"]) for h in history if h.get("surface") == surface]
    same_band_vals = [float(h["perf"]) for h in history if h.get("distance_band") == distance_band]

    same_surface = _weighted_history_mean(same_surface_vals, limit=5) if same_surface_vals else overall
    same_band = _weighted_history_mean(same_band_vals, limit=5) if same_band_vals else overall
    recent_best = max(float(h["perf"]) for h in history[-5:]) if history else overall

    rating = (
        0.40 * overall
        + 0.25 * same_surface
        + 0.25 * same_band
        + 0.10 * recent_best
    )
    return round(rating, 4)


def _style_from_passing(passing_order: str, field_size: int) -> str:
    if not passing_order or field_size <= 1:
        return "neutral"
    try:
        first = int(str(passing_order).split("-")[0])
    except (TypeError, ValueError, IndexError):
        return "neutral"
    norm = (first - 1) / max(field_size - 1, 1)
    if norm <= 0.25:
        return "front"
    if norm >= 0.60:
        return "closer"
    return "neutral"


def _pace_adjustment(
    passing_order: str,
    field_size: int,
    pace_shape_class: str,
    grind_index: float,
    burst_index: float,
) -> float:
    style = _style_from_passing(passing_order, field_size)
    adj = 0.0
    if style == "front":
        adj += max(0.0, grind_index - 0.5) * 2.0
        adj -= max(0.0, burst_index - 0.5) * 1.0
    elif style == "closer":
        adj += max(0.0, burst_index - 0.5) * 2.0
        adj -= max(0.0, grind_index - 0.5) * 1.0
    else:
        if pace_shape_class == "balanced":
            adj += 0.1
    return round(adj, 4)


def _winner_margin_adjustment(second_gap: float, sec_per_point: float) -> float:
    if second_gap <= 0 or sec_per_point <= 0:
        return 0.0
    return round(1.35 * math.log1p(second_gap / sec_per_point), 4)


def _loser_margin_adjustment(gap_to_winner: float, sec_per_point: float) -> float:
    if sec_per_point <= 0:
        return 0.0
    return round(-(gap_to_winner / sec_per_point), 4)


def _build_race_fill_map(base_dir: str | Path = ".") -> dict[str, dict[str, Any]]:
    store = FeatureStore(base_dir=base_dir)
    years = _default_years(base_dir)
    df = store.load_source(
        "race_shutuba",
        years=years,
        columns=["distance", "surface", "field_size", "grade", "race_class", "track_condition", "venue", "venue_code"],
    )
    if df.empty:
        return {}
    race_level = (
        df.sort_values(["race_id", "horse_number"])
        .drop_duplicates(subset=["race_id"])
        .loc[:, ["race_id", "distance", "surface", "field_size", "grade", "race_class", "track_condition", "venue", "venue_code"]]
    )
    return {
        str(r["race_id"]): {
            "distance": r.get("distance"),
            "surface": r.get("surface"),
            "field_size": r.get("field_size"),
            "grade": r.get("grade"),
            "race_class": r.get("race_class"),
            "track_condition": r.get("track_condition"),
            "venue": r.get("venue"),
            "venue_code": r.get("venue_code"),
        }
        for _, r in race_level.iterrows()
    }


def _build_race_meta_map(
    race_result_map: dict[str, dict[str, Any]],
    race_fill_map: dict[str, dict[str, Any]] | None = None,
) -> dict[str, RaceMeta]:
    race_meta: dict[str, RaceMeta] = {}
    for race_id, race in race_result_map.items():
        fill = (race_fill_map or {}).get(race_id, {})
        entries = [e for e in race.get("entries", []) if _safe_float(e.get("time_sec"), -1) > 0 and int(e.get("finish_position") or 0) > 0]
        if len(entries) < 2:
            continue

        sorted_entries = sorted(entries, key=lambda e: int(e.get("finish_position") or 999))
        top3 = [e for e in sorted_entries if int(e.get("finish_position") or 999) <= 3]
        if not top3:
            continue

        try:
            race_date = _parse_date(str(race.get("date", "")))
        except ValueError:
            continue

        distance = int(_safe_float(race.get("distance"), 0) or _safe_float(fill.get("distance"), 0))
        class_group = _class_group(
            str(race.get("grade", "") or fill.get("grade", "")),
            str(race.get("race_class", "") or fill.get("race_class", "")),
        )
        venue_code = str(race.get("venue_code", "") or fill.get("venue_code", "") or "")
        if not venue_code:
            venue_code = str(race_id[4:6])
        venue_era = _venue_era(venue_code, race_date)
        top3_mean_time = sum(_safe_float(e.get("time_sec")) for e in top3) / len(top3)
        winner_time = _safe_float(sorted_entries[0].get("time_sec"))
        second_gap = max(0.0, _safe_float(sorted_entries[1].get("time_sec")) - winner_time) if len(sorted_entries) >= 2 else 0.0
        lap_times = _parse_lap_times(sorted_entries[0].get("lap_times") or race.get("lap_times"))
        pace_shape_class, grind_index, burst_index = _pace_shape(lap_times, sorted_entries[0].get("pace") or race.get("pace"))
        baseline_key = "|".join([
            venue_code,
            str(race.get("surface", "")),
            _distance_band(distance),
            class_group,
            str(race.get("track_condition", "")),
            venue_era,
        ])

        uncertainty = 0.0
        if int(race.get("field_size") or len(entries)) < 8:
            uncertainty += 0.12
        if class_group in {"新馬", "未勝利"}:
            uncertainty += 0.10
        if pace_shape_class == "unknown":
            uncertainty += 0.05
        uncertainty = min(0.45, uncertainty)

        race_meta[race_id] = RaceMeta(
            race_id=race_id,
            race_date=race_date,
            venue_code=venue_code,
            venue=str(race.get("venue", "") or fill.get("venue", "")),
            surface=str(race.get("surface", "") or fill.get("surface", "")),
            distance=distance,
            grade=str(race.get("grade", "")),
            race_class=str(race.get("race_class", "")),
            class_group=class_group,
            track_condition=str(race.get("track_condition", "") or fill.get("track_condition", "")),
            field_size=int(race.get("field_size") or fill.get("field_size") or len(entries)),
            baseline_key=baseline_key,
            top3_mean_time=round(top3_mean_time, 4),
            winner_time=winner_time,
            second_gap=round(second_gap, 4),
            pace_shape_class=pace_shape_class,
            grind_index=grind_index,
            burst_index=burst_index,
            race_uncertainty=round(uncertainty, 4),
        )
    return race_meta


def _stats_from_values(values: list[float], default_mean: float) -> dict[str, float]:
    if not values:
        return {"mean": round(default_mean, 4), "std": 1.0, "count": 0}
    mean = sum(values) / len(values)
    if len(values) < 2:
        std = 1.0
    else:
        var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        std = max(math.sqrt(var), 0.35)
    return {"mean": round(mean, 4), "std": round(std, 4), "count": len(values)}


def _assign_online_baselines_and_track_variants(race_meta_map: dict[str, RaceMeta]) -> None:
    """future を見ない online baseline / track variant を RaceMeta に付与する。

    baseline は前日以前の履歴のみで算出し、track_variant は当日同場同surfaceの
    レース群から日単位で集計する。これにより「その日の終わりに算出」という運用条件に合う。
    """
    exact_hist: dict[str, list[float]] = defaultdict(list)
    fb1_hist: dict[str, list[float]] = defaultdict(list)
    fb2_hist: dict[str, list[float]] = defaultdict(list)

    by_date: dict[datetime.date, list[RaceMeta]] = defaultdict(list)
    for meta in race_meta_map.values():
        by_date[meta.race_date.date()].append(meta)

    for race_date in sorted(by_date.keys()):
        metas = sorted(by_date[race_date], key=lambda m: m.race_id)
        day_residuals: dict[tuple[str, str], list[float]] = defaultdict(list)

        for meta in metas:
            key1 = "|".join(["ALL", meta.surface, _distance_band(meta.distance), meta.class_group, meta.track_condition])
            key2 = "|".join(["ALL", meta.surface, _distance_band(meta.distance), meta.class_group, "ALL"])

            if len(exact_hist[meta.baseline_key]) >= 5:
                bl = _stats_from_values(exact_hist[meta.baseline_key], meta.top3_mean_time)
            elif len(fb1_hist[key1]) >= 5:
                bl = _stats_from_values(fb1_hist[key1], meta.top3_mean_time)
            elif len(fb2_hist[key2]) >= 5:
                bl = _stats_from_values(fb2_hist[key2], meta.top3_mean_time)
            else:
                bl = {"mean": round(meta.top3_mean_time, 4), "std": 1.0, "count": 0}

            meta.baseline_mean = bl["mean"]
            meta.baseline_std = bl["std"]
            residual = meta.top3_mean_time - bl["mean"]
            day_residuals[(meta.venue_code, meta.surface)].append(residual)

        day_variant_map: dict[tuple[str, str], float] = {}
        for key, vals in day_residuals.items():
            vals_sorted = sorted(vals)
            if len(vals_sorted) >= 3:
                trimmed = vals_sorted[1:-1]
                vals_use = trimmed if trimmed else vals_sorted
            else:
                vals_use = vals_sorted
            day_variant_map[key] = round(sum(vals_use) / len(vals_use), 4)

        for meta in metas:
            meta.track_variant = day_variant_map.get((meta.venue_code, meta.surface), 0.0)

        for meta in metas:
            exact_hist[meta.baseline_key].append(meta.top3_mean_time)
            key1 = "|".join(["ALL", meta.surface, _distance_band(meta.distance), meta.class_group, meta.track_condition])
            key2 = "|".join(["ALL", meta.surface, _distance_band(meta.distance), meta.class_group, "ALL"])
            fb1_hist[key1].append(meta.top3_mean_time)
            fb2_hist[key2].append(meta.top3_mean_time)


def _future_emergence_scores(df: pd.DataFrame, horizon_days: int = 365) -> dict[tuple[str, int], float]:
    by_horse: dict[str, pd.DataFrame] = {
        hid: g.sort_values("race_date_dt").reset_index(drop=True)
        for hid, g in df.groupby("horse_id")
    }
    out: dict[tuple[str, int], float] = {}
    horizon = timedelta(days=horizon_days)

    for hid, g in by_horse.items():
        dates = g["race_date_dt"].tolist()
        perfs = g["run_performance_final"].tolist()
        keys = list(zip(g["race_id"].tolist(), g["horse_number"].tolist()))
        for i, (race_key, perf_now) in enumerate(zip(keys, perfs)):
            cur_date = dates[i]
            future_vals = [
                perfs[j]
                for j in range(i + 1, len(g))
                if dates[j] - cur_date <= horizon
            ]
            if future_vals:
                out[race_key] = round(0.6 * max(future_vals) + 0.4 * (sum(future_vals) / len(future_vals)), 4)
            else:
                out[race_key] = round(perf_now, 4)
    return out


def build_race_performance_dataframe(
    years: list[str] | None = None,
    *,
    base_dir: str | Path = ".",
    future_horizon_days: int = 0,
    target_race_ids: set[str] | None = None,
) -> pd.DataFrame:
    years = years or _default_years(base_dir)
    race_result_map = load_all_races_grouped("race_result", years=years, base_dir=base_dir)
    if not race_result_map:
        return pd.DataFrame()

    race_fill_map = _build_race_fill_map(base_dir=base_dir)
    race_meta_map = _build_race_meta_map(race_result_map, race_fill_map)
    _assign_online_baselines_and_track_variants(race_meta_map)

    horse_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    track_history: dict[tuple[str, str], list[float]] = defaultdict(list)
    rows: list[dict[str, Any]] = []

    ordered_races = sorted(race_meta_map.values(), key=lambda r: (r.race_date, r.race_id))
    for meta in ordered_races:
        race = race_result_map.get(meta.race_id, {})
        entries = [e for e in race.get("entries", []) if int(e.get("finish_position") or 0) > 0 and _safe_float(e.get("time_sec"), -1) > 0]
        if not entries:
            continue

        track_variant = meta.track_variant
        track_key = (meta.venue_code, meta.surface)
        prior_track_vals = track_history.get(track_key, [])
        if prior_track_vals:
            recent_track_vals = prior_track_vals[-20:]
            meta.track_variant_online = round(sum(recent_track_vals) / len(recent_track_vals), 4)
        else:
            meta.track_variant_online = 0.0

        entrant_pre_ratings: list[float] = []
        entrant_rating_map: dict[tuple[str, int], float] = {}
        for e in entries:
            hid = str(e.get("horse_id", "") or "")
            rating = _horse_pre_rating(
                horse_history.get(hid, []),
                meta.surface,
                band := _distance_band(meta.distance),
                meta.class_group,
            )
            entrant_pre_ratings.append(rating)
            entrant_rating_map[(hid, int(e.get("horse_number") or 0))] = rating

        sorted_prior = sorted(entrant_pre_ratings, reverse=True)
        top3_prior = sorted_prior[:3] if sorted_prior else [0.0]
        race_level_pre = 0.6 * (sum(top3_prior) / len(top3_prior)) + 0.4 * (sum(entrant_pre_ratings) / len(entrant_pre_ratings))

        sec_per_point = SECONDS_PER_POINT[band]
        kg_to_sec = KG_TO_SECONDS[band]
        race_reference_eq_weight = _race_standard_eq_weight(entries)
        field_avg_pre = sum(entrant_pre_ratings) / max(len(entrant_pre_ratings), 1)
        field_top3_avg_pre = sum(top3_prior) / len(top3_prior)

        winner_time = meta.winner_time
        second_gap = meta.second_gap

        race_rows: list[dict[str, Any]] = []
        for e in sorted(entries, key=lambda x: int(x.get("finish_position") or 999)):
            fp = int(e.get("finish_position") or 0)
            horse_number = int(e.get("horse_number") or 0)
            horse_id = str(e.get("horse_id", "") or "")
            time_sec = _safe_float(e.get("time_sec"))
            sex_age = str(e.get("sex_age", "") or "")
            jockey_weight = _safe_float(e.get("jockey_weight"), 0.0)

            norm_w = _norm_weight(jockey_weight, sex_age)
            weight_adjustment_sec = (norm_w - race_reference_eq_weight) * kg_to_sec

            adjusted_time = time_sec - track_variant - weight_adjustment_sec
            time_figure = (meta.baseline_mean - adjusted_time) / sec_per_point

            gap_to_winner = max(0.0, time_sec - winner_time)
            if fp == 1:
                margin_adjustment = _winner_margin_adjustment(second_gap, sec_per_point)
            else:
                margin_adjustment = _loser_margin_adjustment(gap_to_winner, sec_per_point)

            pace_adjustment = _pace_adjustment(
                str(e.get("passing_order", "") or ""),
                meta.field_size,
                meta.pace_shape_class,
                meta.grind_index,
                meta.burst_index,
            )

            horse_pre_rating_neutral = entrant_rating_map.get((horse_id, horse_number), CLASS_PRIOR.get(meta.class_group, 0.0))
            current_weight_advantage_points = ((race_reference_eq_weight - norm_w) * kg_to_sec) / sec_per_point
            horse_pre_rating = horse_pre_rating_neutral + current_weight_advantage_points
            field_relative_pre = horse_pre_rating - field_avg_pre
            field_top3_gap = horse_pre_rating - field_top3_avg_pre
            field_strength_contrib = field_relative_pre
            run_raw = (
                time_figure
                + 0.25 * margin_adjustment
                + 0.15 * pace_adjustment
                + 0.20 * field_relative_pre
                + 0.10 * field_top3_gap
            )
            run_final = run_raw * (1.0 - meta.race_uncertainty)

            race_rows.append({
                "race_id": meta.race_id,
                "race_date": meta.race_date.strftime("%Y-%m-%d"),
                "race_date_dt": meta.race_date,
                "horse_number": horse_number,
                "horse_id": horse_id,
                "horse_name": str(e.get("horse_name", "") or ""),
                "finish_position": fp,
                "venue_code": meta.venue_code,
                "venue": meta.venue,
                "surface": meta.surface,
                "distance": meta.distance,
                "distance_band": band,
                "class_group": meta.class_group,
                "venue_era": _venue_era(meta.venue_code, meta.race_date),
                "track_condition": meta.track_condition,
                "field_size": meta.field_size,
                "pace_shape_class": meta.pace_shape_class,
                "track_variant": round(track_variant, 4),
                "track_variant_online": round(meta.track_variant_online, 4),
                "baseline_mean_time": round(meta.baseline_mean, 4),
                "adjusted_finish_time": round(adjusted_time, 4),
                "time_figure": round(time_figure, 4),
                "margin_adjusted_figure": round(margin_adjustment, 4),
                "weight_adjusted_figure": round(weight_adjustment_sec / sec_per_point, 4),
                "pace_adjusted_figure": round(pace_adjustment, 4),
                "reference_weight_equivalent": round(race_reference_eq_weight, 4),
                "normalized_weight_carried": round(norm_w, 4),
                "current_weight_advantage_points": round(current_weight_advantage_points, 4),
                "horse_pre_rating_neutral": round(horse_pre_rating_neutral, 4),
                "horse_pre_rating_pre_race": round(horse_pre_rating, 4),
                "field_relative_pre_rating": round(field_relative_pre, 4),
                "field_top3_gap": round(field_top3_gap, 4),
                "field_strength_contrib": round(field_strength_contrib, 4),
                "race_level_pre_race": round(race_level_pre, 4),
                "race_uncertainty": round(meta.race_uncertainty, 4),
                "run_performance_raw": round(run_raw, 4),
                "run_performance_final": round(run_final, 4),
            })

        for rr in race_rows:
            if rr["horse_id"]:
                horse_history[rr["horse_id"]].append({
                    "perf": rr["run_performance_final"],
                    "surface": rr["surface"],
                    "distance_band": rr["distance_band"],
                    "class_group": rr["class_group"],
                    "race_date": rr["race_date"],
                })
        track_history[track_key].append(track_variant)
        if target_race_ids is None or meta.race_id in target_race_ids:
            rows.extend(race_rows)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if future_horizon_days and future_horizon_days > 0:
        future_scores = _future_emergence_scores(df, horizon_days=future_horizon_days)
        df["future_emergence_score"] = df.apply(
            lambda r: future_scores.get((r["race_id"], int(r["horse_number"])), r["run_performance_final"]),
            axis=1,
        )

        retro_level_map: dict[str, float] = {}
        for race_id, g in df.groupby("race_id"):
            vals = list(g["future_emergence_score"].astype(float))
            vals_sorted = sorted(vals, reverse=True)
            top3 = vals_sorted[:3] if vals_sorted else [0.0]
            retro_level = 0.6 * (sum(top3) / len(top3)) + 0.4 * (sum(vals) / len(vals))
            retro_level_map[race_id] = round(retro_level, 4)
        df["race_level_retrospective"] = df["race_id"].map(retro_level_map).astype(float)
    else:
        df["future_emergence_score"] = df["run_performance_final"]
        df["race_level_retrospective"] = df["race_level_pre_race"]

    seg_cols = ["venue_era", "surface", "distance_band", "class_group"]
    seg_stats = (
        df.groupby(seg_cols)["run_performance_final"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "_seg_mean", "std": "_seg_std", "count": "_seg_count"})
    )
    df = df.merge(seg_stats, on=seg_cols, how="left")
    global_mean = float(df["run_performance_final"].mean())
    global_std = float(df["run_performance_final"].std() or 1.0)
    seg_mean = df["_seg_mean"].fillna(global_mean)
    seg_std = df["_seg_std"].fillna(global_std).replace(0, global_std)
    df["run_performance_final_std"] = ((df["run_performance_final"] - seg_mean) / seg_std).round(4)
    df["run_performance_final_pct"] = (
        df.groupby(seg_cols)["run_performance_final"]
        .rank(method="average", pct=True)
        .fillna(0.5)
        .round(4)
    )
    df = df.drop(columns=["_seg_mean", "_seg_std", "_seg_count"])

    return df.sort_values(["race_date_dt", "race_id", "horse_number"]).reset_index(drop=True)


def statistics_median(values: list[float]) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return 0.0
    n = len(vals)
    mid = n // 2
    if n % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2


def save_race_performance_features(
    df: pd.DataFrame,
    years: list[str],
    *,
    base_dir: str | Path = ".",
    overwrite: bool = True,
    write_race_store: bool = True,
    future_horizon_days: int = 0,
) -> dict[str, Any]:
    base = Path(base_dir)
    out_dir = base / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if write_race_store:
        _save_race_store(df, base_dir=base)

    year_files: dict[str, str] = {}
    for year in years:
        ydf = df[df["race_id"].astype(str).str[:4] == str(year)].copy()
        if ydf.empty:
            continue
        path = out_dir / f"{year}.parquet"
        ydf.to_parquet(path, index=False)
        year_files[str(year)] = str(path)

    store = FeatureStore(base_dir=base)
    snap_name = f"race_performance_{years[0]}_{years[-1]}"
    store.save_snapshot(snap_name, df.drop(columns=["race_date_dt"]))

    sub_all = df[["race_id", "horse_id", *FEATURE_COLUMNS]].copy()
    sub_all["horse_id"] = sanitize_netkeiba_string_id(sub_all["horse_id"])
    sub_all = sub_all.loc[sub_all["horse_id"].notna()].copy()
    for col in FEATURE_COLUMNS:
        sub = sub_all[["race_id", "horse_id", col]].copy()
        store.save_feature_column(
            col,
            sub,
            table_block=BLOCK_RACE_HORSE_TBL,
            merge_keys=list(MERGE_KEYS_RACE_HORSE_TBL),
            overwrite=overwrite,
            registry_extra={"source": "race_performance", "source_column": col},
        )

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "years": years,
        "rows": int(len(df)),
        "races": int(df["race_id"].nunique()),
        "horses": int(df["horse_id"].nunique()),
        "year_files": year_files,
        "feature_columns": FEATURE_COLUMNS,
        "snapshot_name": snap_name,
        "future_horizon_days": future_horizon_days,
        "as_of_date": datetime.now(timezone.utc).date().isoformat(),
        "coefficients": {
            "seconds_per_point": SECONDS_PER_POINT,
            "kg_to_seconds": KG_TO_SECONDS,
            "class_prior": CLASS_PRIOR,
        },
        "notes": {
            "race_level_pre_race": "safe for future feature generation",
            "race_level_retrospective": (
                "same as pre_race when future_horizon_days=0; "
                "future-aware only when horizon > 0"
            ),
            "race_store": "per-race JSON files are the source of truth for incremental accumulation",
        },
        "feature_policies": {
            "online_safe": [
                "run_performance_raw",
                "run_performance_final",
                "run_performance_final_std",
                "run_performance_final_pct",
                "time_figure",
                "margin_adjusted_figure",
                "weight_adjusted_figure",
                "pace_adjusted_figure",
                "reference_weight_equivalent",
                "normalized_weight_carried",
                "current_weight_advantage_points",
                "horse_pre_rating_neutral",
                "horse_pre_rating_pre_race",
                "field_relative_pre_rating",
                "field_top3_gap",
                "field_strength_contrib",
                "race_level_pre_race",
                "track_variant",
                "track_variant_online",
                "race_uncertainty"
            ],
            "retro_only": [
                "race_level_retrospective"
            ]
        }
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _save_race_store(df: pd.DataFrame, *, base_dir: str | Path = ".") -> None:
    base = Path(base_dir)
    root = base / RACE_STORE_DIR
    root.mkdir(parents=True, exist_ok=True)
    meta_cols = [
        "race_id",
        "race_date",
        "venue_code",
        "venue",
        "surface",
        "distance",
        "distance_band",
        "class_group",
        "venue_era",
        "track_condition",
        "field_size",
        "pace_shape_class",
        "track_variant",
        "race_level_pre_race",
        "race_level_retrospective",
        "race_uncertainty",
    ]
    for race_id, g in df.groupby("race_id", sort=False):
        year = str(race_id)[:4]
        path = root / year / f"{race_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        first = g.iloc[0]
        payload = {
            "race_id": race_id,
            "year": year,
            "as_of_date": _json_safe(first.get("race_date")),
            "meta": {k: _json_safe(first[k]) for k in meta_cols if k in g.columns},
            "entries": [
                {k: _json_safe(v) for k, v in row.items() if k != "race_date_dt"}
                for row in g.to_dict("records")
            ],
            "feature_policies": {
                "online_safe": [
                    "run_performance_raw",
                    "run_performance_final",
                    "run_performance_final_std",
                    "run_performance_final_pct",
                    "time_figure",
                    "margin_adjusted_figure",
                    "weight_adjusted_figure",
                    "pace_adjusted_figure",
                    "reference_weight_equivalent",
                    "normalized_weight_carried",
                    "current_weight_advantage_points",
                    "horse_pre_rating_neutral",
                    "horse_pre_rating_pre_race",
                    "field_relative_pre_rating",
                    "field_top3_gap",
                    "field_strength_contrib",
                    "race_level_pre_race",
                    "track_variant",
                    "track_variant_online",
                    "race_uncertainty"
                ],
                "retro_only": [
                    "future_emergence_score",
                    "race_level_retrospective"
                ]
            }
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_race_store_seeded(
    years: list[str],
    *,
    base_dir: str | Path = ".",
) -> None:
    base = Path(base_dir)
    root = base / RACE_STORE_DIR
    year_table_root = base / OUTPUT_DIR
    need_seed = []
    for year in years:
        ydir = root / str(year)
        if not ydir.exists() or not any(ydir.glob("*.json")):
            need_seed.append(str(year))
    if not need_seed:
        return
    for year in need_seed:
        parquet_path = year_table_root / f"{year}.parquet"
        if not parquet_path.exists():
            continue
        df = pd.read_parquet(parquet_path)
        if "race_date_dt" not in df.columns and "race_date" in df.columns:
            df["race_date_dt"] = pd.to_datetime(df["race_date"])
        _save_race_store(df, base_dir=base)


def _json_safe(v: Any) -> Any:
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    try:
        import numpy as np  # local import to avoid hard dependency surface
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return None if np.isnan(v) else float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
    except Exception:
        pass
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    return v


def load_race_store_dataframe(
    years: list[str] | None = None,
    *,
    base_dir: str | Path = ".",
) -> pd.DataFrame:
    base = Path(base_dir)
    root = base / RACE_STORE_DIR
    years = years or _default_years(base)
    rows: list[dict[str, Any]] = []
    for year in years:
        ydir = root / str(year)
        if not ydir.exists():
            continue
        for path in sorted(ydir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            meta = payload.get("meta", {})
            for entry in payload.get("entries", []):
                row = dict(entry)
                for k, v in meta.items():
                    row.setdefault(k, v)
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "race_date_dt" in df.columns:
        df = df.drop(columns=["race_date_dt"])
    return df


def rebuild_year_tables_from_race_store(
    years: list[str] | None = None,
    *,
    base_dir: str | Path = ".",
    overwrite: bool = True,
) -> dict[str, Any]:
    years = years or _default_years(base_dir)
    df = load_race_store_dataframe(years=years, base_dir=base_dir)
    if df.empty:
        raise ValueError("race store is empty")
    if "race_date_dt" not in df.columns and "race_date" in df.columns:
        df["race_date_dt"] = pd.to_datetime(df["race_date"])
    return save_race_performance_features(
        df,
        years=years,
        base_dir=base_dir,
        overwrite=overwrite,
        write_race_store=False,
        future_horizon_days=0,
    )


def build_race_performance_for_race_ids(
    race_ids: list[str],
    *,
    base_dir: str | Path = ".",
    future_horizon_days: int = 0,
    overwrite: bool = True,
) -> dict[str, Any]:
    if not race_ids:
        raise ValueError("race_ids required")
    race_ids = sorted(set(str(r) for r in race_ids))
    years = sorted(set(r[:4] for r in race_ids))
    _ensure_race_store_seeded(years, base_dir=base_dir)
    all_years = _default_years(base_dir)
    df = build_race_performance_dataframe(
        years=all_years,
        base_dir=base_dir,
        future_horizon_days=future_horizon_days,
        target_race_ids=set(race_ids),
    )
    if df.empty:
        raise ValueError("no rows built for target races")
    _save_race_store(df, base_dir=base_dir)
    return rebuild_year_tables_from_race_store(years=years, base_dir=base_dir, overwrite=overwrite)


def main() -> int:
    script_basic_config()
    ap = argparse.ArgumentParser(description="レースパフォーマンス特徴量生成")
    ap.add_argument("--race-ids", default="", help="対象 race_id (comma separated). 指定時はレース単位算出")
    ap.add_argument("--years", default="auto", help="対象年 (comma separated or 'auto')")
    ap.add_argument("--future-horizon-days", type=int, default=0, help="retrospective race level horizon (0=disabled)")
    args = ap.parse_args()

    race_ids = [r.strip() for r in args.race_ids.split(",") if r.strip()]
    if race_ids:
        logger.info("レース単位算出開始: %s", race_ids)
        summary = build_race_performance_for_race_ids(
            race_ids,
            base_dir=".",
            future_horizon_days=args.future_horizon_days,
        )
    else:
        if args.years.strip().lower() == "auto":
            years = _default_years(".")
        else:
            years = [y.strip() for y in args.years.split(",") if y.strip()]
        logger.info("レースパフォーマンス算出開始: %s", years)
        df = build_race_performance_dataframe(years, future_horizon_days=args.future_horizon_days)
        if df.empty:
            logger.error("算出結果が空です")
            return 1
        summary = save_race_performance_features(df, years, future_horizon_days=args.future_horizon_days)
    logger.info("完了: rows=%d races=%d snapshot=%s", summary["rows"], summary["races"], summary["snapshot_name"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
