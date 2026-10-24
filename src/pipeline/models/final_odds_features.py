"""最終オッズ予測用の特徴量拡張（フィールド相対・カテゴリ符号化）。"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from src.pipeline.features.feature_builder import get_feature_columns

# フィールド内比較の対象（数値能力系）
_FIELD_COMPARE_COLS = [
    "speed_max",
    "speed_avg",
    "speed_best_dist",
    "speed_best_surface",
    "avg_finish_5",
    "min_finish_5",
    "top3_count_5",
    "win_count_5",
    "avg_last_3f_5",
    "min_last_3f_5",
    "career_runs",
    "career_win_rate",
    "career_top3_rate",
    "same_surface_win_rate",
    "same_dist_win_rate",
    "training_impression_score",
    "barometer_total",
    "barometer_start",
    "barometer_chase",
    "barometer_closing",
    "paddock_score",
    "days_since_last",
    "weight",
    "jockey_weight",
    "avg_prev_pop_rank_diff",
    "upset_count",
]

_GRADE_PATTERNS = [
    ("G1", r"G[IＩ]"),
    ("G2", r"G[IIＩＩ]"),
    ("G3", r"G[IIIＩＩＩ]"),
    ("OP", r"OP|オープン"),
    ("L", r"L|リステッド"),
    ("3勝", r"3勝"),
    ("2勝", r"2勝"),
    ("1勝", r"1勝"),
    ("新馬", r"新馬"),
    ("未勝利", r"未勝利"),
    ("障害", r"障害"),
]


def _encode_surface(s: str) -> int:
    if "芝" in s:
        return 1
    if "ダ" in s:
        return 2
    return 0


def _encode_direction(s: str) -> int:
    if "右" in s:
        return 1
    if "左" in s:
        return 2
    return 0


def _encode_track_condition(s: str) -> int:
    if "不良" in s:
        return 4
    if "重" in s:
        return 3
    if "稍" in s:
        return 2
    if "良" in s:
        return 1
    return 0


def _encode_grade(race_name: str) -> int:
    name = race_name or ""
    for i, (_, pat) in enumerate(_GRADE_PATTERNS, start=1):
        if re.search(pat, name, re.I):
            return i
    return 0


def _encode_venue(venue: str) -> int:
    venues = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]
    for i, v in enumerate(venues, start=1):
        if v in (venue or ""):
            return i
    return 0


def enrich_final_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    build_race_features 出力にレース内相対統計・符号化カテゴリを付与する。
    """
    if df.empty:
        return df

    out = df.copy()
    n = len(out)
    fs = float(out["field_size"].iloc[0]) if "field_size" in out.columns else float(n)
    fs = max(fs, float(n), 1.0)

    if "horse_number" in out.columns:
        out["horse_number_norm"] = out["horse_number"].astype(float) / fs
    if "bracket_number" in out.columns:
        out["bracket_gate_ratio"] = out["bracket_number"].astype(float) / 8.0

    race_name = str(out.get("race_name", pd.Series([""])).iloc[0] if "race_name" in out.columns else "")
    out["surface_code"] = _encode_surface(str(out.get("surface", pd.Series([""])).iloc[0]))
    out["direction_code"] = _encode_direction(str(out.get("direction", pd.Series([""])).iloc[0]))
    out["track_condition_code"] = _encode_track_condition(
        str(out.get("track_condition", pd.Series([""])).iloc[0])
    )
    out["venue_code"] = _encode_venue(str(out.get("venue", pd.Series([""])).iloc[0]))
    out["grade_code"] = _encode_grade(race_name)
    out["distance_km"] = out["distance"].astype(float) / 1000.0 if "distance" in out.columns else 0.0
    out["is_turf"] = (out["surface_code"] == 1).astype(int)
    out["is_dirt"] = (out["surface_code"] == 2).astype(int)
    out["log_field_size"] = np.log1p(fs)

    for col in _FIELD_COMPARE_COLS:
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        mu = float(s.mean())
        sd = float(s.std()) + 1e-6
        out[f"fld_z_{col}"] = (s - mu) / sd
        out[f"fld_rank_{col}"] = s.rank(ascending=False, method="average") / n
        out[f"fld_gap_best_{col}"] = float(s.max()) - s

    # 頭数あたりの混戦度
    if "speed_max" in out.columns:
        sm = pd.to_numeric(out["speed_max"], errors="coerce").fillna(0)
        out["field_speed_std"] = float(sm.std())
        out["field_speed_spread"] = float(sm.max() - sm.min())

    return out


def select_model_feature_columns(df: pd.DataFrame) -> list[str]:
    """学習・推論に使う列（ID・ラベル・文字列を除く数値列）。"""
    exclude = {
        "race_id",
        "horse_id",
        "horse_name",
        "jockey_name",
        "trainer_name",
        "venue",
        "surface",
        "direction",
        "weather",
        "track_condition",
        "race_date",
        "sex",
        "race_name",
        "split",
        "data_year",
        "target_log_win_odds",
        "target_log_place_min",
        "target_log_place_max",
        "target_win_odds",
        "target_place_min",
        "target_place_max",
    }
    # get_feature_columns で大衆指標は既に除外済み
    _ = get_feature_columns(df)
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if df[c].dtype == object:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cols.append(c)
    return sorted(cols)
