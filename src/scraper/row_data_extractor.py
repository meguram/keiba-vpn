"""
要件表の各行（row_id）に対応する「行固有 JSON」を正本 canonical JSON から抽出する。

新カテゴリ（派生）:
  race_shutuba_meta       ← race_shutuba         （entries 以外のメタフィールド）
  race_result_on_time_payoff ← race_result_on_time  （payoff フィールド）
  race_result_on_time_lap    ← race_result_on_time  （lap_times + pace）
  race_result_on_time_corner ← race_result_on_time  （corner_passing）
  horse_profile           ← horse_result         （race_history 以外のプロフィール）
  horse_race_history      ← horse_result         （race_history + horse_id/name）
  race_result_meta        ← race_result          （entries/payoff/lap_times/corner_passing 以外）
  race_result_payoff      ← race_result          （payoff フィールド）
  race_result_track       ← race_result          （track_condition/weather 等）
  race_result_corner      ← race_result_lap      （corner_passing フィールド）
  race_result_lap_times   ← race_result_lap      （lap_times + pace フィールド）
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------- #
# フィールドセット定義
# ---------------------------------------------------------------------------- #

_SHUTUBA_META_FIELDS: frozenset[str] = frozenset({
    "race_id", "race_name", "date", "venue", "round", "surface", "distance",
    "direction", "weather", "track_condition", "start_time", "field_size",
    "grade", "race_class", "weight_rule", "course_type",
})

_HORSE_PROFILE_FIELDS: frozenset[str] = frozenset({
    "horse_id", "horse_name", "name_en", "status", "sex", "age", "color",
    "birthday", "trainer", "trainer_id", "owner", "breeder", "birthplace",
    "total_earnings", "career", "career_record", "major_wins", "sire", "dam", "dam_sire",
})

_RESULT_META_FIELDS: frozenset[str] = frozenset({
    "race_id", "race_name", "date", "venue", "round", "surface", "distance",
    "direction", "grade", "field_size", "start_time",
})

_RESULT_TRACK_FIELDS: frozenset[str] = frozenset({
    "race_id", "weather", "track_condition",
    "track_condition_turf", "track_condition_dirt",
})


# ---------------------------------------------------------------------------- #
# 抽出関数
# ---------------------------------------------------------------------------- #

def extract_shutuba_meta(data: dict[str, Any]) -> dict[str, Any]:
    """race_shutuba → race_shutuba_meta: エントリ一覧を除いたレースメタ情報。"""
    return {k: v for k, v in data.items() if k in _SHUTUBA_META_FIELDS}


def extract_result_on_time_payoff(data: dict[str, Any]) -> dict[str, Any]:
    """race_result_on_time → race_result_on_time_payoff: 払戻情報。"""
    return {
        "race_id": data.get("race_id", ""),
        "payoff": data.get("payoff"),
    }


def extract_result_on_time_lap(data: dict[str, Any]) -> dict[str, Any]:
    """race_result_on_time → race_result_on_time_lap: ラップタイム + ペース。"""
    return {
        "race_id": data.get("race_id", ""),
        "lap_times": data.get("lap_times"),
        "pace": data.get("pace"),
    }


def extract_result_on_time_corner(data: dict[str, Any]) -> dict[str, Any]:
    """race_result_on_time → race_result_on_time_corner: コーナー通過順位（速報）。"""
    return {
        "race_id": data.get("race_id", ""),
        "corner_passing": data.get("corner_passing"),
    }


def extract_horse_profile(data: dict[str, Any]) -> dict[str, Any]:
    """horse_result → horse_profile: プロフィール系フィールドのみ（race_history 除く）。"""
    return {k: v for k, v in data.items() if k in _HORSE_PROFILE_FIELDS}


def extract_horse_race_history(data: dict[str, Any]) -> dict[str, Any]:
    """horse_result → horse_race_history: 過去成績リスト。"""
    return {
        "horse_id": data.get("horse_id", ""),
        "horse_name": data.get("horse_name", ""),
        "race_history": data.get("race_history", []),
    }


def extract_result_meta(data: dict[str, Any]) -> dict[str, Any]:
    """race_result → race_result_meta: エントリ・払戻・ラップ・コーナーを除いたレースメタ。"""
    return {k: v for k, v in data.items() if k in _RESULT_META_FIELDS}


def extract_result_payoff(data: dict[str, Any]) -> dict[str, Any]:
    """race_result → race_result_payoff: 払戻情報（確定）。"""
    return {
        "race_id": data.get("race_id", ""),
        "payoff": data.get("payoff"),
    }


def extract_result_track(data: dict[str, Any]) -> dict[str, Any]:
    """race_result → race_result_track: 馬場情報（track_condition・weather 等）。"""
    result: dict[str, Any] = {k: v for k, v in data.items() if k in _RESULT_TRACK_FIELDS}
    result.setdefault("race_id", data.get("race_id", ""))
    return result


def extract_result_corner(data: dict[str, Any]) -> dict[str, Any]:
    """race_result_lap → race_result_corner: コーナー通過順位（確定）。"""
    return {
        "race_id": data.get("race_id", ""),
        "corner_passing": data.get("corner_passing"),
    }


def extract_result_lap_times(data: dict[str, Any]) -> dict[str, Any]:
    """race_result_lap → race_result_lap_times: ラップタイム + ペース（確定）。"""
    return {
        "race_id": data.get("race_id", ""),
        "lap_times": data.get("lap_times"),
        "pace": data.get("pace"),
    }


# ---------------------------------------------------------------------------- #
# カテゴリ → (ソースカテゴリ, 抽出関数) の対応表
# ---------------------------------------------------------------------------- #

DERIVED_CATEGORY_MAP: dict[str, tuple[str, Any]] = {
    "race_shutuba_meta":           ("race_shutuba",         extract_shutuba_meta),
    "race_result_on_time_payoff":  ("race_result_on_time",  extract_result_on_time_payoff),
    "race_result_on_time_lap":     ("race_result_on_time",  extract_result_on_time_lap),
    "race_result_on_time_corner":  ("race_result_on_time",  extract_result_on_time_corner),
    "horse_profile":               ("horse_result",          extract_horse_profile),
    "horse_race_history":          ("horse_result",          extract_horse_race_history),
    "race_result_meta":            ("race_result",           extract_result_meta),
    "race_result_payoff":          ("race_result",           extract_result_payoff),
    "race_result_track":           ("race_result",           extract_result_track),
    "race_result_corner":          ("race_result_lap",       extract_result_corner),
    "race_result_lap_times":       ("race_result_lap",       extract_result_lap_times),
}
