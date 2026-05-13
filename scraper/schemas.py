"""
全スクレイピングカテゴリの期待スキーマ定義 + バリデーション。

保存前に validate() を呼び、結果を _meta.schema_validation に記録する。
保存は常に行い（lenient モード）、診断だけを付加する。

スキーマ dict の構造:
  top_required  : トップレベルで必ず存在すべきフィールド
  top_optional  : 存在してもしなくてもよいフィールド
  entry_required: エントリ（行）ごとに必ず存在すべきフィールド
  entry_optional: エントリで任意のフィールド
  entry_list_key: entries / race_history など（default "entries"）
  lists         : race_pair_odds のように複数リストキーを持つ場合の定義

各フィールド記述子:
  type      : "str" | "int" | "float" | "list" | "dict" | "any" | "bool"
  non_empty : True → 空文字列 / None を不可とする
  min / max : int/float の範囲
  min_length: list の最小長
  pattern   : str の正規表現パターン
"""

from __future__ import annotations

import re
from typing import Any

SCHEMA_VERSION = 1

_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "str": str,
    "int": (int,),
    "float": (int, float),
    "list": list,
    "dict": dict,
    "bool": bool,
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# レース系スキーマ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_RACE_RESULT: dict[str, Any] = {
    "top_required": {
        "race_id": {"type": "str"},
        "race_name": {"type": "str"},
        "date": {"type": "str", "pattern": r"\d{4}-\d{2}-\d{2}"},
        "venue": {"type": "str"},
        "entries": {"type": "list", "min_length": 1},
    },
    "top_optional": {
        "round": {"type": "int"},
        "surface": {"type": "str"},
        "distance": {"type": "int"},
        "field_size": {"type": "int"},
        "grade": {"type": "str"},
        "direction": {"type": "str"},
        "weather": {"type": "str"},
        "track_condition": {"type": "str"},
        "start_time": {"type": "str"},
        "payoff": {"type": "dict"},
        "lap_times": {"type": "list"},
        "pace": {"type": "dict"},
        "corner_passing": {"type": "list"},
    },
    "entry_required": {
        "horse_number": {"type": "int", "min": 1},
        "horse_name": {"type": "str", "non_empty": True},
        "horse_id": {"type": "str", "non_empty": True},
        "bracket_number": {"type": "int"},
    },
    "entry_optional": {
        "finish_position": {"type": "any"},
        "sex_age": {"type": "str"},
        "jockey_weight": {"type": "any"},
        "jockey_name": {"type": "str"},
        "jockey_id": {"type": "str"},
        "finish_time": {"type": "str"},
        "time_sec": {"type": "float"},
        "margin": {"type": "str"},
        "passing_order": {"type": "any"},
        "last_3f": {"type": "any"},
        "odds": {"type": "any"},
        "popularity": {"type": "any"},
        "weight": {"type": "any"},
        "weight_change": {"type": "any"},
        "trainer_name": {"type": "str"},
        "trainer_id": {"type": "str"},
    },
    "entry_list_key": "entries",
}

_RACE_SHUTUBA: dict[str, Any] = {
    "top_required": {
        "race_id": {"type": "str"},
        "race_name": {"type": "str"},
        "entries": {"type": "list"},
    },
    "top_optional": {
        "date": {"type": "str"},
        "venue": {"type": "str"},
        "round": {"type": "int"},
        "surface": {"type": "str"},
        "distance": {"type": "int"},
        "direction": {"type": "str"},
        "weather": {"type": "str"},
        "track_condition": {"type": "str"},
        "start_time": {"type": "str"},
        "field_size": {"type": "int"},
        "grade": {"type": "str"},
        "race_class": {"type": "str"},
        "weight_rule": {"type": "str"},
        "course_type": {"type": "str"},
    },
    "entry_required": {
        "horse_number": {"type": "int"},
        "horse_name": {"type": "str", "non_empty": True},
        "horse_id": {"type": "str", "non_empty": True},
    },
    "entry_optional": {
        "bracket_number": {"type": "int"},
        "sex_age": {"type": "str"},
        "jockey_weight": {"type": "any"},
        "jockey_name": {"type": "str"},
        "jockey_id": {"type": "str"},
        "trainer_name": {"type": "str"},
        "trainer_id": {"type": "str"},
        "weight": {"type": "any"},
        "weight_change": {"type": "any"},
        "odds": {"type": "any"},
        "popularity": {"type": "any"},
        "sire": {"type": "str"},
        "dam_sire": {"type": "str"},
    },
    "entry_list_key": "entries",
}

_RACE_SHUTUBA_PAST: dict[str, Any] = {
    "top_required": {
        "race_id": {"type": "str"},
        "entries": {"type": "list"},
    },
    "top_optional": {
        "training": {"type": "list"},
    },
    "entry_required": {
        "horse_name": {"type": "str", "non_empty": True},
        "horse_id": {"type": "str", "non_empty": True},
        "past_races": {"type": "list"},
    },
    "entry_optional": {
        "bracket_number": {"type": "int"},
        "horse_number": {"type": "int"},
    },
    "entry_list_key": "entries",
}

_RACE_INDEX: dict[str, Any] = {
    "top_required": {
        "race_id": {"type": "str"},
        "entries": {"type": "list"},
    },
    "top_optional": {},
    "entry_required": {
        "horse_number": {"type": "int", "min": 1},
        "horse_name": {"type": "str", "non_empty": True},
        "horse_id": {"type": "str", "non_empty": True},
    },
    "entry_optional": {
        "time_index_m": {"type": "any"},
        "speed_max": {"type": "any"},
        "speed_avg": {"type": "any"},
        "speed_distance": {"type": "any"},
        "speed_course": {"type": "any"},
        "speed_recent": {"type": "list"},
        "all_txt_c": {"type": "list"},
        "odds": {"type": "any"},
        "popularity": {"type": "any"},
    },
    "entry_list_key": "entries",
}

_RACE_ODDS: dict[str, Any] = {
    "top_required": {
        "race_id": {"type": "str"},
        "entries": {"type": "list"},
    },
    "top_optional": {},
    "entry_required": {
        "horse_number": {"type": "int", "min": 1},
    },
    "entry_optional": {
        "win_odds": {"type": "any"},
        "place_odds_min": {"type": "any"},
        "place_odds_max": {"type": "any"},
        "popularity": {"type": "any"},
    },
    "entry_list_key": "entries",
}

_RACE_PAIR_ODDS: dict[str, Any] = {
    "top_required": {
        "race_id": {"type": "str"},
    },
    "top_optional": {
        "umaren": {"type": "list"},
        "wide": {"type": "list"},
        "umatan": {"type": "list"},
    },
    "lists": {
        "umaren": {
            "entry_required": {"pair": {"type": "list"}, "odds": {"type": "any"}},
            "entry_optional": {"popularity": {"type": "any"}},
        },
        "wide": {
            "entry_required": {"pair": {"type": "list"}},
            "entry_optional": {"odds_min": {"type": "any"}, "odds_max": {"type": "any"}, "popularity": {"type": "any"}},
        },
        "umatan": {
            "entry_required": {"pair": {"type": "list"}, "odds": {"type": "any"}},
            "entry_optional": {"popularity": {"type": "any"}},
        },
    },
}

_RACE_PADDOCK: dict[str, Any] = {
    "top_required": {
        "race_id": {"type": "str"},
        "entries": {"type": "list"},
    },
    "top_optional": {},
    "entry_required": {
        "horse_number": {"type": "int", "min": 1},
        "horse_name": {"type": "str", "non_empty": True},
        "horse_id": {"type": "str", "non_empty": True},
    },
    "entry_optional": {
        "paddock_rank": {"type": "any"},
        "paddock_comment": {"type": "str"},
    },
    "entry_list_key": "entries",
}

_RACE_BAROMETER: dict[str, Any] = {
    "top_required": {
        "race_id": {"type": "str"},
        "entries": {"type": "list"},
    },
    "top_optional": {},
    "entry_required": {
        "horse_number": {"type": "int", "min": 1},
        "horse_name": {"type": "str", "non_empty": True},
        "horse_id": {"type": "str", "non_empty": True},
    },
    "entry_optional": {
        "finish_order": {"type": "any"},
        "index_total": {"type": "any"},
        "index_start": {"type": "any"},
        "index_chase": {"type": "any"},
        "index_closing": {"type": "any"},
    },
    "entry_list_key": "entries",
}

_RACE_OIKIRI: dict[str, Any] = {
    "top_required": {
        "race_id": {"type": "str"},
        "entries": {"type": "list"},
    },
    "top_optional": {},
    "entry_required": {
        "horse_number": {"type": "int", "min": 1},
        "horse_name": {"type": "str", "non_empty": True},
        "horse_id": {"type": "str", "non_empty": True},
    },
    "entry_optional": {
        "training_date": {"type": "str"},
        "course": {"type": "str"},
        "condition": {"type": "str"},
        "rider": {"type": "str"},
        "lap_times": {"type": "str"},
        "impression": {"type": "str"},
        "evaluation": {"type": "str"},
        "comment": {"type": "str"},
    },
    "entry_list_key": "entries",
}

_RACE_RESULT_LAP: dict[str, Any] = {
    "top_required": {
        "race_id": {"type": "str"},
    },
    "top_optional": {
        "lap_times": {"type": "list"},
        "pace": {"type": "dict"},
        "corner_passing": {"type": "list"},
        "entries_lap": {"type": "list"},
        "horse_laptime_table": {"type": "dict"},
    },
}

_RACE_TRAINER_COMMENT: dict[str, Any] = {
    "top_required": {
        "race_id": {"type": "str"},
        "entries": {"type": "list"},
    },
    "top_optional": {},
    "entry_required": {
        "horse_name": {"type": "str", "non_empty": True},
    },
    "entry_optional": {
        "bracket_number": {"type": "int"},
        "horse_number": {"type": "int"},
        "horse_id": {"type": "str"},
        "comment": {"type": "str"},
        "evaluation": {"type": "str"},
        "trainer_name": {"type": "str"},
        "questioner": {"type": "str"},
    },
    "entry_list_key": "entries",
}

_RACE_DETAIL: dict[str, Any] = {
    "top_required": {
        "race_id": {"type": "str"},
        "race_name": {"type": "str"},
        "entries": {"type": "list"},
    },
    "top_optional": {
        "date": {"type": "str"},
        "venue": {"type": "str"},
        "round": {"type": "int"},
        "surface": {"type": "str"},
        "distance": {"type": "int"},
        "direction": {"type": "str"},
        "weather": {"type": "str"},
        "track_condition": {"type": "str"},
        "start_time": {"type": "str"},
        "field_size": {"type": "int"},
        "grade": {"type": "str"},
        "race_class": {"type": "str"},
        "weight_rule": {"type": "str"},
        "course_type": {"type": "str"},
    },
    "entry_required": {
        "horse_number": {"type": "int"},
        "horse_name": {"type": "str", "non_empty": True},
        "horse_id": {"type": "str", "non_empty": True},
    },
    "entry_optional": {
        "bracket_number": {"type": "int"},
        "sex_age": {"type": "str"},
        "jockey_weight": {"type": "any"},
        "jockey_name": {"type": "str"},
        "jockey_id": {"type": "str"},
        "trainer_name": {"type": "str"},
        "trainer_id": {"type": "str"},
        "weight": {"type": "any"},
        "weight_change": {"type": "any"},
        "odds": {"type": "any"},
        "popularity": {"type": "any"},
        "sire": {"type": "str"},
        "dam_sire": {"type": "str"},
        "speed_max": {"type": "any"},
        "speed_avg": {"type": "any"},
        "speed_distance": {"type": "any"},
        "speed_course": {"type": "any"},
        "speed_recent": {"type": "list"},
        "past_races": {"type": "list"},
    },
    "entry_list_key": "entries",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 馬系スキーマ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_HORSE_RESULT: dict[str, Any] = {
    "top_required": {
        "horse_id": {"type": "str", "non_empty": True},
        "horse_name": {"type": "str", "non_empty": True},
        "race_history": {"type": "list"},
    },
    "top_optional": {
        "name_en": {"type": "str"},
        "status": {"type": "str"},
        "sex": {"type": "str"},
        "age": {"type": "any"},
        "color": {"type": "str"},
        "birthday": {"type": "str"},
        "trainer": {"type": "str"},
        "owner": {"type": "str"},
        "breeder": {"type": "str"},
        "birthplace": {"type": "str"},
        "total_earnings": {"type": "any"},
        "career": {"type": "str"},
        "career_record": {"type": "list"},
        "major_wins": {"type": "list"},
        "sire": {"type": "str"},
        "dam": {"type": "str"},
        "dam_sire": {"type": "str"},
    },
    "entry_required": {
        "date": {"type": "str"},
        "race_id": {"type": "str"},
    },
    "entry_optional": {
        "venue": {"type": "str"},
        "weather": {"type": "str"},
        "race_round": {"type": "any"},
        "race_name": {"type": "str"},
        "field_size": {"type": "any"},
        "bracket_number": {"type": "any"},
        "horse_number": {"type": "any"},
        "odds": {"type": "any"},
        "popularity": {"type": "any"},
        "finish_position": {"type": "any"},
        "jockey_name": {"type": "str"},
        "jockey_weight": {"type": "any"},
        "surface": {"type": "str"},
        "distance": {"type": "any"},
        "time_index": {"type": "any"},
        "track_condition": {"type": "str"},
        "finish_time": {"type": "str"},
        "time_sec": {"type": "float"},
        "margin": {"type": "str"},
        "passing_order": {"type": "any"},
        "last_3f": {"type": "any"},
        "weight": {"type": "any"},
        "weight_change": {"type": "any"},
        "winner": {"type": "str"},
    },
    "entry_list_key": "race_history",
}

_HORSE_PEDIGREE_5GEN: dict[str, Any] = {
    "top_required": {
        "horse_id": {"type": "str", "non_empty": True},
        "ancestors": {"type": "list", "min_length": 1},
    },
    "top_optional": {
        "sire": {"type": "str"},
        "dam": {"type": "str"},
        "dam_sire": {"type": "str"},
        "ancestor_count": {"type": "int"},
        "source": {"type": "str"},
    },
}

_HORSE_TRAINING: dict[str, Any] = {
    "top_required": {
        "horse_id": {"type": "str", "non_empty": True},
        "entries": {"type": "list"},
    },
    "top_optional": {
        "total_items": {"type": "int"},
        "pages_fetched": {"type": "int"},
    },
    "entry_required": {
        "date": {"type": "str"},
    },
    "entry_optional": {
        "race_info": {"type": "str"},
        "day_of_week": {"type": "str"},
        "course": {"type": "str"},
        "track_condition": {"type": "str"},
        "rider": {"type": "str"},
        "time_raw": {"type": "str"},
        "lap_times": {"type": "any"},
        "position": {"type": "any"},
        "leg_color": {"type": "str"},
        "evaluation": {"type": "str"},
        "rank": {"type": "any"},
        "comment": {"type": "str"},
    },
    "entry_list_key": "entries",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 外部系スキーマ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_SMARTRC_RACE: dict[str, Any] = {
    "top_required": {
        "race_id": {"type": "str"},
    },
    "top_optional": {
        "rcode": {"type": "str"},
        "source": {"type": "str"},
        "runners": {"type": "list"},
        "horses": {"type": "dict"},
        "fullresults": {"type": "dict"},
    },
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 全スキーマ登録
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCHEMAS: dict[str, dict[str, Any]] = {
    "race_result": _RACE_RESULT,
    "race_shutuba": _RACE_SHUTUBA,
    "race_shutuba_past": _RACE_SHUTUBA_PAST,
    "race_index": _RACE_INDEX,
    "race_odds": _RACE_ODDS,
    "race_pair_odds": _RACE_PAIR_ODDS,
    "race_paddock": _RACE_PADDOCK,
    "race_barometer": _RACE_BAROMETER,
    "race_oikiri": _RACE_OIKIRI,
    "race_result_lap": _RACE_RESULT_LAP,
    "race_trainer_comment": _RACE_TRAINER_COMMENT,
    "race_detail": _RACE_DETAIL,
    "horse_result": _HORSE_RESULT,
    "horse_pedigree_5gen": _HORSE_PEDIGREE_5GEN,
    "horse_training": _HORSE_TRAINING,
    "smartrc_race": _SMARTRC_RACE,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# バリデーションエンジン
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _check_field(value: Any, spec: dict[str, Any]) -> str | None:
    """フィールド値を spec に照らし合わせ、最初のエラーメッセージを返す。None = OK."""
    expected_type = spec.get("type", "any")
    if expected_type == "any":
        return None

    py_types = _TYPE_MAP.get(expected_type)
    if py_types is not None:
        if not isinstance(py_types, tuple):
            py_types = (py_types,)
        if value is not None and not isinstance(value, py_types):
            return f"expected {expected_type}, got {type(value).__name__}"

    if spec.get("non_empty") and (value is None or (isinstance(value, str) and not value.strip())):
        return "non_empty violated"

    if expected_type in ("int", "float") and isinstance(value, (int, float)):
        lo = spec.get("min")
        hi = spec.get("max")
        if lo is not None and value < lo:
            return f"min={lo}, got {value}"
        if hi is not None and value > hi:
            return f"max={hi}, got {value}"

    if expected_type == "list" and isinstance(value, list):
        ml = spec.get("min_length")
        if ml is not None and len(value) < ml:
            return f"min_length={ml}, got {len(value)}"

    if expected_type == "str" and isinstance(value, str):
        pat = spec.get("pattern")
        if pat and not re.search(pat, value):
            return f"pattern {pat!r} not matched"

    return None


def _validate_entries(
    data: dict[str, Any],
    schema: dict[str, Any],
    list_key: str,
) -> dict[str, Any]:
    """entry_required / entry_optional に基づきエントリ群を検査する。"""
    items = data.get(list_key)
    if not isinstance(items, list):
        return {"entry_count": 0, "entry_issues": {}}

    req = schema.get("entry_required", {})
    missing_counts: dict[str, int] = {}
    type_error_counts: dict[str, int] = {}
    constraint_error_counts: dict[str, int] = {}

    for item in items:
        if not isinstance(item, dict):
            continue
        for field, spec in req.items():
            if field not in item or item[field] is None:
                missing_counts[field] = missing_counts.get(field, 0) + 1
                continue
            err = _check_field(item[field], spec)
            if err:
                if err.startswith("expected "):
                    type_error_counts[field] = type_error_counts.get(field, 0) + 1
                else:
                    constraint_error_counts[field] = constraint_error_counts.get(field, 0) + 1

    return {
        "entry_count": len(items),
        "entry_issues": {
            k: v
            for k, v in [
                ("missing_field_counts", missing_counts),
                ("type_error_counts", type_error_counts),
                ("constraint_error_counts", constraint_error_counts),
            ]
            if v
        },
    }


def validate(category: str, data: dict[str, Any] | None) -> dict[str, Any]:
    """
    category のスキーマに照らし data を検証する。

    Returns dict:
      passed           : bool  — 全テスト OK なら True
      schema_version   : int
      top_missing      : list[str]
      top_type_errors  : list[dict]
      top_constraint_errors : list[dict]
      entry_count      : int
      entry_issues     : dict
    """
    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "top_missing": [],
        "top_type_errors": [],
        "top_constraint_errors": [],
        "entry_count": 0,
        "entry_issues": {},
    }

    schema = SCHEMAS.get(category)
    if schema is None:
        base["passed"] = True
        base["skipped"] = f"no schema for {category}"
        return base

    if not isinstance(data, dict):
        base["passed"] = False
        base["top_missing"] = list(schema.get("top_required", {}).keys())
        return base

    # ── top-level required ──
    for field, spec in schema.get("top_required", {}).items():
        if field not in data:
            base["top_missing"].append(field)
            continue
        err = _check_field(data[field], spec)
        if err:
            rec = {"field": field, "expected": spec.get("type", "any"), "got": str(type(data[field]).__name__), "detail": err}
            if err.startswith("expected "):
                base["top_type_errors"].append(rec)
            else:
                base["top_constraint_errors"].append(rec)

    # ── entry-level validation ──
    list_key = schema.get("entry_list_key", "entries")
    if "entry_required" in schema:
        entry_result = _validate_entries(data, schema, list_key)
        base["entry_count"] = entry_result["entry_count"]
        base["entry_issues"] = entry_result["entry_issues"]

    # ── multi-list validation (race_pair_odds 等) ──
    lists_spec = schema.get("lists")
    if lists_spec:
        multi_issues: dict[str, Any] = {}
        for lkey, lschema in lists_spec.items():
            items = data.get(lkey)
            if not isinstance(items, list):
                continue
            sub = _validate_entries(data, lschema, lkey)
            if sub["entry_issues"]:
                multi_issues[lkey] = sub
        if multi_issues:
            base["multi_list_issues"] = multi_issues

    # ── passed 判定 ──
    has_issues = (
        base["top_missing"]
        or base["top_type_errors"]
        or base["top_constraint_errors"]
        or base.get("entry_issues")
        or base.get("multi_list_issues")
    )
    base["passed"] = not has_issues

    return base
