"""
全スクレイピングカテゴリの期待スキーマ定義 + バリデーション。

HybridStorage.save 経由では保存前に validate() を実行し、結果を
``_meta.schema_validation`` と ``_meta.scrape_validation_status`` に記録する。

厳格モード（既定）ではスキーマ定義があり検証に通らないデータは GCS に送らず
``SchemaValidationError`` を送出する（キュージョブは failed 扱い）。
``KEIBA_SCHEMA_STRICT=0`` のときは不合格でも保存し、メタに
``scrape_validation_status=schema_failed`` のみ付与する。

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

import copy
import json
from pathlib import Path
import re
from typing import Any



class SchemaValidationError(RuntimeError):
    """カテゴリスキーマ検証に不合格（厳格モードで保存前に送出）。メッセージに「どの値で」引っかかったかを含める。"""

    def __init__(self, category: str, key: str, report: dict[str, Any]) -> None:
        self.category = category
        self.key = key
        self.report = report
        msg = (
            f"[schema_validation] {category}/{key} "
            f"schema_version={report.get('schema_version')!r} passed=False"
        )
        vs = report.get("violations") or []
        if vs:
            msg += " :: " + " | ".join(describe_violation(v) for v in vs[:3])
            if len(vs) > 3:
                msg += f" | …他 {len(vs) - 3} 件"
        super().__init__(msg)


MAX_VIOLATIONS = 50          # 1 レコードあたり記録する違反の上限
MAX_VALUE_CHARS = 120        # 記録する「実際の値」の最大文字数
_IDENTIFYING_KEYS = ("horse_number", "horse_id", "horse_name", "date", "race_id", "generation", "position", "corner")


def describe_violation(v: dict[str, Any]) -> str:
    """違反 1 件を 1 行にする。例: entries[3].win_odds: type（期待 float / 実際 str '—'）[horse_number=4]"""
    where = ",".join(f"{k}={w}" for k, w in (v.get("where") or {}).items())
    field = v["field"].replace("[]", f"[{v['index']}]") if v.get("index") is not None else v["field"]
    return (f"{field}: {v['rule']}（期待 {v.get('expected')} / 実際 {v.get('actual_type')} {v.get('actual')}）"
            + (f" ※{where}" if where else ""))


def validation_report_for_meta(report: dict[str, Any]) -> dict[str, Any]:
    """_meta.schema_validation 用。JSON 直列化可能な dict のみを返す。"""
    try:
        json.dumps(report, ensure_ascii=False)
        return dict(report)
    except (TypeError, ValueError):
        return {"schema_version": report.get("schema_version"), "passed": False, "note": "non_json_report"}

_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "str": str,
    "int": (int,),
    "float": (int, float),
    "list": list,
    "dict": dict,
    "bool": bool,
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# スキーマ定義（正本: schema_defs.json。git 管理され、全環境で同一）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 各カテゴリ: top_required / top_optional / entry_list_key / entry_required / entry_optional / lists
#   advisory=True … 不合格でも保存を止めない（_meta に記録のみ）。実データでの検証を経て厳格化する。
#   _provenance   … 定義の根拠（source / samples / period など）。検証には使わない。
# 実データからの再構成: python -m src.scraper.schema_infer --help

_DEFS_PATH = Path(__file__).with_name("schema_defs.json")


def _load_defs(path: Path = _DEFS_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_DEFS = _load_defs()
SCHEMA_VERSION = _DEFS["schema_version"]
SCHEMAS: dict[str, dict[str, Any]] = _DEFS["categories"]

# スキーマを持たない理由（CATEGORY_MAP の全カテゴリは SCHEMAS か、ここのどちらかに載せる）
NO_SCHEMA_REASON: dict[str, str] = _DEFS.get("no_schema_reason", {})


def category_fingerprint(category: str) -> str:
    """1 カテゴリの定義のハッシュ。定義が変わったカテゴリだけ再検証するために使う。"""
    import hashlib

    canon = json.dumps({"v": SCHEMA_VERSION, "s": SCHEMAS.get(category)}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:12]


def schema_fingerprint() -> str:
    """定義全体のハッシュ。環境間で同一であること（git で共有されていること）の確認に使う。"""
    import hashlib

    canon = json.dumps({"v": SCHEMA_VERSION, "c": SCHEMAS, "n": NO_SCHEMA_REASON}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:12]


def _field_error(value: Any, spec: dict[str, Any]) -> tuple[str, str] | None:
    """フィールド値を spec に照らし、最初の違反を (規則, メッセージ) で返す。None = OK。"""
    expected_type = spec.get("type", "any")
    if expected_type == "any":
        return None

    py_types = _TYPE_MAP.get(expected_type)
    if py_types is not None:
        if not isinstance(py_types, tuple):
            py_types = (py_types,)
        if value is not None and not isinstance(value, py_types):
            return "type", f"expected {expected_type}, got {type(value).__name__}"

    if spec.get("non_empty") and (value is None or (isinstance(value, str) and not value.strip())):
        return "non_empty", "non_empty violated"

    if expected_type in ("int", "float") and isinstance(value, (int, float)):
        lo = spec.get("min")
        hi = spec.get("max")
        if lo is not None and value < lo:
            return "min", f"min={lo}, got {value}"
        if hi is not None and value > hi:
            return "max", f"max={hi}, got {value}"

    if expected_type == "list" and isinstance(value, list):
        ml = spec.get("min_length")
        if ml is not None and len(value) < ml:
            return "min_length", f"min_length={ml}, got {len(value)}"

    if expected_type == "str" and isinstance(value, str):
        pat = spec.get("pattern")
        if pat and not re.search(pat, value):
            return "pattern", f"pattern {pat!r} not matched"

    return None


def _check_field(value: Any, spec: dict[str, Any]) -> str | None:
    """フィールド値を spec に照らし合わせ、最初のエラーメッセージを返す。None = OK."""
    err = _field_error(value, spec)
    return err[1] if err else None


def _expected_text(rule: str, spec: dict[str, Any]) -> str:
    t = spec.get("type", "any")
    return {"type": t, "non_empty": f"{t} 非空", "min": f"{t} ≥ {spec.get('min')}", "max": f"{t} ≤ {spec.get('max')}",
            "min_length": f"list 長さ ≥ {spec.get('min_length')}", "pattern": f"{t} /{spec.get('pattern')}/",
            "missing": t, "null": t}.get(rule, t)


def _show(value: Any) -> str:
    """違反した「実際の値」を、長さを抑えて文字列にする。"""
    if isinstance(value, (list, dict)):
        text = f"{type(value).__name__}(len={len(value)}) " + json.dumps(value, ensure_ascii=False, default=str)
    else:
        text = repr(value)
    return text if len(text) <= MAX_VALUE_CHARS else text[:MAX_VALUE_CHARS] + "…"


def _violation(field: str, rule: str, spec: dict[str, Any], value: Any, *, absent: bool = False, index: int | None = None,
               item: dict[str, Any] | None = None, message: str = "") -> dict[str, Any]:
    v: dict[str, Any] = {"field": field, "rule": rule, "expected": _expected_text(rule, spec),
                         "actual": "<キー無し>" if absent else _show(value),
                         "actual_type": "absent" if absent else type(value).__name__}
    if message:
        v["message"] = message
    if index is not None:
        v["index"] = index
    if item:
        where = {k: item[k] for k in _IDENTIFYING_KEYS if k in item and not isinstance(item[k], (list, dict))}
        if where:
            v["where"] = dict(list(where.items())[:3])
    return v


def _validate_entries(
    data: dict[str, Any],
    schema: dict[str, Any],
    list_key: str,
    violations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """entry_required / entry_optional に基づきエントリ群を検査する。違反の中身は violations に追加する。"""
    items = data.get(list_key)
    if not isinstance(items, list):
        return {"entry_count": 0, "entry_issues": {}}

    req = schema.get("entry_required", {})
    missing_counts: dict[str, int] = {}
    type_error_counts: dict[str, int] = {}
    constraint_error_counts: dict[str, int] = {}

    def add(v: dict[str, Any]) -> None:
        if violations is not None and len(violations) < MAX_VIOLATIONS:
            violations.append(v)

    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        for field, spec in req.items():
            name = f"{list_key}[].{field}"
            if field not in item or item[field] is None:
                missing_counts[field] = missing_counts.get(field, 0) + 1
                add(_violation(name, "missing" if field not in item else "null", spec, None, absent=field not in item,
                               index=i, item=item))
                continue
            err = _field_error(item[field], spec)
            if err:
                rule, msg = err
                if msg.startswith("expected "):
                    type_error_counts[field] = type_error_counts.get(field, 0) + 1
                else:
                    constraint_error_counts[field] = constraint_error_counts.get(field, 0) + 1
                add(_violation(name, rule, spec, item[field], index=i, item=item, message=msg))

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
      violations       : list[dict] — 違反ごとの中身（field / rule / expected / actual / actual_type / index / where）。
                         1 レコード MAX_VIOLATIONS 件まで、実際の値は MAX_VALUE_CHARS 文字まで
      violations_total : int  — 上限で切る前の違反数の目安（件数ベース）
    """
    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "top_missing": [],
        "top_type_errors": [],
        "top_constraint_errors": [],
        "entry_count": 0,
        "entry_issues": {},
        "violations": [],
    }

    schema = SCHEMAS.get(category)
    if schema is None:
        base["passed"] = True
        base["skipped"] = f"no schema for {category}"
        return base

    if not isinstance(data, dict):
        base["passed"] = False
        base["top_missing"] = list(schema.get("top_required", {}).keys())
        base["violations"] = [{"field": "(全体)", "rule": "type", "expected": "dict", "actual": _show(data),
                               "actual_type": type(data).__name__}]
        return base

    violations: list[dict[str, Any]] = base["violations"]

    # ── top-level required ──
    for field, spec in schema.get("top_required", {}).items():
        if field not in data:
            base["top_missing"].append(field)
            violations.append(_violation(field, "missing", spec, None, absent=True))
            continue
        err = _field_error(data[field], spec)
        if err:
            rule, msg = err
            rec = {"field": field, "expected": spec.get("type", "any"), "got": str(type(data[field]).__name__), "detail": msg}
            if msg.startswith("expected "):
                base["top_type_errors"].append(rec)
            else:
                base["top_constraint_errors"].append(rec)
            violations.append(_violation(field, rule, spec, data[field], message=msg))

    # ── entry-level validation ──
    list_key = schema.get("entry_list_key", "entries")
    if "entry_required" in schema:
        entry_result = _validate_entries(data, schema, list_key, violations)
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
            sub = _validate_entries(data, lschema, lkey, violations)
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
    if schema.get("advisory"):
        base["advisory"] = True     # 診断のみ。不合格でも保存は止めない
    if len(violations) >= MAX_VIOLATIONS:
        base["violations_truncated"] = True
    del violations[MAX_VIOLATIONS:]

    return base
