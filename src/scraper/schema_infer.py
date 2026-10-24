"""実際に取得したデータからスキーマ定義（schema_defs.json）を再構成する。

流れ:
  collect … 実データを走査して「観測プロファイル」を作る（キーの出現率・型・空/null・値のパターン・範囲）。
            プロファイルは加算マージできるので、学習PC・VPS など複数環境・複数回の結果を積み上げられる。
  report  … プロファイルと現行スキーマの差（未定義キー・必須の出現率・型の食い違い・適合率）を出す。
  apply   … プロファイルから導いた定義を schema_defs.json へ反映する（保守的。下記）。

プロファイルは ``docs/requirements/data/schemas/observed/<category>.json``、定義は ``schema_defs.json``。
どちらも git 管理する（全環境で同一の定義を使うため）。

データの出所（--source）:
  samples … docs/requirements/data/scrape_process_samples/（過去に実スクレイプして保存したサンプル。1件ずつ）
  storage … 現在の環境の HybridStorage（stg/prod は GCS を list + 必要分 download。dev は data/dev_mock だが、
            モックは架空データなので ``_meta.dev_mock`` 付きは必ず除外する）
  dir:PATH … GCS ミラー配置のディレクトリ（``<category>/<shard>/<key>.json``、``others/<category>/<key>.json``）

反映の方針（apply）: 既存の必須/任意・制約は観測だけで勝手に変えない。
  - 既定: 新カテゴリの追加（advisory）と、観測されたが未定義のキーを任意として追加、provenance の更新のみ。
  - ``--promote`` / ``--demote``: 出現率が十分な任意→必須／必須の出現率不足→任意 を反映。
  - 型の食い違い・一度も観測されない定義済みキーは report に出すだけ（人が判断）。
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from src.scraper import schemas

ROOT = Path(__file__).resolve().parents[2]
OBSERVED_DIR = ROOT / "docs" / "requirements" / "data" / "schemas" / "observed"
SAMPLES_DIR = ROOT / "docs" / "requirements" / "data" / "scrape_process_samples"
DEFS_PATH = schemas._DEFS_PATH

MAX_DISTINCT = 25             # 値の種類を数える上限（これを超えたら列挙しない）
REQUIRED_RATIO = 0.995        # これ以上の出現率なら必須候補
MIN_SAMPLES = 30              # 必須/任意を判断するのに必要な最小サンプル数（レコード数）

# サンプルファイル名 → カテゴリ（生 HTML 由来の *_html は対象外）
SAMPLE_CATEGORY = {
    "nk_shutuba_entries": "race_shutuba", "nk_shutuba_race_meta": "race_shutuba_meta", "nk_speed_index": "race_index",
    "nk_barometer": "race_barometer", "nk_paddock": "race_paddock", "nk_odds": "race_odds",
    "nk_result_on_time": "race_result_on_time", "nk_db_race_result": "race_result", "nk_db_per_horse_lap": "race_result_lap",
    "nk_db_race_info": "race_result_meta", "nk_db_payoff": "race_result_payoff", "nk_db_track": "race_result_track",
    "nk_db_corner": "race_result_corner", "nk_db_lap": "race_result_lap_times", "nk_horse_profile": "horse_profile",
    "nk_horse_history": "horse_race_history", "nk_horse_pedigree": "horse_pedigree_5gen", "nk_horse_training": "horse_training",
    "nk_race_list": "race_lists", "nk_race_day_schedule": "race_day_schedule",
}


# ── 値の分類 ──────────────────────────────────────────────────────────────

def type_name(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "dict"
    return type(v).__name__


def str_pattern(s: str) -> str:
    if not s.strip():
        return "empty"
    if s.isdigit():
        return f"digits{len(s)}"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return "date-iso"
    if re.fullmatch(r"\d{4}/\d{2}/\d{2}", s):
        return "date-slash"
    if re.fullmatch(r"\d{1,2}:\d{2}", s):
        return "time-hhmm"
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        return "numeric-str"
    return "text"


def new_stat() -> dict[str, Any]:
    return {"n": 0, "null": 0, "empty": 0, "types": {}, "patterns": {}, "values": {}, "values_overflow": False}


def observe(stat: dict[str, Any], value: Any) -> None:
    stat["n"] += 1
    t = type_name(value)
    stat["types"][t] = stat["types"].get(t, 0) + 1
    if value is None:
        stat["null"] += 1
        return
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        stat["min"] = value if "min" not in stat else min(stat["min"], value)
        stat["max"] = value if "max" not in stat else max(stat["max"], value)
    elif isinstance(value, str):
        p = str_pattern(value)
        stat["patterns"][p] = stat["patterns"].get(p, 0) + 1
        if p == "empty":
            stat["empty"] += 1
        stat["len_min"] = len(value) if "len_min" not in stat else min(stat["len_min"], len(value))
        stat["len_max"] = len(value) if "len_max" not in stat else max(stat["len_max"], len(value))
        if p in ("text", "empty") and len(value) <= 20:
            _count_value(stat, value)
    elif isinstance(value, (list, dict)):
        if not value:
            stat["empty"] += 1
        stat["len_min"] = len(value) if "len_min" not in stat else min(stat["len_min"], len(value))
        stat["len_max"] = len(value) if "len_max" not in stat else max(stat["len_max"], len(value))


def _count_value(stat: dict[str, Any], value: str) -> None:
    vals = stat["values"]
    if value in vals or len(vals) < MAX_DISTINCT:
        vals[value] = vals.get(value, 0) + 1
    else:
        stat["values_overflow"] = True


def merge_stat(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = {"n": a["n"] + b["n"], "null": a["null"] + b["null"], "empty": a["empty"] + b["empty"],
           "types": _add(a["types"], b["types"]), "patterns": _add(a["patterns"], b["patterns"]),
           "values": dict(a["values"]), "values_overflow": a["values_overflow"] or b["values_overflow"]}
    for v, c in b["values"].items():
        if v in out["values"] or len(out["values"]) < MAX_DISTINCT:
            out["values"][v] = out["values"].get(v, 0) + c
        else:
            out["values_overflow"] = True
    for k, fn in (("min", min), ("max", max), ("len_min", min), ("len_max", max)):
        vals = [s[k] for s in (a, b) if k in s]
        if vals:
            out[k] = fn(vals)
    return out


def _add(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + v
    return out


# ── プロファイル ────────────────────────────────────────────────────────────

def new_profile(category: str) -> dict[str, Any]:
    return {"category": category, "records": 0, "top": {}, "lists": {}, "nested": {}, "conformance": {"validated": 0, "passed": 0, "failures": {}},
            "keys": {"min": None, "max": None}, "runs": []}


def add_record(profile: dict[str, Any], data: dict[str, Any], key: str = "") -> None:
    """1 レコード（1 ファイル分の JSON）を観測する。"""
    profile["records"] += 1
    for k, v in data.items():
        if k == "_meta":
            continue
        observe(profile["top"].setdefault(k, new_stat()), v)
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            lst = profile["lists"].setdefault(k, {"items": 0, "records": 0, "fields": {}})
            lst["records"] += 1
            for item in v:
                lst["items"] += 1
                for fk, fv in item.items():
                    observe(lst["fields"].setdefault(fk, new_stat()), fv)
        elif isinstance(v, dict):
            sub = profile["nested"].setdefault(k, {})
            for sk, sv in v.items():
                observe(sub.setdefault(sk, new_stat()), sv)
    rep = schemas.validate(profile["category"], data)
    if "skipped" not in rep:
        c = profile["conformance"]
        c["validated"] += 1
        c["passed"] += 1 if rep["passed"] else 0
        for f in rep["top_missing"]:
            c["failures"][f"top_missing:{f}"] = c["failures"].get(f"top_missing:{f}", 0) + 1
        for e in rep["top_type_errors"] + rep["top_constraint_errors"]:
            kk = f"top_error:{e['field']}"
            c["failures"][kk] = c["failures"].get(kk, 0) + 1
        for grp, d in (rep.get("entry_issues") or {}).items():
            for f, n in d.items():
                kk = f"entry_{grp.split('_')[0]}:{f}"
                c["failures"][kk] = c["failures"].get(kk, 0) + n
    if key:
        kmin, kmax = profile["keys"]["min"], profile["keys"]["max"]
        profile["keys"] = {"min": key if kmin is None else min(kmin, key), "max": key if kmax is None else max(kmax, key)}


def merge_profiles(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = new_profile(a["category"])
    out["records"] = a["records"] + b["records"]
    for part in ("top",):
        for k in set(a[part]) | set(b[part]):
            out[part][k] = merge_stat(a[part].get(k) or new_stat(), b[part].get(k) or new_stat())
    for k in set(a["lists"]) | set(b["lists"]):
        la = a["lists"].get(k, {"items": 0, "records": 0, "fields": {}})
        lb = b["lists"].get(k, {"items": 0, "records": 0, "fields": {}})
        fields = {f: merge_stat(la["fields"].get(f) or new_stat(), lb["fields"].get(f) or new_stat())
                  for f in set(la["fields"]) | set(lb["fields"])}
        out["lists"][k] = {"items": la["items"] + lb["items"], "records": la["records"] + lb["records"], "fields": fields}
    for k in set(a["nested"]) | set(b["nested"]):
        na, nb = a["nested"].get(k, {}), b["nested"].get(k, {})
        out["nested"][k] = {f: merge_stat(na.get(f) or new_stat(), nb.get(f) or new_stat()) for f in set(na) | set(nb)}
    ca, cb = a["conformance"], b["conformance"]
    out["conformance"] = {"validated": ca["validated"] + cb["validated"], "passed": ca["passed"] + cb["passed"],
                          "failures": _add(ca["failures"], cb["failures"])}
    keys = [x for x in (a["keys"]["min"], b["keys"]["min"]) if x is not None]
    keys_max = [x for x in (a["keys"]["max"], b["keys"]["max"]) if x is not None]
    out["keys"] = {"min": min(keys) if keys else None, "max": max(keys_max) if keys_max else None}
    out["runs"] = a["runs"] + b["runs"]
    return out


def profile_path(category: str, directory: Path = OBSERVED_DIR) -> Path:
    return directory / f"{category}.json"


def load_profile(category: str, directory: Path = OBSERVED_DIR) -> dict[str, Any] | None:
    p = profile_path(category, directory)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save_profile(profile: dict[str, Any], directory: Path = OBSERVED_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    p = profile_path(profile["category"], directory)
    p.write_text(json.dumps(profile, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return p


# ── データの出所 ─────────────────────────────────────────────────────────────

Record = tuple[str, str, dict[str, Any]]      # (category, key, data)


def _is_mock(data: dict[str, Any]) -> bool:
    return bool((data.get("_meta") or {}).get("dev_mock"))


def iter_samples(samples_dir: Path = SAMPLES_DIR) -> Iterator[Record]:
    for name, cat in SAMPLE_CATEGORY.items():
        p = samples_dir / f"{name}.json"
        if not p.is_file():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and not _is_mock(data):
            yield cat, str(data.get("race_id") or data.get("horse_id") or data.get("date") or name), data


def iter_dir(root: Path, categories: Iterable[str], per_category: int) -> Iterator[Record]:
    for cat in categories:
        files = sorted([*root.glob(f"{cat}/*/*.json"), *root.glob(f"others/{cat}/*.json")], key=lambda p: p.stem)
        for p in _spread(files, per_category):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(data, dict) and not _is_mock(data):
                yield cat, p.stem, data


def _spread(items: list, n: int) -> list:
    """全体から等間隔に n 件（年・期間が偏らないように）。"""
    if n <= 0 or len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def iter_storage(storage: Any, categories: Iterable[str], per_category: int, years: list[str]) -> Iterator[Record]:
    cmap = storage.CATEGORY_MAP
    for cat in categories:
        if cmap.get(cat) == "local_only":
            keys = storage.list_keys(cat)
        else:
            keys = sorted(k for y in years for k in (storage.batch_list_blobs(cat, y) or {}))
        for key in _spread(keys, per_category):
            data = storage.load(cat, key)
            if isinstance(data, dict) and not _is_mock(data):
                yield cat, key, data


# ── collect / 導出 ──────────────────────────────────────────────────────────

def collect(records: Iterable[Record], *, source: str, merge: bool = True, directory: Path = OBSERVED_DIR
            ) -> dict[str, dict[str, Any]]:
    fresh: dict[str, dict[str, Any]] = {}
    for cat, key, data in records:
        add_record(fresh.setdefault(cat, new_profile(cat)), data, key)
    out = {}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for cat, prof in fresh.items():
        prof["runs"] = [{"at": now, "source": source, "host": socket.gethostname(), "records": prof["records"]}]
        prev = load_profile(cat, directory) if merge else None
        final = merge_profiles(prev, prof) if prev else prof
        save_profile(final, directory)
        out[cat] = final
    return out


def _ratio(stat: dict[str, Any], denom: int) -> float:
    return (stat["n"] - stat["null"]) / denom if denom else 0.0


def _dominant_type(stat: dict[str, Any]) -> str:
    types = {t for t, c in stat["types"].items() if t != "null" and c}
    if not types:
        return "any"
    if types == {"int"}:
        return "int"
    if types <= {"int", "float"}:
        return "float"
    return next(iter(types)) if len(types) == 1 else "any"


def _constraints(stat: dict[str, Any], typ: str, required: bool) -> dict[str, Any]:
    spec: dict[str, Any] = {"type": typ}
    nonnull = stat["n"] - stat["null"]
    if typ == "str" and required and nonnull and stat["empty"] == 0:
        spec["non_empty"] = True
    if typ == "str" and nonnull and len(stat["patterns"]) == 1:
        (pat,) = stat["patterns"]
        m = re.fullmatch(r"digits(\d+)", pat)
        if m and stat.get("len_min") == stat.get("len_max"):
            spec["pattern"] = rf"^\d{{{m.group(1)}}}$"
    if typ == "list" and required and nonnull and stat["empty"] == 0:
        spec["min_length"] = 1
    return spec


def derive_fields(fields: dict[str, dict[str, Any]], denom: int, *, enough: bool, ratio: float
                  ) -> tuple[dict[str, Any], dict[str, Any]]:
    req: dict[str, Any] = {}
    opt: dict[str, Any] = {}
    for k, st in sorted(fields.items()):
        typ = _dominant_type(st)
        is_req = enough and _ratio(st, denom) >= ratio
        spec = _constraints(st, typ, is_req)
        (req if is_req else opt)[k] = spec
    return req, opt


def derive_schema(profile: dict[str, Any], *, min_samples: int = MIN_SAMPLES, ratio: float = REQUIRED_RATIO) -> dict[str, Any]:
    """プロファイルから定義を導く。レコード数が足りないキーは任意にする（根拠不足）。"""
    schema: dict[str, Any] = {}
    enough = profile["records"] >= min_samples          # 1 ファイル内の複数要素は相関するので、ファイル数で根拠を数える
    top_req, top_opt = derive_fields(profile["top"], profile["records"], enough=enough, ratio=ratio)
    schema["top_required"], schema["top_optional"] = top_req, top_opt
    if profile["lists"]:
        key = max(profile["lists"], key=lambda k: profile["lists"][k]["items"])
        lst = profile["lists"][key]
        e_req, e_opt = derive_fields(lst["fields"], lst["items"], enough=enough and lst["items"] >= min_samples, ratio=ratio)
        if key != "entries":
            schema["entry_list_key"] = key
        schema["entry_required"], schema["entry_optional"] = e_req, e_opt
    return schema


# ── 既存定義との照合・反映 ──────────────────────────────────────────────────

def reconcile(existing: dict[str, Any] | None, derived: dict[str, Any], profile: dict[str, Any], *, promote: bool,
              demote: bool, min_samples: int = MIN_SAMPLES, ratio: float = REQUIRED_RATIO
              ) -> tuple[dict[str, Any], list[dict[str, str]]]:
    changes: list[dict[str, str]] = []
    enough_records = profile["records"] >= min_samples
    if existing is None:
        new = dict(derived)
        new["advisory"] = True
        changes.append({"kind": "new_category", "detail": f"{profile['records']} レコードから新規定義（advisory）"})
        return new, changes
    new = json.loads(json.dumps(existing))
    for scope, fields_key, denom, stats in (
        ("top", ("top_required", "top_optional"), profile["records"], profile["top"]),
        ("entry", ("entry_required", "entry_optional"),
         (profile["lists"].get(existing.get("entry_list_key", "entries"), {}) or {}).get("items", 0),
         (profile["lists"].get(existing.get("entry_list_key", "entries"), {}) or {}).get("fields", {})),
    ):
        rk, ok = fields_key
        if rk not in existing and ok not in existing:
            continue
        req, opt = new.setdefault(rk, {}), new.setdefault(ok, {})
        for k, st in stats.items():
            typ = _dominant_type(st)
            r = _ratio(st, denom)
            if k not in req and k not in opt:
                opt[k] = {"type": typ}
                changes.append({"kind": "add_optional", "detail": f"{scope}.{k} ({typ}, 出現率 {r:.1%})"})
                continue
            cur = req.get(k) or opt.get(k)
            if cur.get("type", "any") not in ("any", typ) and not (cur["type"] == "float" and typ == "int"):
                changes.append({"kind": "type_conflict", "detail": f"{scope}.{k}: 定義 {cur['type']} / 観測 {typ}（要判断）"})
            if k in opt and enough_records and denom >= min_samples and r >= ratio:
                if promote:
                    req[k] = opt.pop(k)
                changes.append({"kind": "promote" if promote else "promote_candidate",
                                "detail": f"{scope}.{k} の出現率 {r:.1%}（任意→必須）"})
            if k in req and enough_records and denom >= min_samples and r < ratio:
                if demote:
                    opt[k] = req.pop(k)
                changes.append({"kind": "demote" if demote else "demote_candidate",
                                "detail": f"{scope}.{k} の出現率 {r:.1%}（必須→任意）"})
        if enough_records and denom >= min_samples:
            for k in [*req, *opt]:
                if k not in stats:
                    changes.append({"kind": "never_observed", "detail": f"{scope}.{k}: {denom} 件で一度も観測されない"})
    return new, changes


def provenance(profile: dict[str, Any]) -> dict[str, Any]:
    return {"source": "observed", "samples": profile["records"], "conformance": profile["conformance"],
            "key_range": profile["keys"], "runs": profile["runs"][-5:],
            "note": "実データの観測プロファイル（docs/requirements/data/schemas/observed/）から再構成"}


def load_defs() -> dict[str, Any]:
    return json.loads(DEFS_PATH.read_text(encoding="utf-8"))


def apply_profiles(profiles: dict[str, dict[str, Any]], *, promote: bool = False, demote: bool = False,
                   min_samples: int = MIN_SAMPLES, write: bool = True) -> dict[str, list[dict[str, str]]]:
    defs = load_defs()
    report: dict[str, list[dict[str, str]]] = {}
    for cat, prof in sorted(profiles.items()):
        derived = derive_schema(prof, min_samples=min_samples)
        new, changes = reconcile(defs["categories"].get(cat), derived, prof, promote=promote, demote=demote, min_samples=min_samples)
        new["_provenance"] = provenance(prof)
        defs["categories"][cat] = new
        report[cat] = changes
    if write:
        DEFS_PATH.write_text(json.dumps(defs, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return report


# ── 出力 ────────────────────────────────────────────────────────────────────

def render_report(profiles: dict[str, dict[str, Any]], changes: dict[str, list[dict[str, str]]] | None = None) -> str:
    lines = ["カテゴリ別: 観測レコード数 / 現行スキーマへの適合 / 差分"]
    for cat, prof in sorted(profiles.items()):
        c = prof["conformance"]
        conf = f"{c['passed']}/{c['validated']} 適合" if c["validated"] else "スキーマ未定義"
        lines.append(f"\n■ {cat}: {prof['records']} レコード（{prof['keys']['min']}〜{prof['keys']['max']}） {conf}")
        for k, n in sorted(c["failures"].items(), key=lambda x: -x[1])[:8]:
            lines.append(f"    不適合: {k} × {n}")
        for ch in (changes or {}).get(cat, []):
            lines.append(f"    [{ch['kind']}] {ch['detail']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="実データを走査して観測プロファイルを保存（既存に加算マージ）")
    c.add_argument("--source", default="samples", help="samples | storage | dir:PATH")
    c.add_argument("--category", action="append", help="対象カテゴリ（複数可。省略時は定義済み＋地図上の全カテゴリ）")
    c.add_argument("--per-category", type=int, default=200, help="カテゴリごとの最大件数（等間隔に抽出）")
    c.add_argument("--years", help="storage のとき対象の年（カンマ区切り。既定: 2020〜今年）")
    c.add_argument("--replace", action="store_true", help="既存プロファイルに加算せず置き換える")
    sub.add_parser("report", help="保存済みプロファイルと現行スキーマの差を表示")
    a = sub.add_parser("apply", help="プロファイルから schema_defs.json を更新（保守的）")
    a.add_argument("--promote", action="store_true", help="出現率が十分な任意キーを必須にする")
    a.add_argument("--demote", action="store_true", help="出現率が足りない必須キーを任意にする")
    a.add_argument("--min-samples", type=int, default=MIN_SAMPLES)
    a.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "collect":
        from src.scraper.storage import HybridStorage

        cats = args.category or sorted(set(schemas.SCHEMAS) | {k for k, v in HybridStorage.CATEGORY_MAP.items()})
        if args.source == "samples":
            records = iter_samples()
        elif args.source.startswith("dir:"):
            records = iter_dir(Path(args.source[4:]).expanduser(), cats, args.per_category)
        elif args.source == "storage":
            from src.utils.project_env import load_project_dotenv

            load_project_dotenv()
            years = args.years.split(",") if args.years else [str(y) for y in range(2020, datetime.now().year + 1)]
            records = iter_storage(HybridStorage(base_dir=str(ROOT)), cats, args.per_category, years)
        else:
            ap.error("--source は samples / storage / dir:PATH")
        profiles = collect(records, source=args.source, merge=not args.replace)
        print(render_report(profiles))
        print(f"\nプロファイル: {OBSERVED_DIR}（{len(profiles)} カテゴリ）")
        return 0
    profiles = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(OBSERVED_DIR.glob("*.json"))}
    if not profiles:
        print("プロファイルがありません。先に collect を実行してください。")
        return 1
    if args.cmd == "report":
        defs = load_defs()
        changes = {}
        for cat, prof in profiles.items():
            derived = derive_schema(prof)
            _, changes[cat] = reconcile(defs["categories"].get(cat), derived, prof, promote=False, demote=False)
        print(render_report(profiles, changes))
        return 0
    changes = apply_profiles(profiles, promote=args.promote, demote=args.demote, min_samples=args.min_samples, write=not args.dry_run)
    print(render_report(profiles, changes))
    print("\n" + ("（dry-run: 書き込みなし）" if args.dry_run else f"更新: {DEFS_PATH}（schema_fingerprint={schemas.schema_fingerprint()} は再読込後に変わります）"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
