"""
スキーマサンプリングによるリファレンス定義の構築。

年ごとに (グレード x コース種別 x 開催場所) の全組み合わせから
N件をランダムサンプリングし、2/3 多数決でスキーマを特定する。

Usage:
    python -m scraper.schema_sampler --years 2024
    python -m scraper.schema_sampler --years 2022,2023,2024 --samples 3
    python -m scraper.schema_sampler --years 2024 --export-tables
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from utils.keiba_logging import script_basic_config

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))

JRA_PLACE_CODES = frozenset(f"{i:02d}" for i in range(1, 11))
JRA_VENUE_NAMES = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}

SAMPLE_CATEGORIES = [
    "race_result", "race_shutuba", "race_index", "race_shutuba_past",
    "race_odds", "race_paddock", "race_barometer", "race_oikiri",
    "race_trainer_comment", "race_result_lap", "race_pair_odds", "smartrc_race",
]

_OBSTACLE_PATTERN = re.compile(r"障害|ジャンプ|ステープル|ハードル")

GRADE_MAP = {
    "新馬": "新馬", "未勝利": "未勝利",
    "1勝": "1勝", "2勝": "2勝", "3勝": "3勝",
    "500万下": "1勝", "1000万下": "2勝", "1600万下": "3勝",
    "オープン": "OP", "OP": "OP", "L": "L",
    "G3": "G3", "G2": "G2", "G1": "G1",
}


def is_jra_race(race_id: str) -> bool:
    return len(race_id) >= 6 and race_id[4:6] in JRA_PLACE_CODES


def _venue_from_race_id(race_id: str) -> str:
    code = race_id[4:6] if len(race_id) >= 6 else "??"
    return JRA_VENUE_NAMES.get(code, code)


def _classify_race(data: dict) -> tuple[str, str, str] | None:
    """race_shutuba データからレースを (grade, surface, venue) に分類する。"""
    race_id = data.get("race_id", "")
    if not is_jra_race(race_id):
        return None

    venue = _venue_from_race_id(race_id)

    race_name = data.get("race_name", "")
    surface_raw = data.get("surface", "")
    if _OBSTACLE_PATTERN.search(race_name):
        surface = "障害"
    elif surface_raw in ("芝", "ダート"):
        surface = surface_raw
    elif "芝" in surface_raw:
        surface = "芝"
    elif "ダ" in surface_raw:
        surface = "ダート"
    else:
        surface = surface_raw or "不明"

    grade_raw = data.get("grade", "")
    race_class = data.get("race_class", "")

    grade = "不明"
    for src in [grade_raw, race_class]:
        for key, val in GRADE_MAP.items():
            if key in str(src):
                grade = val
                break
        if grade != "不明":
            break

    if grade == "不明" and race_class:
        grade = str(race_class)

    return (grade, surface, venue)


def _field_info(value: Any) -> dict:
    """値が non-empty かどうかを判定する。"""
    if value is None:
        return {"present": True, "non_empty": False}
    if isinstance(value, str) and value.strip() == "":
        return {"present": True, "non_empty": False}
    if isinstance(value, (list, dict)) and len(value) == 0:
        return {"present": True, "non_empty": False}
    return {"present": True, "non_empty": True}


def _analyze_data(data: dict) -> dict:
    """1レースのデータからフィールド情報を抽出する。"""
    result = {"top_fields": {}, "entry_fields": {}}

    for k, v in data.items():
        if k == "entries":
            continue
        if k == "_meta":
            continue
        result["top_fields"][k] = _field_info(v)

    entries = data.get("entries", [])
    if isinstance(entries, list) and entries:
        all_keys: set[str] = set()
        for e in entries:
            if isinstance(e, dict):
                all_keys.update(e.keys())
        for field in sorted(all_keys):
            non_empty_count = 0
            for e in entries:
                if isinstance(e, dict):
                    info = _field_info(e.get(field))
                    if info["non_empty"]:
                        non_empty_count += 1
            result["entry_fields"][field] = {
                "present": True,
                "non_empty": non_empty_count > 0,
            }

    return result


def _merge_field_stats(analyses: list[dict], threshold: float = 2 / 3) -> dict:
    """複数サンプルのフィールド情報を多数決でマージする。"""
    n = len(analyses)
    merged = {"sample_count": n, "top_fields": {}, "entry_fields": {}}

    for section in ("top_fields", "entry_fields"):
        all_fields: set[str] = set()
        for a in analyses:
            all_fields.update(a.get(section, {}).keys())

        for field in sorted(all_fields):
            present_count = 0
            non_empty_count = 0
            for a in analyses:
                info = a.get(section, {}).get(field)
                if info:
                    if info.get("present"):
                        present_count += 1
                    if info.get("non_empty"):
                        non_empty_count += 1
            merged[section][field] = {
                "present": present_count,
                "non_empty": non_empty_count,
                "expected": non_empty_count >= max(1, n * threshold),
            }

    return merged


def _build_summary(schemas: dict[str, dict]) -> dict:
    """全カテゴリ横断で、surface / grade / venue 軸での差異フィールドを検出する。"""
    summary: dict[str, Any] = {
        "fields_varying_by_surface": [],
        "fields_varying_by_grade": [],
        "fields_varying_by_venue": [],
        "fields_uniform": [],
    }

    for cat, combos in schemas.items():
        expected_by_surface: dict[str, set[str]] = defaultdict(set)
        expected_by_grade: dict[str, set[str]] = defaultdict(set)
        expected_by_venue: dict[str, set[str]] = defaultdict(set)

        for combo_key, schema in combos.items():
            parts = combo_key.strip("()").split(", ")
            if len(parts) != 3:
                continue
            grade, surface, venue = parts

            for section in ("top_fields", "entry_fields"):
                for field, info in schema.get(section, {}).items():
                    if info.get("expected"):
                        tag = f"{cat}.{section}.{field}"
                        expected_by_surface[surface].add(tag)
                        expected_by_grade[grade].add(tag)
                        expected_by_venue[venue].add(tag)

        all_fields = set()
        for s in expected_by_surface.values():
            all_fields |= s
        for s in expected_by_grade.values():
            all_fields |= s
        for s in expected_by_venue.values():
            all_fields |= s

        for field in sorted(all_fields):
            surface_sets = [field in s for s in expected_by_surface.values()]
            grade_sets = [field in s for s in expected_by_grade.values()]
            venue_sets = [field in s for s in expected_by_venue.values()]

            varies_surface = len(set(surface_sets)) > 1
            varies_grade = len(set(grade_sets)) > 1
            varies_venue = len(set(venue_sets)) > 1

            if varies_surface and field not in summary["fields_varying_by_surface"]:
                summary["fields_varying_by_surface"].append(field)
            if varies_grade and field not in summary["fields_varying_by_grade"]:
                summary["fields_varying_by_grade"].append(field)
            if varies_venue and field not in summary["fields_varying_by_venue"]:
                summary["fields_varying_by_venue"].append(field)
            if not varies_surface and not varies_grade and not varies_venue:
                if field not in summary["fields_uniform"]:
                    summary["fields_uniform"].append(field)

    return summary


def sample_year(year: str, *, samples_per_combo: int = 3,
                seed: int = 42,
                storage: Any = None) -> dict:
    """1年分のスキーマサンプリングを実行する。"""
    from scraper.storage import HybridStorage

    if storage is None:
        storage = HybridStorage()

    rng = random.Random(seed)
    t0 = time.time()

    # Phase 1: race_shutuba をスキャンして分類マッピングを構築
    logger.info("[%s] race_shutuba スキャン開始 ...", year)
    blobs = storage.batch_list_blobs("race_shutuba", year)
    jra_keys = [k for k in blobs if is_jra_race(k)]
    logger.info("[%s] race_shutuba: %d JRA races (全%d)", year, len(jra_keys), len(blobs))

    combo_map: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    classify_errors = 0

    for key in jra_keys:
        data = storage.load("race_shutuba", key)
        if not data:
            continue
        combo = _classify_race(data)
        if combo:
            combo_map[combo].append(key)
        else:
            classify_errors += 1

    logger.info("[%s] 分類完了: %d 組み合わせ, %d races, %d 分類エラー",
                year, len(combo_map), sum(len(v) for v in combo_map.values()),
                classify_errors)

    # Phase 2: サンプリング
    sampled_race_ids: set[str] = set()
    combo_samples: dict[str, list[str]] = {}

    for combo, race_ids in sorted(combo_map.items()):
        combo_key = f"({combo[0]}, {combo[1]}, {combo[2]})"
        n = min(samples_per_combo, len(race_ids))
        selected = rng.sample(race_ids, n)
        combo_samples[combo_key] = selected
        sampled_race_ids.update(selected)

    logger.info("[%s] サンプリング: %d 組み合わせ → %d unique races",
                year, len(combo_samples), len(sampled_race_ids))

    # Phase 3: サンプルレースの全カテゴリデータをロードし分析
    schemas: dict[str, dict] = {}
    load_errors = 0
    loaded_count = 0

    for cat in SAMPLE_CATEGORIES:
        cat_schemas: dict[str, dict] = {}

        for combo_key, race_ids in combo_samples.items():
            analyses = []
            for rid in race_ids:
                data = storage.load(cat, rid)
                loaded_count += 1
                if data:
                    analyses.append(_analyze_data(data))
                elif cat in ("race_barometer", "smartrc_race"):
                    pass  # era-dependent missing is expected
                else:
                    load_errors += 1

            if analyses:
                cat_schemas[combo_key] = _merge_field_stats(analyses)
            else:
                cat_schemas[combo_key] = {
                    "sample_count": 0,
                    "top_fields": {},
                    "entry_fields": {},
                    "note": f"no data found for {cat}",
                }

        schemas[cat] = cat_schemas

    # Phase 4: 1件のみの組み合わせにフォールバック適用
    for cat, cat_schemas in schemas.items():
        for combo_key, schema in cat_schemas.items():
            if schema.get("sample_count", 0) <= 1:
                parts = combo_key.strip("()").split(", ")
                if len(parts) != 3:
                    continue
                grade, surface, _ = parts
                fallback_analyses = []
                for other_key, other_schema in cat_schemas.items():
                    if other_key == combo_key:
                        continue
                    other_parts = other_key.strip("()").split(", ")
                    if len(other_parts) == 3 and other_parts[0] == grade and other_parts[1] == surface:
                        if other_schema.get("sample_count", 0) >= 2:
                            fallback_analyses.append(other_schema)

                if fallback_analyses:
                    schema["_fallback_from"] = [
                        k for k, s in cat_schemas.items()
                        if k != combo_key
                        and k.strip("()").split(", ")[:2] == [grade, surface]
                        and s.get("sample_count", 0) >= 2
                    ][:3]

    elapsed = time.time() - t0
    summary = _build_summary(schemas)

    result = {
        "year": year,
        "generated_at": datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S JST"),
        "seed": seed,
        "combinations_total": len(combo_map),
        "combinations_sampled": len(combo_samples),
        "races_sampled": len(sampled_race_ids),
        "loads_performed": loaded_count,
        "load_errors": load_errors,
        "elapsed_seconds": round(elapsed, 1),
        "combination_sizes": {
            f"({k[0]}, {k[1]}, {k[2]})": len(v)
            for k, v in sorted(combo_map.items())
        },
        "schemas": schemas,
        "summary": summary,
    }

    return result


def main() -> int:
    script_basic_config()
    ap = argparse.ArgumentParser(description="スキーマサンプリング")
    ap.add_argument("--years", required=True,
                    help="対象年 (カンマ区切り: 2022,2023,2024)")
    ap.add_argument("--samples", type=int, default=3,
                    help="組み合わせ毎のサンプル数 (default: 3)")
    ap.add_argument("--seed", type=int, default=42, help="乱数シード")
    ap.add_argument("--json-out", help="出力先パス (省略時: data/meta/schema_reference_{year}.json)")
    ap.add_argument("--export-tables", action="store_true",
                    help="スキーマサンプリング後にテーブルエクスポートも実行")
    ap.add_argument("--workers", type=int, default=6,
                    help="テーブルエクスポート時の並列数")
    args = ap.parse_args()

    years = [y.strip() for y in args.years.split(",")]

    from scraper.storage import HybridStorage
    storage = HybridStorage()

    for year in years:
        logger.info("=" * 60)
        logger.info("スキーマサンプリング: %s年", year)
        logger.info("=" * 60)

        result = sample_year(
            year, samples_per_combo=args.samples,
            seed=args.seed, storage=storage,
        )

        out_path = args.json_out or f"data/meta/schema_reference_{year}.json"
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(result, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

        print(f"\n{'='*60}")
        print(f"  {year}年 スキーマサンプリング結果")
        print(f"{'='*60}")
        print(f"  組み合わせ数: {result['combinations_total']}")
        print(f"  サンプルレース: {result['races_sampled']}")
        print(f"  GCS ロード: {result['loads_performed']}")
        print(f"  ロードエラー: {result['load_errors']}")
        print(f"  所要時間: {result['elapsed_seconds']}s")
        print(f"  出力: {out_path}")
        print()

        ns = len(result["summary"]["fields_varying_by_surface"])
        ng = len(result["summary"]["fields_varying_by_grade"])
        nv = len(result["summary"]["fields_varying_by_venue"])
        nu = len(result["summary"]["fields_uniform"])
        print(f"  サーフェス差異: {ns} fields")
        print(f"  グレード差異: {ng} fields")
        print(f"  場所差異: {nv} fields")
        print(f"  共通: {nu} fields")
        print()

    if args.export_tables:
        logger.info("テーブルエクスポートを開始 ...")
        from scraper.export_tables import export_year
        for year in years:
            export_year(year, storage=storage, workers=args.workers)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
