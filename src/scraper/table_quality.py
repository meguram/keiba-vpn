"""
テーブル品質管理モジュール。

年×カテゴリ単位でローカル Parquet テーブルの品質を検証し、
ステータスファイル (_table_quality.json) で管理する。

OOM 防止: 1カテゴリ処理ごとに storage キャッシュ解放 + GC。

Usage:
    # 品質チェックのみ
    python -m src.scraper.table_quality --years 2020,2021,2022,2023,2024,2025

    # チェック + 不合格テーブルの再作成
    python -m src.scraper.table_quality --years 2020,2021,2022,2023,2024,2025 --rebuild

    # 全テーブル強制再作成
    python -m src.scraper.table_quality --years 2020,2021,2022,2023,2024,2025 --rebuild --force
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from src.utils.keiba_logging import script_basic_config

logger = logging.getLogger(__name__)

QUALITY_FILE = Path("data/page_reference/tables/_table_quality.json")
TABLES_DIR = Path("data/page_reference/tables")

SCHEMA_SAMPLE_SIZE = 30
ENTRY_ISSUE_TOLERANCE = 0.15
CATEGORY_ENTRY_ISSUE_TOLERANCE = {
    "race_oikiri": 0.50,
}

CATEGORIES = [
    "race_result", "race_shutuba", "race_index", "race_shutuba_past",
    "race_odds", "race_paddock", "race_barometer", "race_oikiri",
    "race_trainer_comment", "race_result_lap", "smartrc_race",
    "race_pair_odds",
]

COMPLETENESS_EXEMPT = frozenset({
    "race_barometer", "race_paddock", "race_trainer_comment",
})

MIN_AVG_ENTRIES = {
    "race_result": 10.0,
    "race_shutuba": 10.0,
    "race_shutuba_past": 10.0,
    "race_odds": 10.0,
    "race_result_lap": 10.0,
    "race_oikiri": 10.0,
    "race_index": 3.0,
    "race_pair_odds": 100.0,
    "smartrc_race": 10.0,
    "race_paddock": 0.5,
    "race_trainer_comment": 0.5,
    "race_barometer": 0.5,
}


# ── ステータスファイル I/O ──

def load_quality_status() -> dict[str, Any]:
    try:
        if QUALITY_FILE.exists():
            return json.loads(QUALITY_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"version": 1, "updated_at": None, "tables": {}}


def save_quality_status(status: dict[str, Any]) -> None:
    status["updated_at"] = datetime.now(timezone.utc).isoformat()
    QUALITY_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = QUALITY_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(QUALITY_FILE)


# ── メモリ監視 ──

def _log_memory(phase: str, year: str, cat: str) -> None:
    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()
        mem = {}
        for line in lines[:5]:
            parts = line.split()
            mem[parts[0].rstrip(":")] = int(parts[1]) // 1024
        used = mem.get("MemTotal", 0) - mem.get("MemAvailable", 0)
        total = mem.get("MemTotal", 0)
        logger.info(
            "[%s/%s] MEM %s: %dMB / %dMB (%.0f%%)",
            year, cat, phase, used, total, used / max(total, 1) * 100,
        )
    except Exception:
        pass


def _gc_and_clear(storage: Any) -> None:
    if hasattr(storage, "_load_cache"):
        with storage._load_cache_lock:
            storage._load_cache.clear()
    gc.collect()


# ── Parquet ファイル検査 ──

def _check_parquet_file(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"exists": False}
    if not path.exists():
        return info
    try:
        pf = pq.ParquetFile(path)
        info["exists"] = True
        info["num_rows"] = pf.metadata.num_rows
        info["num_columns"] = pf.metadata.num_columns
        info["size_mb"] = round(path.stat().st_size / (1024 * 1024), 2)
        info["mtime"] = datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat()
    except Exception as e:
        info["error"] = str(e)
    return info


def _count_unique_races(path: Path) -> int:
    try:
        tbl = pq.read_table(path, columns=["race_id"])
        n = tbl.column("race_id").to_pandas().nunique()
        del tbl
        return n
    except Exception:
        return 0


# ── スキーマ検証 ──

def _is_schema_soft_pass(category: str, vr: dict[str, Any]) -> bool:
    """top-level OK でエントリの問題が軽微なら soft pass。"""
    if vr.get("top_missing") or vr.get("top_type_errors") or vr.get("top_constraint_errors"):
        return False
    entry_count = vr.get("entry_count", 0)
    if entry_count == 0:
        return True
    entry_issues = vr.get("entry_issues", {})
    total_issues = 0
    for kind_counts in entry_issues.values():
        if isinstance(kind_counts, dict):
            total_issues += sum(kind_counts.values())
    tolerance = CATEGORY_ENTRY_ISSUE_TOLERANCE.get(
        category, ENTRY_ISSUE_TOLERANCE
    )
    return total_issues / max(entry_count, 1) <= tolerance


def _schema_sample_check(
    year: str, category: str, storage: Any,
    sample_size: int = SCHEMA_SAMPLE_SIZE,
) -> dict[str, Any]:
    from src.scraper.schemas import validate as validate_schema
    from src.scraper.export_tables import is_jra_race

    blobs = storage.batch_list_blobs(category, year)
    jra_keys = [k for k in blobs if is_jra_race(k)]
    n_total = len(jra_keys)

    result: dict[str, Any] = {
        "gcs_total": n_total, "sample_size": 0,
        "passed": 0, "soft_passed": 0, "failed": 0, "errors": [],
    }
    if n_total == 0:
        return result

    sample_keys = random.sample(jra_keys, min(sample_size, n_total))
    result["sample_size"] = len(sample_keys)

    for key in sample_keys:
        data = storage.load(category, key)
        if not data:
            result["errors"].append({"key": key, "error": "load_failed"})
            result["failed"] += 1
            continue

        vr = validate_schema(category, data)
        if vr.get("passed"):
            result["passed"] += 1
        elif _is_schema_soft_pass(category, vr):
            result["soft_passed"] += 1
        else:
            result["failed"] += 1
            if len(result["errors"]) < 5:
                result["errors"].append({
                    "key": key,
                    "top_missing": vr.get("top_missing", []),
                    "entry_issues": bool(vr.get("entry_issues")),
                })

    _gc_and_clear(storage)
    return result


# ── カテゴリ品質チェック ──

def check_table_quality(
    year: str, category: str, storage: Any,
    *, skip_schema: bool = False,
) -> dict[str, Any]:
    from src.scraper.export_tables import is_jra_race

    report: dict[str, Any] = {
        "category": category, "year": year, "status": "pending",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "issues": [],
    }

    flat_path = TABLES_DIR / year / f"{category}_flat.parquet"
    race_path = TABLES_DIR / year / f"{category}_race.parquet"

    flat_info = _check_parquet_file(flat_path)
    race_info = _check_parquet_file(race_path)
    report["flat"] = flat_info
    report["race"] = race_info

    if not flat_info["exists"]:
        report["issues"].append("flat_parquet_missing")
        report["status"] = "fail"
        return report

    if not race_info["exists"]:
        report["issues"].append("race_parquet_missing")

    parquet_races = _count_unique_races(flat_path)
    report["parquet_races"] = parquet_races

    blobs = storage.batch_list_blobs(category, year)
    jra_keys = [k for k in blobs if is_jra_race(k)]
    gcs_races = len(jra_keys)
    report["gcs_races"] = gcs_races

    if gcs_races == 0:
        if category in COMPLETENESS_EXEMPT:
            report["status"] = "pass"
            report["note"] = "exempt category, no GCS data"
            return report
        report["issues"].append("no_gcs_data")
        report["status"] = "fail"
        return report

    race_coverage = parquet_races / gcs_races if gcs_races > 0 else 0
    report["race_coverage"] = round(race_coverage, 4)
    if race_coverage < 0.95:
        report["issues"].append(
            f"race_coverage_low: {parquet_races}/{gcs_races} = {race_coverage:.1%}"
        )

    flat_rows = flat_info.get("num_rows", 0)
    if parquet_races > 0:
        avg_entries = flat_rows / parquet_races
        report["avg_entries"] = round(avg_entries, 1)
        min_avg = MIN_AVG_ENTRIES.get(category, 1.0)
        if avg_entries < min_avg and category not in COMPLETENESS_EXEMPT:
            report["issues"].append(f"avg_entries_low: {avg_entries:.1f} < {min_avg}")

    if not skip_schema:
        schema_result = _schema_sample_check(year, category, storage)
        report["schema_check"] = {
            "gcs_total": schema_result["gcs_total"],
            "sample_size": schema_result["sample_size"],
            "strict_passed": schema_result["passed"],
            "soft_passed": schema_result["soft_passed"],
            "failed": schema_result["failed"],
        }
        if schema_result["sample_size"] > 0:
            total_pass = schema_result["passed"] + schema_result["soft_passed"]
            pass_rate = total_pass / schema_result["sample_size"]
            report["schema_pass_rate"] = round(pass_rate, 3)
            if pass_rate < 0.8:
                report["issues"].append(f"schema_pass_rate_low: {pass_rate:.1%}")
                if schema_result["errors"]:
                    report["schema_errors_sample"] = schema_result["errors"][:3]

    report["status"] = "pass" if not report["issues"] else "fail"
    return report


# ── メイン処理 ──

def run_quality_check(
    years: list[str],
    *, force: bool = False, skip_schema: bool = False,
) -> dict[str, Any]:
    from src.scraper.storage import HybridStorage
    storage = HybridStorage()
    status = load_quality_status()

    if force:
        for y in years:
            status["tables"].pop(y, None)

    results: dict[str, list[dict]] = {}

    for year in years:
        year_results = []
        year_status = status["tables"].setdefault(year, {})

        for cat in CATEGORIES:
            if not force and cat in year_status:
                existing = year_status[cat]
                if existing.get("status") == "pass":
                    logger.info("[%s/%s] SKIP (already pass)", year, cat)
                    year_results.append(existing)
                    continue

            logger.info("[%s/%s] 品質チェック開始...", year, cat)
            t0 = time.time()
            report = check_table_quality(
                year, cat, storage, skip_schema=skip_schema,
            )
            report["check_elapsed_s"] = round(time.time() - t0, 1)
            year_status[cat] = report
            year_results.append(report)

            mark = "PASS" if report["status"] == "pass" else "FAIL"
            issues = ", ".join(report.get("issues", [])) or "none"
            logger.info("[%s/%s] %s (%.1fs) issues=[%s]",
                        year, cat, mark, report["check_elapsed_s"], issues)

            _gc_and_clear(storage)

        results[year] = year_results
        save_quality_status(status)

    return results


def rebuild_failed_tables(
    years: list[str],
    *, force: bool = False, skip_schema: bool = False,
) -> dict[str, Any]:
    """品質チェック → 不合格テーブルを1カテゴリずつ再作成 → 再チェック。

    OOM 防止: カテゴリ単位でキャッシュ解放 + GC。
    GCS 最小化: meta_index は年内で1回だけ構築、完了後すぐ解放。
    """
    from src.scraper.storage import HybridStorage
    from src.scraper.export_tables import (
        export_category_chunked, is_jra_race,
        _build_meta_index, _clear_storage_cache,
    )

    storage = HybridStorage()

    check_results = run_quality_check(
        years, force=force, skip_schema=skip_schema,
    )

    status = load_quality_status()
    rebuild_summary: dict[str, list[str]] = {}

    for year in years:
        year_results = check_results.get(year, [])
        cats_to_rebuild = [
            r["category"] for r in year_results if r["status"] != "pass"
        ]

        cats_to_rebuild = [
            c for c in cats_to_rebuild
            if not _is_no_data_exempt(year, c, status)
        ]

        if not cats_to_rebuild:
            logger.info("[%s] 全カテゴリ pass / exempt — 再作成不要", year)
            rebuild_summary[year] = []
            continue

        logger.info("[%s] 再作成対象: %d → %s", year, len(cats_to_rebuild), cats_to_rebuild)
        _log_memory("before_meta", year, "all")

        meta_index = _build_meta_index(year, storage)
        base_blobs = storage.batch_list_blobs("race_result", year)
        expected_keys = frozenset(k for k in base_blobs if is_jra_race(k))
        _clear_storage_cache(storage)
        gc.collect()
        _log_memory("after_meta", year, "all")

        for cat in cats_to_rebuild:
            _log_memory("before", year, cat)

            flat_path = TABLES_DIR / year / f"{cat}_flat.parquet"
            race_path = TABLES_DIR / year / f"{cat}_race.parquet"
            flat_path.unlink(missing_ok=True)
            race_path.unlink(missing_ok=True)

            logger.info("[%s/%s] エクスポート開始...", year, cat)
            export_category_chunked(
                year, cat, storage, meta_index,
                expected_keys=expected_keys,
            )

            _clear_storage_cache(storage)
            gc.collect()

            logger.info("[%s/%s] 再チェック...", year, cat)
            q = check_table_quality(year, cat, storage, skip_schema=skip_schema)
            status["tables"].setdefault(year, {})[cat] = q
            save_quality_status(status)

            mark = "PASS" if q["status"] == "pass" else "FAIL"
            logger.info("[%s/%s] → %s", year, cat, mark)

            _clear_storage_cache(storage)
            gc.collect()
            _log_memory("after", year, cat)

        del meta_index
        gc.collect()
        rebuild_summary[year] = cats_to_rebuild

    return {
        "check_results": {
            y: [{"category": r["category"], "status": r["status"],
                 "issues": r.get("issues", [])} for r in rs]
            for y, rs in check_results.items()
        },
        "rebuilt": rebuild_summary,
    }


def _is_no_data_exempt(year: str, category: str, status: dict) -> bool:
    """GCS にデータがない exempt カテゴリは再作成スキップ → pass 扱い。"""
    if category not in COMPLETENESS_EXEMPT:
        return False
    info = status.get("tables", {}).get(year, {}).get(category, {})
    gcs = info.get("gcs_races", -1)
    if gcs == 0:
        info["status"] = "pass"
        info["note"] = "exempt, no GCS data"
        save_quality_status(status)
        logger.info("[%s/%s] exempt (GCS データなし) → pass", year, category)
        return True
    if "flat_parquet_missing" in info.get("issues", []) and gcs == -1:
        pass
    return False


def print_summary(years: list[str]) -> None:
    status = load_quality_status()
    print(f"\n{'='*70}")
    print(f"  テーブル品質ステータス  (updated: {status.get('updated_at', 'N/A')})")
    print(f"{'='*70}\n")

    for year in years:
        year_data = status.get("tables", {}).get(year, {})
        if not year_data:
            print(f"  {year}: (未チェック)")
            continue

        n_pass = sum(1 for v in year_data.values() if v.get("status") == "pass")
        n_fail = sum(1 for v in year_data.values() if v.get("status") == "fail")
        n_total = len(year_data)
        print(f"  {year}: {n_pass}/{n_total} pass, {n_fail} fail")

        for cat in CATEGORIES:
            info = year_data.get(cat)
            if not info:
                print(f"    {cat:28s}  -")
                continue
            st = info.get("status", "?")
            pq_r = info.get("parquet_races", "?")
            gcs_r = info.get("gcs_races", "?")
            avg = info.get("avg_entries", "?")
            sr = info.get("schema_pass_rate", "?")
            fmb = info.get("flat", {}).get("size_mb", "?")
            mark = "✓" if st == "pass" else "✗"
            issues_str = ", ".join(info.get("issues", []))[:40]
            print(f"    {mark} {cat:26s}  races={pq_r!s:>5s}/{gcs_r!s:<5s}"
                  f"  avg={avg!s:>5s}  schema={sr!s:>5s}"
                  f"  {fmb!s:>6s}MB  {issues_str}")
        print()


def main() -> int:
    script_basic_config()

    ap = argparse.ArgumentParser(description="テーブル品質チェック & 再作成")
    ap.add_argument("--years", required=True,
                    help="対象年 (カンマ区切り)")
    ap.add_argument("--rebuild", action="store_true",
                    help="不合格テーブルを自動で再作成")
    ap.add_argument("--force", action="store_true",
                    help="品質ステータスをリセットして全再チェック")
    ap.add_argument("--skip-schema", action="store_true",
                    help="GCS スキーマサンプルチェックをスキップ")
    ap.add_argument("--summary", action="store_true",
                    help="既存ステータスからサマリー表示のみ")
    args = ap.parse_args()

    years = [y.strip() for y in args.years.split(",")]

    if args.summary:
        print_summary(years)
        return 0

    if args.rebuild:
        rebuild_failed_tables(
            years, force=args.force, skip_schema=args.skip_schema,
        )
    else:
        run_quality_check(
            years, force=args.force, skip_schema=args.skip_schema,
        )

    print_summary(years)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
