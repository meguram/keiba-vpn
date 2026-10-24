"""
データ品質監査 — 統計プロファイルベースの異常検出。

2パス方式:
  Pass 1: 軽量スキャン — 各ファイルの entries 数とフィールド非空率を集計。
  Pass 2: 異常検出 — Pass 1 の統計を基準に、再スキャンして異常を検出。

Usage:
    python -m src.scraper.audit_quality --years 2024
    python -m src.scraper.audit_quality --years 2022,2023,2024 --workers 6
    python -m src.scraper.audit_quality --years 2024 --json-out data/meta/audit_2024.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.utils.keiba_logging import script_basic_config

logger = logging.getLogger(__name__)

# ── カテゴリ定義 ─────────────────────────────────────────

_ENTRY_CATEGORIES: list[tuple[str, str, bool]] = [
    # (category, entries_key, has_field_size)
    ("race_result",          "entries",     True),
    ("race_shutuba",         "entries",     True),
    ("race_index",           "entries",     False),
    ("race_shutuba_past",    "entries",     False),
    ("race_odds",            "entries",     False),
    ("race_paddock",         "entries",     False),
    ("race_barometer",       "entries",     False),
    ("race_oikiri",          "entries",     False),
    ("race_trainer_comment", "entries",     False),
    ("race_result_lap",      "entries_lap", False),
]

_SPECIAL_CATEGORIES = ["race_pair_odds", "smartrc_race"]

ALL_AUDIT_CATS = [c for c, _, _ in _ENTRY_CATEGORIES] + _SPECIAL_CATEGORIES
_ENTRY_CFG = {c: (ek, hfs) for c, ek, hfs in _ENTRY_CATEGORIES}

_FIELD_SIZE_CATS = frozenset({"race_result", "race_shutuba"})

_SKIP_HEADCOUNT_CATS = frozenset({
    "race_paddock",
    "race_trainer_comment",
    "race_oikiri",
    "race_pair_odds",
    "race_index",      # 過去データがある馬のみ表示 → 常に field_size 未満
    "race_barometer",  # 偏差値対象馬のみ → 常に field_size 未満
})


def _is_empty(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    if isinstance(v, (int, float)) and v == 0:
        return True
    if isinstance(v, list) and len(v) == 0:
        return True
    if isinstance(v, dict) and len(v) == 0:
        return True
    return False


# ── Pass 1: lightweight stats ────────────────────────────

def _pass1_one(storage: Any, cat: str, key: str) -> dict:
    """1ファイルの軽量統計を返す。"""
    data = storage.load(cat, key)
    if data is None:
        return {"cat": cat, "key": key, "ok": False}

    result: dict[str, Any] = {"cat": cat, "key": key, "ok": True}

    if cat in _ENTRY_CFG:
        ek, hfs = _ENTRY_CFG[cat]
        entries = data.get(ek) or []
        entries = entries if isinstance(entries, list) else []
        n = len(entries)
        result["n_entries"] = n

        fs = data.get("field_size")
        if hfs and isinstance(fs, (int, float)) and fs > 0:
            result["field_size"] = int(fs)

        # entry field emptiness: {field: empty_count}
        field_empty: dict[str, int] = defaultdict(int)
        field_total: dict[str, int] = defaultdict(int)
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            for k, v in entry.items():
                field_total[k] += 1
                if _is_empty(v):
                    field_empty[k] += 1
        result["entry_field_empty"] = dict(field_empty)
        result["entry_field_total"] = dict(field_total)

        # top-level field emptiness
        top_empty: dict[str, bool] = {}
        for k, v in data.items():
            if k in ("_meta", ek):
                continue
            top_empty[k] = _is_empty(v)
        result["top_field_empty"] = top_empty

    elif cat == "smartrc_race":
        runners = data.get("runners") or []
        n = len(runners) if isinstance(runners, list) else 0
        result["n_entries"] = n
        horses = data.get("horses") or {}
        result["n_horses"] = len(horses) if isinstance(horses, dict) else 0

    elif cat == "race_pair_odds":
        for lk in ("umaren", "wide", "umatan"):
            items = data.get(lk) or []
            result[f"n_{lk}"] = len(items) if isinstance(items, list) else 0

    return result


def _run_pass1(storage: Any, tasks: list[tuple[str, str]], workers: int) -> list[dict]:
    results: list[dict] = []
    done = 0
    total = len(tasks)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_pass1_one, storage, cat, key): (cat, key)
            for cat, key in tasks
        }
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if done % 5000 == 0:
                logger.info("  Pass 1: %d / %d ...", done, total)

    return results


# ── Pass 2: anomaly detection using pass1 stats ──────────

def _build_profiles(pass1_results: list[dict]) -> dict[str, dict]:
    """Pass 1 の結果を集計してプロファイルを構築する。"""
    profiles: dict[str, dict[str, Any]] = {}

    for cat in ALL_AUDIT_CATS:
        cat_results = [r for r in pass1_results if r["cat"] == cat and r["ok"]]
        if not cat_results:
            continue

        p: dict[str, Any] = {"file_count": len(cat_results)}

        if cat in _ENTRY_CFG or cat == "smartrc_race":
            counts = [r.get("n_entries", 0) for r in cat_results]
            if counts:
                sc = sorted(counts)
                p["entry_count_stats"] = {
                    "mean": round(sum(sc) / len(sc), 1),
                    "median": sc[len(sc) // 2],
                    "min": min(sc),
                    "max": max(sc),
                }

        if cat in _ENTRY_CFG:
            # field_size match rate
            with_fs = [r for r in cat_results if "field_size" in r]
            if with_fs:
                matches = sum(
                    1 for r in with_fs if r.get("n_entries") == r.get("field_size")
                )
                p["field_size_match_rate"] = round(matches / len(with_fs), 4)

            # entry field non-empty rates
            agg_total: dict[str, int] = defaultdict(int)
            agg_non_empty: dict[str, int] = defaultdict(int)
            for r in cat_results:
                ft = r.get("entry_field_total", {})
                fe = r.get("entry_field_empty", {})
                for field, total in ft.items():
                    agg_total[field] += total
                    agg_non_empty[field] += total - fe.get(field, 0)

            entry_fields: dict[str, dict] = {}
            for field in sorted(agg_total.keys()):
                t = agg_total[field]
                ne = agg_non_empty[field]
                rate = round(ne / t, 4) if t > 0 else 0.0
                cls = (
                    "normally_populated" if rate >= 0.95
                    else "normally_empty" if rate <= 0.05
                    else "conditional"
                )
                entry_fields[field] = {
                    "non_empty_rate": rate,
                    "classification": cls,
                }
            p["entry_fields"] = entry_fields

            # top-level field non-empty rates
            top_total: dict[str, int] = defaultdict(int)
            top_non_empty: dict[str, int] = defaultdict(int)
            for r in cat_results:
                for field, is_empty in r.get("top_field_empty", {}).items():
                    top_total[field] += 1
                    if not is_empty:
                        top_non_empty[field] += 1

            top_fields: dict[str, dict] = {}
            for field in sorted(top_total.keys()):
                t = top_total[field]
                ne = top_non_empty[field]
                rate = round(ne / t, 4) if t > 0 else 0.0
                cls = (
                    "normally_populated" if rate >= 0.95
                    else "normally_empty" if rate <= 0.05
                    else "conditional"
                )
                top_fields[field] = {
                    "non_empty_rate": rate,
                    "classification": cls,
                }
            p["top_fields"] = top_fields

        profiles[cat] = p

    return profiles


def _detect_anomalies(
    pass1_results: list[dict],
    profiles: dict[str, dict],
    threshold: float,
    max_samples: int,
) -> dict[str, Any]:
    """Pass 1 の統計を基準に異常を検出する。"""

    # headcount reference: race_result > race_shutuba
    headcount_ref: dict[str, int] = {}
    for r in pass1_results:
        if r["cat"] in _FIELD_SIZE_CATS and r.get("ok") and "field_size" in r:
            key = r["key"]
            if key not in headcount_ref:
                headcount_ref[key] = r["field_size"]

    anomalies: list[dict] = []

    for r in pass1_results:
        if not r.get("ok"):
            continue

        cat = r["cat"]
        key = r["key"]
        prof = profiles.get(cat, {})

        if cat in _ENTRY_CFG:
            n_entries = r.get("n_entries", 0)

            # (a) field_size vs entries
            fs = r.get("field_size")
            if fs is not None and n_entries != fs:
                anomalies.append({
                    "cat": cat, "key": key, "kind": "field_size_mismatch",
                    "field_size": fs, "entries": n_entries, "diff": n_entries - fs,
                })

            # (b) headcount reference
            if (
                cat not in _SKIP_HEADCOUNT_CATS
                and cat not in _FIELD_SIZE_CATS
                and n_entries > 0
            ):
                ref = headcount_ref.get(key)
                if ref is not None and abs(n_entries - ref) > 2:
                    anomalies.append({
                        "cat": cat, "key": key, "kind": "headcount_mismatch",
                        "reference": ref, "entries": n_entries,
                        "diff": n_entries - ref,
                    })

            # (c) entry field: all-empty for normally_populated field
            ef_profile = prof.get("entry_fields", {})
            fe = r.get("entry_field_empty", {})
            ft = r.get("entry_field_total", {})
            for field, fp in ef_profile.items():
                if fp["non_empty_rate"] < threshold:
                    continue
                total = ft.get(field, 0)
                empty = fe.get(field, 0)
                if total > 0 and empty == total and n_entries >= 4:
                    anomalies.append({
                        "cat": cat, "key": key, "kind": "field_all_empty",
                        "field": field,
                        "expected_rate": fp["non_empty_rate"],
                        "entries_count": n_entries,
                    })

            # (d) top-level field empty for normally_populated
            tf_profile = prof.get("top_fields", {})
            for field, fp in tf_profile.items():
                if fp["non_empty_rate"] < threshold:
                    continue
                if r.get("top_field_empty", {}).get(field, False):
                    anomalies.append({
                        "cat": cat, "key": key, "kind": "top_field_empty",
                        "field": field,
                        "expected_rate": fp["non_empty_rate"],
                    })

        elif cat == "smartrc_race":
            n = r.get("n_entries", 0)
            ref = headcount_ref.get(key)
            if ref is not None and n > 0 and abs(n - ref) > 2:
                anomalies.append({
                    "cat": cat, "key": key, "kind": "headcount_mismatch",
                    "reference": ref, "runners": n, "diff": n - ref,
                })

    # aggregate
    kind_counts: dict[str, int] = defaultdict(int)
    cat_kind_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for a in anomalies:
        kind_counts[a["kind"]] += 1
        cat_kind_counts[a["cat"]][a["kind"]] += 1

    daily: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "cats": set()}
    )
    for a in anomalies:
        k = a["key"]
        date_part = ""
        if len(k) >= 8 and k[:4].isdigit():
            # race_id YYYYJJRRDDNN -> race_lists date mapping 不可
            # -> race_result の date を使えないので race_id 先頭4桁=年で代替
            # 正確な日付推定: 各開催場・回・日に対応
            pass
        daily_key = k  # just aggregate by race_id prefix
        # skip daily aggregation for now — use cat-level instead

    # sample anomalies (capped per kind)
    samples: dict[str, list[dict]] = defaultdict(list)
    for a in anomalies:
        if len(samples[a["kind"]]) < max_samples:
            samples[a["kind"]].append(a)

    return {
        "total": len(anomalies),
        "by_kind": dict(sorted(kind_counts.items(), key=lambda x: -x[1])),
        "by_category": {
            cat: dict(sorted(kinds.items(), key=lambda x: -x[1]))
            for cat, kinds in sorted(cat_kind_counts.items())
        },
        "samples": {k: v for k, v in sorted(samples.items())},
    }


# ── Main ─────────────────────────────────────────────────

def run_audit(
    years: list[str],
    *,
    workers: int = 6,
    anomaly_threshold: float = 0.95,
    max_anomaly_samples: int = 200,
) -> dict[str, Any]:
    from src.scraper.storage import HybridStorage

    storage = HybridStorage()
    t0 = time.time()

    # blob keys
    tasks: list[tuple[str, str]] = []
    for year in years:
        for cat in ALL_AUDIT_CATS:
            blobs = storage.batch_list_blobs(cat, year)
            for key in sorted(blobs.keys()):
                tasks.append((cat, key))

    total = len(tasks)
    logger.info("監査対象: %d files (%s)", total, ",".join(years))

    # Pass 1
    logger.info("Pass 1: 統計プロファイル収集中...")
    pass1 = _run_pass1(storage, tasks, workers)
    t1 = time.time()
    logger.info("Pass 1 完了: %.1fs", t1 - t0)

    # Build profiles
    profiles = _build_profiles(pass1)

    # Pass 2 (in-memory, no GCS calls)
    logger.info("Pass 2: 異常検出中...")
    anomaly_result = _detect_anomalies(
        pass1, profiles, anomaly_threshold, max_anomaly_samples
    )
    t2 = time.time()
    logger.info("Pass 2 完了: %.1fs, %d anomalies", t2 - t1, anomaly_result["total"])

    return {
        "years": years,
        "files_scanned": total,
        "files_loaded": sum(1 for r in pass1 if r.get("ok")),
        "elapsed_seconds": round(t2 - t0, 1),
        "anomaly_threshold": anomaly_threshold,
        "field_profiles": profiles,
        "anomaly_summary": {
            "total": anomaly_result["total"],
            "by_kind": anomaly_result["by_kind"],
            "by_category": anomaly_result["by_category"],
        },
        "anomaly_samples": anomaly_result["samples"],
    }


def main() -> int:
    script_basic_config()
    ap = argparse.ArgumentParser(description="データ品質監査")
    ap.add_argument("--years", required=True, help="対象年 (カンマ区切り)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    years = [y.strip() for y in args.years.split(",") if y.strip()]
    for y in years:
        if not (y.isdigit() and len(y) == 4):
            print(f"ERROR: invalid year '{y}'", file=sys.stderr)
            return 1

    report = run_audit(
        years,
        workers=max(1, args.workers),
        anomaly_threshold=args.threshold,
    )

    print(f"\n{'='*60}")
    print(f"データ品質監査レポート: {','.join(years)}")
    print(f"{'='*60}")
    print(f"ファイル数: {report['files_scanned']:,} / {report['files_loaded']:,} loaded")
    print(f"所要時間: {report['elapsed_seconds']}s")
    print()

    print("【カテゴリ別プロファイル】")
    for cat, prof in report["field_profiles"].items():
        ec = prof.get("entry_count_stats", {})
        fsr = prof.get("field_size_match_rate")
        line = f"  {cat:30s}: {prof['file_count']:>5} files"
        if ec:
            line += f", entries mean={ec['mean']} [{ec['min']}-{ec['max']}]"
        if fsr is not None:
            line += f", fs_match={fsr}"
        print(line)
    print()

    summary = report["anomaly_summary"]
    print(f"【異常サマリー】 合計 {summary['total']} 件")
    for kind, cnt in summary["by_kind"].items():
        print(f"  {kind}: {cnt}")
    print()

    if summary["by_category"]:
        print("【カテゴリ別異常】")
        for cat, kinds in summary["by_category"].items():
            print(f"  {cat}: {dict(kinds)}")
        print()

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report, ensure_ascii=False, indent=1, default=str),
            encoding="utf-8",
        )
        print(f"Wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
