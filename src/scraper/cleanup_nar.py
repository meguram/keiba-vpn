"""
地方競馬 (NAR) データの GCS クリーンアップ。

race_id[4:6] が JRA 10場コード (01-10) に該当しないレースデータを
全年・全カテゴリから特定し、削除する。

Usage:
    python -m src.scraper.cleanup_nar --dry-run        # 確認のみ
    python -m src.scraper.cleanup_nar --execute         # 実削除
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

from src.utils.keiba_logging import script_basic_config

logger = logging.getLogger(__name__)

JRA_PLACE_CODES = frozenset(f"{i:02d}" for i in range(1, 11))

RACE_CATEGORIES = [
    "race_result", "race_shutuba", "race_index", "race_shutuba_past",
    "race_odds", "race_paddock", "race_barometer", "race_oikiri",
    "race_trainer_comment", "race_result_lap", "race_pair_odds", "smartrc_race",
]

SCAN_YEARS = [str(y) for y in range(2020, 2027)]


def is_jra_race(race_id: str) -> bool:
    return len(race_id) >= 6 and race_id[4:6] in JRA_PLACE_CODES


def run_cleanup(*, execute: bool = False) -> dict:
    from src.scraper.storage import HybridStorage

    storage = HybridStorage()
    t0 = time.time()

    nar_blobs: dict[str, list[str]] = defaultdict(list)
    total_scanned = 0

    for cat in RACE_CATEGORIES:
        for year in SCAN_YEARS:
            blobs = storage.batch_list_blobs(cat, year)
            for key in blobs:
                total_scanned += 1
                if not is_jra_race(key):
                    nar_blobs[cat].append(key)

    total_nar = sum(len(v) for v in nar_blobs.values())
    logger.info("スキャン完了: %d files, NAR=%d files (%.1fs)",
                total_scanned, total_nar, time.time() - t0)

    if not execute:
        logger.info("ドライラン: 削除は実行しません")
    else:
        logger.info("削除開始: %d blobs ...", total_nar)

    deleted = 0
    errors: list[dict] = []
    bucket = storage._get_bucket() if execute and storage.gcs_enabled else None

    for cat, keys in sorted(nar_blobs.items()):
        for key in keys:
            if execute and bucket:
                blob_path = storage._gcs_blob_path(cat, key)
                if not blob_path:
                    continue
                try:
                    blob = bucket.blob(blob_path)
                    blob.delete()
                    deleted += 1
                    if deleted % 50 == 0:
                        logger.info("  deleted %d / %d ...", deleted, total_nar)
                except Exception as e:
                    errors.append({"cat": cat, "key": key, "error": str(e)})
                    logger.warning("削除失敗: %s/%s: %s", cat, key, e)

    # clear blob caches for affected categories
    if execute and deleted > 0:
        for cat in nar_blobs:
            cache_attr = "_blob_cache"
            if hasattr(storage, cache_attr):
                bc = getattr(storage, cache_attr)
                if cat in bc:
                    del bc[cat]

    elapsed = time.time() - t0

    place_codes: dict[str, int] = defaultdict(int)
    for keys in nar_blobs.values():
        for k in keys:
            code = k[4:6] if len(k) >= 6 else "??"
            place_codes[code] += 1

    report = {
        "mode": "execute" if execute else "dry_run",
        "scanned": total_scanned,
        "nar_total": total_nar,
        "deleted": deleted if execute else 0,
        "errors": errors,
        "elapsed_seconds": round(elapsed, 1),
        "by_category": {cat: len(keys) for cat, keys in sorted(nar_blobs.items())},
        "by_place_code": dict(sorted(place_codes.items(), key=lambda x: -x[1])),
        "sample_keys": {
            cat: keys[:10] for cat, keys in sorted(nar_blobs.items()) if keys
        },
    }

    return report


def main() -> int:
    script_basic_config()
    ap = argparse.ArgumentParser(description="地方競馬(NAR)データ削除")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="削除せず確認のみ")
    group.add_argument("--execute", action="store_true", help="実際に削除する")
    ap.add_argument("--report-out", default="data/meta/nar_cleanup_report.json")
    args = ap.parse_args()

    report = run_cleanup(execute=args.execute)

    print(f"\n{'='*60}")
    print(f"NAR データクリーンアップ {'【実行】' if args.execute else '【ドライラン】'}")
    print(f"{'='*60}")
    print(f"スキャン: {report['scanned']:,} files")
    print(f"NAR検出: {report['nar_total']:,} files")
    if args.execute:
        print(f"削除済み: {report['deleted']:,} files")
        if report["errors"]:
            print(f"エラー: {len(report['errors'])} 件")
    print(f"所要時間: {report['elapsed_seconds']}s")
    print()

    print("【場所コード別】")
    for code, cnt in report["by_place_code"].items():
        print(f"  {code}: {cnt} files")
    print()

    print("【カテゴリ別】")
    for cat, cnt in report["by_category"].items():
        print(f"  {cat}: {cnt} files")
    print()

    if report.get("sample_keys"):
        print("【サンプル race_id】")
        for cat, keys in report["sample_keys"].items():
            print(f"  {cat}: {keys[:5]}")
        print()

    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
