"""
既存の GCS canonical JSON から、要件表の行固有カテゴリに分割・再格納する。

新カテゴリ（11種）:
  race_shutuba_meta       ← race_shutuba
  race_result_on_time_payoff ← race_result_on_time
  race_result_on_time_lap    ← race_result_on_time
  race_result_on_time_corner ← race_result_on_time
  horse_profile           ← horse_result
  horse_race_history      ← horse_result
  race_result_meta        ← race_result
  race_result_payoff      ← race_result
  race_result_track       ← race_result
  race_result_corner      ← race_result_lap
  race_result_lap_times   ← race_result_lap

使い方:
  # ドライラン（書き込まず件数だけ確認）
  python3 -m src.scripts.scraping.migrate_row_data_to_unique_paths --year 2026 --dry-run

  # 2026 年テスト（race カテゴリのみ）
  python3 -m src.scripts.scraping.migrate_row_data_to_unique_paths --year 2026

  # 2026 年テスト（horse カテゴリを含む）
  python3 -m src.scripts.scraping.migrate_row_data_to_unique_paths --year 2026 --include-horses

  # 2020–2026 本番（race + horse）
  python3 -m src.scripts.scraping.migrate_row_data_to_unique_paths \\
      --year-start 2020 --year-end 2026 --include-horses

  # 特定ソースカテゴリのみ
  python3 -m src.scripts.scraping.migrate_row_data_to_unique_paths \\
      --year 2026 --source race_result

  # 失敗済みキーを再試行（--skip-existing 無効化）
  python3 -m src.scripts.scraping.migrate_row_data_to_unique_paths \\
      --year 2026 --no-skip-existing
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

# ---- KEIBA_SCHEMA_STRICT=0 で派生 JSON の保存を許可（null フィールドが多い場合でも通す）
os.environ.setdefault("KEIBA_SCHEMA_STRICT", "0")

from src.scraper.storage import HybridStorage
from src.scraper.row_data_extractor import DERIVED_CATEGORY_MAP

logger = logging.getLogger("migrate_row_data")

# ソースカテゴリ → 派生カテゴリ一覧
_SOURCE_TO_TARGETS: dict[str, list[str]] = defaultdict(list)
for _tgt, (_src, _fn) in DERIVED_CATEGORY_MAP.items():
    _SOURCE_TO_TARGETS[_src].append(_tgt)


# ---------------------------------------------------------------------------- #
# コアマイグレーションロジック
# ---------------------------------------------------------------------------- #

def _migrate_one_key(
    storage: HybridStorage,
    source_category: str,
    key: str,
    *,
    dry_run: bool,
    skip_existing: bool,
) -> dict[str, Any]:
    """1キーを対象にすべての派生カテゴリへ書き込む。統計 dict を返す。"""
    stats: dict[str, Any] = {
        "key": key,
        "source": source_category,
        "targets": {},
    }

    # ソース JSON 読み込み
    try:
        data = storage.load(source_category, key)
    except Exception as e:
        stats["error"] = f"load failed: {e}"
        return stats
    if not isinstance(data, dict):
        stats["error"] = "load returned non-dict"
        return stats

    targets = _SOURCE_TO_TARGETS[source_category]
    for target_category in targets:
        _src, extract_fn = DERIVED_CATEGORY_MAP[target_category]
        if skip_existing:
            try:
                if storage.exists(target_category, key):
                    stats["targets"][target_category] = "skipped"
                    continue
            except Exception:
                pass  # exists チェック失敗時はそのまま保存試行

        try:
            derived = extract_fn(data)
        except Exception as e:
            stats["targets"][target_category] = f"extract_error: {e}"
            continue

        if not dry_run:
            try:
                storage.save(target_category, key, derived)
                stats["targets"][target_category] = "saved"
            except Exception as e:
                stats["targets"][target_category] = f"save_error: {e}"
        else:
            stats["targets"][target_category] = "dry_run"

    return stats


def migrate_race_category(
    storage: HybridStorage,
    source_category: str,
    years: list[str],
    *,
    dry_run: bool,
    skip_existing: bool,
    max_workers: int = 8,
) -> dict[str, int]:
    """レース系カテゴリを年別に処理する。"""
    counters: dict[str, int] = defaultdict(int)
    targets = _SOURCE_TO_TARGETS.get(source_category, [])
    if not targets:
        return counters

    logger.info("[%s] 開始: years=%s targets=%s", source_category, years, targets)
    for year in years:
        keys = storage.list_keys(source_category, year)
        if not keys:
            logger.info("[%s] %s: キーなし", source_category, year)
            continue
        logger.info("[%s] %s: %d キー処理中...", source_category, year, len(keys))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futs = {
                executor.submit(
                    _migrate_one_key,
                    storage,
                    source_category,
                    key,
                    dry_run=dry_run,
                    skip_existing=skip_existing,
                ): key
                for key in keys
            }
            for fut in as_completed(futs):
                result = fut.result()
                if "error" in result:
                    counters["error"] += 1
                    logger.warning(
                        "[%s] %s: %s", source_category, result["key"], result["error"]
                    )
                else:
                    for tgt, status in result["targets"].items():
                        counters[status] += 1
                        if "error" in status:
                            logger.warning(
                                "[%s→%s] %s: %s",
                                source_category,
                                tgt,
                                result["key"],
                                status,
                            )
    return counters


def migrate_horse_category(
    storage: HybridStorage,
    source_category: str,
    *,
    dry_run: bool,
    skip_existing: bool,
    max_workers: int = 8,
) -> dict[str, int]:
    """馬系カテゴリを全キーで処理する（年フィルタなし）。"""
    counters: dict[str, int] = defaultdict(int)
    targets = _SOURCE_TO_TARGETS.get(source_category, [])
    if not targets:
        return counters

    keys = storage.list_keys(source_category)
    if not keys:
        logger.info("[%s] キーなし", source_category)
        return counters
    logger.info("[%s] %d キー処理中...", source_category, len(keys))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futs = {
            executor.submit(
                _migrate_one_key,
                storage,
                source_category,
                key,
                dry_run=dry_run,
                skip_existing=skip_existing,
            ): key
            for key in keys
        }
        for fut in as_completed(futs):
            result = fut.result()
            if "error" in result:
                counters["error"] += 1
                logger.warning("[%s] %s: %s", source_category, result["key"], result["error"])
            else:
                for tgt, status in result["targets"].items():
                    counters[status] += 1
                    if "error" in status:
                        logger.warning(
                            "[%s→%s] %s: %s",
                            source_category,
                            result["key"].split("/")[-1],
                            tgt,
                            status,
                        )
    return counters


# ---------------------------------------------------------------------------- #
# メインエントリ
# ---------------------------------------------------------------------------- #

_RACE_SOURCE_CATEGORIES = [
    "race_shutuba",
    "race_result_on_time",
    "race_result",
    "race_result_lap",
]

_HORSE_SOURCE_CATEGORIES = [
    "horse_result",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="canonical JSON を行固有カテゴリに分割して GCS に再格納する",
    )
    ap.add_argument("--year", help="単一年 (例: 2026)")
    ap.add_argument("--year-start", help="開始年 (例: 2020)")
    ap.add_argument("--year-end", help="終了年 (例: 2026)")
    ap.add_argument(
        "--source",
        help="処理するソースカテゴリ名 (例: race_result)。省略時は全カテゴリ",
    )
    ap.add_argument(
        "--include-horses",
        action="store_true",
        help="馬系カテゴリ (horse_result) も処理する",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="保存せず件数のみ確認",
    )
    ap.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        default=True,
        help="既存キーを上書きする（デフォルトはスキップ）",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=8,
        help="並列ワーカー数 (デフォルト: 8)",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # 年リスト構築
    if args.year:
        years = [args.year]
    elif args.year_start and args.year_end:
        years = [str(y) for y in range(int(args.year_start), int(args.year_end) + 1)]
    elif args.year_start:
        years = [str(y) for y in range(int(args.year_start), 2027)]
    else:
        ap.error("--year または --year-start を指定してください")

    logger.info(
        "migrate 開始: years=%s dry_run=%s skip_existing=%s workers=%d",
        years,
        args.dry_run,
        args.skip_existing,
        args.workers,
    )

    storage = HybridStorage()
    t0 = time.time()
    total: dict[str, int] = defaultdict(int)

    # レース系カテゴリ
    race_sources = _RACE_SOURCE_CATEGORIES
    if args.source:
        race_sources = [args.source] if args.source in _RACE_SOURCE_CATEGORIES else []

    for src in race_sources:
        if not _SOURCE_TO_TARGETS.get(src):
            continue
        counters = migrate_race_category(
            storage,
            src,
            years,
            dry_run=args.dry_run,
            skip_existing=args.skip_existing,
            max_workers=args.workers,
        )
        for k, v in counters.items():
            total[k] += v

    # 馬系カテゴリ
    if args.include_horses and not args.source:
        for src in _HORSE_SOURCE_CATEGORIES:
            if not _SOURCE_TO_TARGETS.get(src):
                continue
            counters = migrate_horse_category(
                storage,
                src,
                dry_run=args.dry_run,
                skip_existing=args.skip_existing,
                max_workers=args.workers,
            )
            for k, v in counters.items():
                total[k] += v
    elif args.include_horses and args.source in _HORSE_SOURCE_CATEGORIES:
        counters = migrate_horse_category(
            storage,
            args.source,
            dry_run=args.dry_run,
            skip_existing=args.skip_existing,
            max_workers=args.workers,
        )
        for k, v in counters.items():
            total[k] += v

    elapsed = time.time() - t0
    logger.info(
        "migrate 完了: elapsed=%.1fs 結果=%s",
        elapsed,
        dict(sorted(total.items())),
    )
    has_errors = total.get("error", 0) > 0
    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
