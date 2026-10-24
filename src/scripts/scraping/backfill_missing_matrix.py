"""
GCS データマトリクスの欠損（✗）を一括バックフィルするスクリプト。

対象: cutoff 日（既定: 今日 -5 日）以前の全開催日で、
      直接スクレイプ可能カテゴリの GCS データが存在しないレース。

使い方:
  # 欠損数を確認するだけ（キューに投入しない）
  python3 -m src.scripts.scraping.backfill_missing_matrix --dry-run

  # 5日前以前の全欠損をキューに投入
  python3 -m src.scripts.scraping.backfill_missing_matrix

  # 特定カテゴリのみ
  python3 -m src.scripts.scraping.backfill_missing_matrix --category race_barometer

  # days-ago を変更（例: 7日前以前）
  python3 -m src.scripts.scraping.backfill_missing_matrix --days-ago 7

派生カテゴリ（race_shutuba_meta 等）の生成は scraping/migrate_row_data_to_unique_paths
を別途実行してください。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backfill_missing_matrix")

# 直接スクレイプ可能カテゴリ → スクレイピングタスク名
DIRECT_CATEGORY_TASK: dict[str, str] = {
    "race_shutuba":      "race_shutuba",
    "race_index":        "race_index",
    "race_paddock":      "race_paddock",
    "race_odds":         "race_odds",
    "race_result_on_time": "race_result_on_time",
    "race_result":       "race_result",
    "race_result_lap":   "race_result_lap",
    "race_barometer":    "race_barometer",
}


def _load_env() -> None:
    env_path = Path(__file__).resolve().parents[3] / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _collect_target_dates(
    cov_root: Path, cutoff_date
) -> dict[str, list[str]]:
    """cutoff_date 以前の date → race_ids を収集する。"""
    date_race_ids: dict[str, list[str]] = {}
    for year_dir in sorted(cov_root.iterdir()):
        if not year_dir.is_dir():
            continue
        for f in sorted(year_dir.glob("*.json")):
            date_str = f.stem
            if not (len(date_str) == 8 and date_str.isdigit()):
                continue
            d = datetime.strptime(date_str, "%Y%m%d").date()
            if d <= cutoff_date:
                try:
                    cov = json.loads(f.read_text())
                    ids = cov.get("race_ids", [])
                    if ids:
                        date_race_ids[date_str] = ids
                except Exception as e:
                    logger.warning("coverage 読み込み失敗 [%s]: %s", date_str, e)
    return date_race_ids


def _build_missing_jobs(
    storage,
    date_race_ids: dict[str, list[str]],
    target_cats: list[str],
) -> dict[str, set[str]]:
    """race_id → 欠損スクレイプタスク名 set を返す。"""
    years = sorted({d[:4] for d in date_race_ids})
    logger.info("GCS blob 一覧取得: %s × %s", target_cats, years)

    cat_year_keys: dict[str, dict[str, set[str]]] = defaultdict(dict)
    for cat in target_cats:
        for yr in years:
            try:
                cat_year_keys[cat][yr] = set(storage.batch_list_blobs(cat, yr).keys())
            except Exception as e:
                logger.warning("batch_list_blobs 失敗 [%s/%s]: %s", cat, yr, e)
                cat_year_keys[cat][yr] = set()

    # race_id → 欠損タスク set
    rid_to_tasks: dict[str, set[str]] = defaultdict(set)
    stats: dict[str, int] = defaultdict(int)

    for date_str, race_ids in sorted(date_race_ids.items()):
        yr = date_str[:4]
        for cat in target_cats:
            existing = cat_year_keys[cat].get(yr, set())
            task = DIRECT_CATEGORY_TASK[cat]
            for rid in race_ids:
                if rid not in existing:
                    rid_to_tasks[rid].add(task)
                    stats[cat] += 1

    logger.info("欠損集計完了: %s", dict(stats))
    return rid_to_tasks, stats


def run(
    days_ago: int = 5,
    category_filter: str | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    _load_env()

    from src.scraper.storage import HybridStorage
    from src.scraper.job_queue import ScrapeJobQueue

    storage = HybridStorage()
    jst = timezone(timedelta(hours=9))
    today = datetime.now(jst).date()
    cutoff = today - timedelta(days=days_ago)

    logger.info("バックフィル開始: cutoff=%s (今日--%d日), dry_run=%s", cutoff, days_ago, dry_run)

    cov_root = Path("data/local/meta/date_coverage")
    date_race_ids = _collect_target_dates(cov_root, cutoff)
    if not date_race_ids:
        logger.warning("対象日なし")
        return {"status": "no_target_dates"}

    total_races = sum(len(v) for v in date_race_ids.values())
    logger.info("対象日数: %d, 合計レース数: %d", len(date_race_ids), total_races)

    # カテゴリフィルタ
    target_cats = (
        [category_filter] if category_filter and category_filter in DIRECT_CATEGORY_TASK
        else list(DIRECT_CATEGORY_TASK.keys())
    )

    rid_to_tasks, cat_stats = _build_missing_jobs(storage, date_race_ids, target_cats)

    total_missing_cells = sum(cat_stats.values())
    unique_races = len(rid_to_tasks)
    logger.info(
        "欠損: %d セル / %d レース (一覧: %s)",
        total_missing_cells, unique_races,
        {k: v for k, v in cat_stats.items() if v > 0},
    )

    if dry_run or not rid_to_tasks:
        logger.info("dry_run=True または欠損なし → キューへの投入をスキップ")
        return {
            "status": "dry_run",
            "missing_cells": total_missing_cells,
            "unique_races_with_missing": unique_races,
            "by_category": dict(cat_stats),
        }

    # バルクキューイング
    queue = ScrapeJobQueue()
    job_specs = []
    for rid, tasks in sorted(rid_to_tasks.items()):
        job_specs.append({
            "job_kind": "race",
            "target_id": rid,
            "tasks": sorted(tasks),
            "overwrite": force,
            "smart_skip": not force,
        })

    logger.info("キューに %d ジョブを投入します...", len(job_specs))
    # bulk_add_jobs が無ければ個別投入
    added = 0
    skipped = 0
    if hasattr(queue, "bulk_add_jobs"):
        result = queue.bulk_add_jobs(job_specs)
        added = result.get("added", len(job_specs))
        skipped = result.get("skipped", 0)
    else:
        for spec in job_specs:
            r = queue.add_job(spec)
            action = r.get("action", r.get("status", ""))
            if action in ("added", "queued"):
                added += 1
            else:
                skipped += 1
            if added % 500 == 0 and added > 0:
                logger.info("  投入中... %d / %d", added, len(job_specs))

    logger.info("投入完了: added=%d, skipped=%d", added, skipped)

    # キューワーカー起動
    try:
        from src.scraper.job_queue import ScrapeJobQueueWorker  # noqa
        logger.info("キューワーカーは API サーバー経由で処理されます (/api/scrape-queue/resume)")
    except Exception:
        pass

    return {
        "status": "enqueued",
        "missing_cells": total_missing_cells,
        "unique_races_with_missing": unique_races,
        "by_category": dict(cat_stats),
        "jobs_added": added,
        "jobs_skipped": skipped,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCS マトリクス欠損のバックフィル")
    parser.add_argument("--days-ago", type=int, default=5,
                        help="何日前以前を対象にするか (既定: 5)")
    parser.add_argument("--category", type=str, default=None,
                        help="特定カテゴリのみ (例: race_barometer)")
    parser.add_argument("--dry-run", action="store_true",
                        help="欠損数を表示するだけでキューに投入しない")
    parser.add_argument("--force", action="store_true",
                        help="既存データを上書きして再取得")
    args = parser.parse_args()

    result = run(
        days_ago=args.days_ago,
        category_filter=args.category,
        dry_run=args.dry_run,
        force=args.force,
    )
    import json as _json
    print(_json.dumps(result, ensure_ascii=False, indent=2))
