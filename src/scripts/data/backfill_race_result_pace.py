#!/usr/bin/env python3
"""
race_result の pace 欠損を再スクレイプで補完し、flat Parquet を再エクスポートする。

対象: JRA 芝・ダート（障害レース名は除外）、pace が null または前半3F 欠損

Usage:
  python -m src.scripts.data.backfill_race_result_pace --dry-run
  python -m src.scripts.data.backfill_race_result_pace --years 2020 2021 2022 2023 2024 2025
  python -m src.scripts.data.backfill_race_result_pace --limit 50
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from src.research.race.track_speed_engine import (
    extract_pace_features,
    is_obstacle_race,
    JRA_VENUES,
)
from src.scraper.local_tables import load_flat_df
from src.scraper.pace_utils import merge_race_result_pace, pace_has_first_half
from src.scraper.run import ScraperRunner
from src.scraper.storage import HybridStorage
from src.utils.keiba_logging import script_basic_config

logger = logging.getLogger(__name__)

DEFAULT_PROGRESS_PATH = Path("logs/backfill_race_result_pace_progress.json")


def write_progress(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_missing_race_ids(years: list[str], base_dir: str) -> list[str]:
    rr = load_flat_df(
        "race_result",
        years=years,
        columns=["race_id", "venue", "surface", "pace", "race_name", "grade"],
        base_dir=base_dir,
    )
    if rr.empty:
        return []
    rr = rr[rr["venue"].isin(JRA_VENUES) & rr["surface"].isin(["芝", "ダート"])]
    races = rr.groupby("race_id", as_index=False).first()
    races = races[~races.apply(is_obstacle_race, axis=1)]
    # ジャンプ競走はラップ形式が異なりペース行が無いことが多い
    races = races[
        ~races["race_name"].astype(str).str.contains("ジャンプ|障害", na=False)
    ]

    def _needs(p: object) -> bool:
        if p is None:
            return True
        return extract_pace_features(p).first_half_3f is None

    miss = races[races["pace"].map(_needs)]
    return sorted(miss["race_id"].astype(str).tolist())


def backfill_one(
    runner: ScraperRunner,
    storage: HybridStorage,
    race_id: str,
    dry_run: bool,
) -> str:
    """1レース補完。戻り値: ok | empty | fail"""
    if dry_run:
        return "dry_run"

    result = runner.scrape_race_result(race_id, skip_existing=False)
    lap = runner.scrape_race_result_lap(race_id, skip_existing=False)
    if result is None:
        return "fail"

    merged = merge_race_result_pace(result, lap)
    pace = (merged or {}).get("pace")
    if not pace_has_first_half(pace):
        return "empty"

    storage.save("race_result", race_id, merged)
    if lap is not None:
        lap_pace = dict(lap.get("pace") or {})
        if pace_has_first_half(lap_pace):
            storage.save("race_result_lap", race_id, lap)
    return "ok"


def main() -> None:
    script_basic_config()
    parser = argparse.ArgumentParser(description="race_result pace 欠損バックフィル")
    parser.add_argument(
        "--years",
        nargs="*",
        default=["2020", "2021", "2022", "2023", "2024", "2025"],
    )
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--interval", type=float, default=1.2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--export",
        action="store_true",
        help="補完後に export_tables で flat Parquet を再生成",
    )
    parser.add_argument(
        "--progress-file",
        default=str(DEFAULT_PROGRESS_PATH),
        help="進捗 JSON の出力先（watch_backfill_pace.sh が参照）",
    )
    args = parser.parse_args()

    missing = collect_missing_race_ids(args.years, args.base_dir)
    print(f"pace 欠損レース: {len(missing)} 件 ({', '.join(args.years)})", flush=True)
    if args.limit > 0:
        missing = missing[: args.limit]
        print(f"  → --limit により {len(missing)} 件に制限")

    if args.dry_run:
        for rid in missing[:20]:
            print(f"  {rid}")
        if len(missing) > 20:
            print(f"  ... 他 {len(missing) - 20} 件")
        return

    runner = ScraperRunner(interval=args.interval)
    storage = HybridStorage(args.base_dir)
    stats = {"ok": 0, "empty": 0, "fail": 0}
    progress_path = Path(args.progress_file)
    started_at = time.time()
    write_progress(progress_path, {
        "started_at": started_at,
        "total": len(missing),
        "done": 0,
        "ok": 0,
        "empty": 0,
        "fail": 0,
        "phase": "scrape",
        "years": args.years,
    })

    for i, rid in enumerate(missing, 1):
        try:
            status = backfill_one(runner, storage, rid, dry_run=False)
        except Exception as e:
            logger.exception("backfill failed %s: %s", rid, e)
            status = "fail"
        stats[status] = stats.get(status, 0) + 1
        write_progress(progress_path, {
            "started_at": started_at,
            "total": len(missing),
            "done": i,
            "ok": stats["ok"],
            "empty": stats["empty"],
            "fail": stats["fail"],
            "phase": "scrape",
            "last_race_id": rid,
            "last_status": status,
            "years": args.years,
        })
        if i % 25 == 0 or i == len(missing):
            print(
                f"[{i}/{len(missing)}] ok={stats['ok']} empty={stats['empty']} fail={stats['fail']}",
                flush=True,
            )
        if i < len(missing):
            time.sleep(args.interval)

    print("完了:", stats, flush=True)
    write_progress(progress_path, {
        "started_at": started_at,
        "total": len(missing),
        "done": len(missing),
        **stats,
        "phase": "export" if args.export and stats["ok"] > 0 else "done",
        "finished": not args.export or stats["ok"] == 0,
        "years": args.years,
    })

    if args.export and stats["ok"] > 0:
        from src.scraper.export_tables import export_category_chunked, _build_meta_index

        storage_exp = HybridStorage(args.base_dir)
        for year in args.years:
            meta_index = _build_meta_index(year, storage_exp)
            base_blobs = storage_exp.batch_list_blobs("race_result", year)
            from src.scraper.export_tables import is_jra_race

            expected = frozenset(k for k in base_blobs if is_jra_race(k))
            for cat in ("race_result", "race_result_lap"):
                print(f"export {cat} {year} ...", flush=True)
                write_progress(progress_path, {
                    "started_at": started_at,
                    "total": len(missing),
                    "done": len(missing),
                    **stats,
                    "phase": "export",
                    "export_year": year,
                    "export_category": cat,
                    "years": args.years,
                })
                export_category_chunked(
                    year, cat, storage_exp, meta_index, expected_keys=expected,
                )
        print("flat Parquet 再エクスポート完了", flush=True)
        write_progress(progress_path, {
            "started_at": started_at,
            "total": len(missing),
            "done": len(missing),
            **stats,
            "phase": "done",
            "finished": True,
            "years": args.years,
        })


if __name__ == "__main__":
    main()
