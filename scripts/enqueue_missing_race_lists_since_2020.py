#!/usr/bin/env python3
"""
2020-01-01 から今日までを走査し、race_lists が無い日・
レース件数が is_plausible_race_day_races に合わない日（極端に少ない等）へ
job_kind=date / tasks=[race_list] をキューへ一括投入する。

注意: 「開催日だけ」ローカルに揃えたい場合は、全日でファイルが無い日を
欠損とみなす本スクリプトより、次のバックフィル向け:
  python3 scripts/backfill_race_lists_kaisai_since_2020.py

使い方:
  python3 scripts/enqueue_missing_race_lists_since_2020.py
  python3 scripts/enqueue_missing_race_lists_since_2020.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scraper.job_queue import ScrapeJobQueue  # noqa: E402
from scraper.netkeiba_top_race_list import is_plausible_race_day_races  # noqa: E402
from scraper.storage import HybridStorage  # noqa: E402


def collect_targets(
    storage: HybridStorage,
    start: date,
    end: date,
) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    implausible: list[str] = []
    cur = start
    while cur <= end:
        k = cur.strftime("%Y%m%d")
        data = storage.load("race_lists", k)
        races = (data or {}).get("races") or []
        if not data:
            missing.append(k)
        elif races and not is_plausible_race_day_races(races):
            implausible.append(k)
        cur += timedelta(days=1)
    # 同一日が両方に入らないよう、implausible はファイルあり
    return missing, implausible


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="キューへ入れず件数だけ表示")
    p.add_argument(
        "--end",
        type=str,
        default="",
        help="終了日 YYYY-MM-DD（省略時は今日・ローカル日付）",
    )
    args = p.parse_args()

    storage = HybridStorage()
    start = date(2020, 1, 1)
    if args.end:
        end = date.fromisoformat(args.end)
    else:
        end = date.today()

    missing, implausible = collect_targets(storage, start, end)
    todo = sorted(set(missing + implausible))
    print(f"期間: {start} .. {end}（{(end - start).days + 1} 日）")
    print(f"race_lists 未保存（load が空）: {len(missing)} 日")
    print(f"レースありだが件数が不自然（極端に少ない等）: {len(implausible)} 日")
    print(f"キュー投入対象（重複除く）: {len(todo)} 日")

    if args.dry_run:
        print("--dry-run のため投入しません")
        if todo[:15]:
            print("先頭例:", ", ".join(todo[:15]))
        return 0

    specs = [
        {
            "job_kind": "date",
            "target_id": d,
            "tasks": ["race_list"],
            "smart_skip": False,
        }
        for d in todo
    ]
    q = ScrapeJobQueue()
    stats = q.bulk_add_jobs(specs)
    print("bulk_add_jobs:", stats)
    print(
        "API サーバーのキューワーカーが動いていれば順次処理されます。"
        " 単体ワーカー: python3 -m scraper.queue_worker"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
