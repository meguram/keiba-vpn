#!/usr/bin/env python3
"""
指定期間の「開催日」だけについて race_lists をローカルに揃える（既定開始は 2024-01-01）。

全日付を走査するが、db.netkeiba の日別一覧にレースが 1 件も無い日は JRA 非開催とみなし
保存しない（カレンダー全日に空ファイルを作らない）。

既に is_plausible_race_day_races を満たすデータがあればスキップ（--force で再取得）。

使い方:
  python3 scripts/backfill_race_lists_kaisai_since_2020.py --dry-run
  python3 scripts/backfill_race_lists_kaisai_since_2020.py
  python3 scripts/backfill_race_lists_kaisai_since_2020.py --start 2020-01-01 --end 2025-12-31 --sleep 0.3
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scraper.client import NetkeibaClient  # noqa: E402
from scraper.netkeiba_top_race_list import is_plausible_race_day_races  # noqa: E402
from scraper.parsers import RaceListParser  # noqa: E402
from scraper.storage import HybridStorage  # noqa: E402

logger = logging.getLogger(__name__)

DB_LIST_URL = "https://db.netkeiba.com/race/list/{date}/"


def run(
    *,
    start: date,
    end: date,
    dry_run: bool,
    force: bool,
    accept_implausible: bool,
    sleep_sec: float,
) -> int:
    storage = HybridStorage()
    client = NetkeibaClient(auto_login=True)
    parser = RaceListParser()
    stats: Counter[str] = Counter()
    cur = start

    while cur <= end:
        k = cur.strftime("%Y%m%d")
        existing = storage.load("race_lists", k)
        ex_races = (existing or {}).get("races") or []

        if (
            existing
            and ex_races
            and is_plausible_race_day_races(ex_races)
            and not force
        ):
            stats["skip_already_ok"] += 1
            cur += timedelta(days=1)
            continue

        url = DB_LIST_URL.format(date=k)
        try:
            html = client.fetch(url, use_cache=False)
        except Exception as e:
            logger.error("取得失敗 %s: %s", k, e)
            stats["fetch_error"] += 1
            cur += timedelta(days=1)
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            continue

        data = parser.parse(html, date=k)
        races = data.get("races") or []

        if not races:
            stats["no_meeting_day"] += 1
            cur += timedelta(days=1)
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            continue

        if not is_plausible_race_day_races(races):
            logger.warning(
                "%s: 件数が想定範囲外 (%d 件) — %s",
                k,
                len(races),
                "保存します" if accept_implausible else "スキップ（--accept-implausible で保存可）",
            )
            stats["implausible"] += 1
            if not accept_implausible:
                cur += timedelta(days=1)
                if sleep_sec > 0:
                    time.sleep(sleep_sec)
                continue

        if "_meta" not in data:
            data["_meta"] = {}
        data["_meta"]["source"] = "backfill_race_lists_kaisai_since_2020"

        if dry_run:
            stats["would_save"] += 1
            logger.info("[dry-run] 保存予定 %s (%d レース)", k, len(races))
        else:
            storage.save("race_lists", k, data)
            stats["saved"] += 1
            logger.info("保存 %s (%d レース)", k, len(races))

        cur += timedelta(days=1)
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    print("--- 集計 ---")
    for key in sorted(stats.keys()):
        print(f"  {key}: {stats[key]}")
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    p = argparse.ArgumentParser(description="2020年以降の開催日のみ race_lists をバックフィル")
    p.add_argument("--start", type=str, default="2024-01-01", help="開始日 YYYY-MM-DD")
    p.add_argument(
        "--end",
        type=str,
        default="",
        help="終了日 YYYY-MM-DD（省略時は今日）",
    )
    p.add_argument("--dry-run", action="store_true", help="保存せず件数のみ")
    p.add_argument(
        "--force",
        action="store_true",
        help="既に妥当な race_lists があっても再取得・上書き",
    )
    p.add_argument(
        "--accept-implausible",
        action="store_true",
        help="件数が想定範囲外（4 未満・37 件以上）でも保存する",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.4,
        help="1 日あたりの待機秒（サーバー負荷軽減）",
    )
    args = p.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today()
    if end < start:
        print("error: --end は --start 以降にしてください", file=sys.stderr)
        return 1

    return run(
        start=start,
        end=end,
        dry_run=args.dry_run,
        force=args.force,
        accept_implausible=args.accept_implausible,
        sleep_sec=args.sleep,
    )


if __name__ == "__main__":
    raise SystemExit(main())
