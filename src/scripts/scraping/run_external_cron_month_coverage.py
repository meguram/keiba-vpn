#!/usr/bin/env python3
"""
「スクレイピング自動実行（外部 cron）」相当のうち、バッチで再現できる範囲をまとめて実行する。

1) auto_scrape サブプロセス（**netkeiba 取得は scrape_queue.json 経由**・既定）
   - daily-race-lists, catchup-missing

2) **金曜 ``weekly-update`` と同一のキュー処理**（``run_weekly_update_for_dates``）
   - 窓 ``[today-days_back .. min(today+days_ahead, today)]`` に含まれる **race_lists が存在する開催日**
     について、レースごとに ``race_result`` / ``race_index`` / ``race_barometer``（上書き再取得）、続けて馬
     ``horse_profile``（成績・プロフィール上書き・``skip_pedigree``）、``horse_pedigree_5gen``（未保持のみキュー・既存スキップ）をキュー投入（``auto_scrape_queue`` 実装＝金曜 cron の正本と同じ）。
   - 金曜 cron 本体は「先週のみ・金曜日のみ」起動のため、**月次バッチではここで明示的に実行**する。

3) 任意: ``run_scrapes_in_cron_window.py``（前日夕＋30日窓の馬名インデックス/成長曲線）

4) スクレイピングキューへ投入（出馬表・指数系）
   - 同一窓の JRA レースに対し **race_shutuba → smartrc**（smart_skip=True）

理由: /cron-jobs の「外部 cron」一覧は ``python -m src.scraper.auto_scrape`` を直接起動する。
auto_scrape は既定で **スクレイピングキュー**に投入してからワーカーが取得する。

Usage:
  cd keiba-vpn && python3 -m src.scripts.scraping.run_external_cron_month_coverage
  python3 -m src.scripts.scraping.run_external_cron_month_coverage --days-back 30 --days-ahead 14 --skip-eve-batch
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")


def _run_auto_scrape_task(repo: Path, task: str, log: logging.Logger) -> int:
    log.info("subprocess: auto_scrape --task %s", task)
    p = subprocess.run(
        [sys.executable, "-m", "src.scraper.auto_scrape", "--task", task],
        cwd=str(repo),
    )
    log.info("  -> exit=%s", p.returncode)
    return int(p.returncode)


def _race_list_dates_in_window(
    repo: Path, start: date, end: date, log: logging.Logger
) -> list[str]:
    """開催日キー YYYYMMDD（race_lists にファイルがある日）を期間で列挙。"""
    import os

    os.chdir(repo)
    from src.scraper.storage import HybridStorage

    lo = start.strftime("%Y%m%d")
    hi = end.strftime("%Y%m%d")
    keys = sorted(HybridStorage().list_keys("race_lists"))
    out = [k for k in keys if lo <= k <= hi]
    log.info("race_lists 開催日 [%s..%s]: %d 日", lo, hi, len(out))
    return out


def _run_weekly_style_for_window(
    repo: Path, start: date, end: date, today: date, log: logging.Logger
) -> None:
    """
    金曜 ``task_weekly_update`` がやる ``run_weekly_update_for_dates`` を、
    窓内の全開催日（ただし未来日は除く）に対して実行する。
    """
    import os

    scan_end = min(end, today)
    dates_ymd = _race_list_dates_in_window(repo, start, scan_end, log)
    if not dates_ymd:
        log.info("=== Phase 2: weekly-update 相当 (skip: 開催日0) ===")
        return

    os.chdir(repo)
    from src.scraper.auto_scrape_queue import run_weekly_update_for_dates

    log.info(
        "=== Phase 2: weekly-update 相当 (金曜定期と同一キュー) %d 開催日 ===",
        len(dates_ymd),
    )
    out = run_weekly_update_for_dates(dates_ymd)
    log.info(
        "  -> status=%s races=%s queue_race=%s",
        out.get("status"),
        out.get("races"),
        (out.get("queue_race_bulk_add") or {}),
    )


def _enqueue_period(
    repo: Path,
    start: date,
    end: date,
    tasks: list[str],
    log: logging.Logger,
) -> dict:
    import os

    os.chdir(repo)
    from src.scraper.job_queue import PRIORITY_URGENT_PEDIGREE_5GEN, ScrapeJobQueue, kick_process_queue_background
    from src.scraper.period_runners import enqueue_race_tasks_for_race_period
    from src.scraper.storage import HybridStorage

    storage = HybridStorage()
    queue = ScrapeJobQueue()
    body_start = start.strftime("%Y%m%d")
    body_end = end.strftime("%Y%m%d")
    log.info("enqueue_race_tasks_for_race_period %s..%s tasks=%s", body_start, body_end, tasks)
    r = enqueue_race_tasks_for_race_period(
        storage,
        queue,
        start_date=body_start,
        end_date=body_end,
        tasks=tasks,
        limit=100_000,
        dry_run=False,
        jra_only=True,
        smart_skip=True,
        priority=PRIORITY_URGENT_PEDIGREE_5GEN,
    )
    log.info("  -> %s", r)
    created = int(r.get("created") or 0)
    if created > 0:
        kick_process_queue_background()
        log.info("kick_process_queue_background() called")
    return r


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days-back", type=int, default=30)
    parser.add_argument("--days-ahead", type=int, default=14)
    parser.add_argument("--skip-eve-batch", action="store_true", help="run_scrapes_in_cron_window.py をスキップ")
    parser.add_argument(
        "--skip-weekly-style",
        action="store_true",
        help="金曜 weekly-update 相当のキュー一括（race_result/指数/馬）をスキップ",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[3]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from src.utils.keiba_logging import script_basic_config

    script_basic_config()
    log = logging.getLogger("external_cron_month_coverage")

    today = datetime.now(JST).date()
    start = today - timedelta(days=max(1, args.days_back))
    end = today + timedelta(days=max(0, args.days_ahead))

    log.info("repo=%s window JST: %s .. %s", repo, start, end)

    log.info("=== Phase 1: auto_scrape（キュー経由） ===")
    for t in ("daily-race-lists", "catchup-missing"):
        _run_auto_scrape_task(repo, t, log)

    if not args.skip_weekly_style:
        _run_weekly_style_for_window(repo, start, end, today, log)

    if not args.skip_eve_batch:
        log.info("=== Phase 3: run_scrapes_in_cron_window.py (eve + 30日窓の馬名/成長曲線) ===")
        p = subprocess.run(
            [sys.executable, str(repo / "scripts" / "run_scrapes_in_cron_window.py")],
            cwd=str(repo),
        )
        log.info("  -> exit=%s", p.returncode)

    log.info("=== Phase 4: enqueue race_shutuba + smartrc (queue) ===")
    _enqueue_period(repo, start, end, ["race_shutuba"], log)
    _enqueue_period(repo, start, end, ["smartrc"], log)

    log.info("=== done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
