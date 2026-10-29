"""取得タスクの計画と投入（GCP の司令塔 = Cloud Scheduler / Cloud Run Job から、またはホストから実行）。

  python -m src.ops.fetch_plan day --date YYYYMMDD [--late] [--no-result] [--weight-poll-sec 180] [--dry-run]   # 開催日のレース別タスクを積む（前日夜〜当日朝）
  python -m src.ops.fetch_plan recurring --task entries-sync [--dry-run]                          # 定期タスク（10 分おき等）を 1 件積む
  python -m src.ops.fetch_plan show [--state pending]                                             # 表の中身

投入は冪等（task_id が決定的）。実行役が担当していないタスク名を積んでも、実行役は拾わない（pending のまま期限切れ → failed）。
発走時刻表は ``auto_scrape_queue`` と同じ（race_day_schedule 優先）。T-60 は既に T-15 を過ぎたレースには積まない（runner と同じ規則）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any

from src.ops import fetch_task as ft

KIND_TASK = {"t60": "race-odds-t60", "t15": "race-odds-t15", "t10": "race-odds-t10", "t05": "race-odds-t05", "t02": "race-odds-t02"}
LATE_LEADS = (10, 5, 2)


def plan_day(schedule: list[dict[str, Any]], date_ymd: str, *, late: bool = False, weight: bool = True,
             result: bool = True, weight_poll_sec: int = 180, weight_start_min: float = 65.0, weight_deadline_min: float = 30.0,
             now: datetime | None = None, build_events=None) -> list[ft.FetchTask]:
    """1 開催日のレース別タスク。``schedule`` は ``[{"race_id", "post_time"(tz 付き datetime)}]``。"""
    if build_events is None:
        from src.scraper.auto_scrape_queue import build_odds_events as build_events
    now = now or ft.now_jst()
    tasks: list[ft.FetchTask] = []
    for e in build_events(schedule, late_leads=LATE_LEADS if late else ()):
        name = KIND_TASK.get(e["kind"])
        if name is None or e["at"] < now - timedelta(seconds=ft.DEFAULT_GRACE_SECONDS.get(name, 600)):
            continue                                                   # 既に過ぎた予定は積まない
        if e["kind"] == "t60" and e["t15_at"] <= now:
            continue                                                   # T-15 が同じデータを取る
        tasks.append(ft.make_task(name, e["at"], args={"race_id": str(e["race"]["race_id"]), "date": date_ymd}))
    if weight:
        for r in schedule:
            start = r["post_time"] - timedelta(minutes=weight_start_min)
            end = r["post_time"] - timedelta(minutes=weight_deadline_min)
            t = start
            while t <= end:
                if t >= now - timedelta(minutes=2):
                    tasks.append(ft.make_task(ft.RACE_WEIGHT_TASK, t, args={"race_id": str(r["race_id"]), "date": date_ymd},
                                              grace_seconds=int(weight_poll_sec)))
                t += timedelta(seconds=weight_poll_sec)
    if result:
        for r in schedule:
            t = r["post_time"] + timedelta(minutes=15)
            if t >= now - timedelta(seconds=ft.DEFAULT_GRACE_SECONDS_ANY):
                tasks.append(ft.make_task(ft.RACE_RESULT_TASK, t, args={"race_id": str(r["race_id"]), "date": date_ymd}))
    tasks.sort(key=lambda t: (t.not_before, t.task_id))
    return tasks


def recurring_task(task: str, now: datetime | None = None, *, interval_min: int = 10) -> ft.FetchTask:
    """定期タスク 1 件。task_id は interval_min 分の区切りで決まる（同じ区間に二重に積まない）。"""
    now = now or ft.now_jst()
    slot = now.replace(minute=(now.minute // interval_min) * interval_min, second=0, microsecond=0)
    return ft.make_task(task, slot, grace_seconds=interval_min * 60)


def _queue() -> ft.TaskQueue:
    from dotenv import load_dotenv

    load_dotenv()
    return ft.TaskQueue(ft.GcsTaskStore(os.environ.get("GCS_BUCKET", "")))


def _load_schedule(date_ymd: str) -> list[dict[str, Any]]:
    from src.scraper.auto_scrape_queue import _fetch_race_schedule_storage
    from src.scraper.storage import HybridStorage

    return _fetch_race_schedule_storage(HybridStorage(), date_ymd)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("day")
    d.add_argument("--date", required=True)
    d.add_argument("--late", action="store_true", help="直前 T-10/T-5/T-2 の単複オッズも積む（KEIBA_ODDS_LATE_SNAPSHOTS=1 のとき）")
    d.add_argument("--no-weight", action="store_true")
    d.add_argument("--no-result", action="store_true")
    d.add_argument("--weight-poll-sec", type=int, default=180)
    d.add_argument("--dry-run", action="store_true")
    r = sub.add_parser("recurring")
    r.add_argument("--task", required=True)
    r.add_argument("--interval-min", type=int, default=10)
    r.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("show")
    s.add_argument("--state", default="pending", choices=ft.STATES)
    a = ap.parse_args(argv)
    if a.cmd == "show":
        q = _queue()
        print(json.dumps([json.loads(q.load(a.state, i).to_json()) for i in q.list_ids(a.state)], ensure_ascii=False, indent=1))
        return 0
    if a.cmd == "recurring":
        tasks = [recurring_task(a.task, interval_min=a.interval_min)]
    else:
        late = a.late or os.environ.get("KEIBA_ODDS_LATE_SNAPSHOTS", "").strip().lower() in ("1", "true", "yes")
        tasks = plan_day(_load_schedule(a.date), a.date, late=late, weight=not a.no_weight, result=not a.no_result, weight_poll_sec=a.weight_poll_sec)
    if a.dry_run:
        for t in tasks[:20]:
            print(t.task_id, t.not_before)
        print(f"... 合計 {len(tasks)} 件（dry-run: 積んでいません）")
        return 0
    q = _queue()
    added = sum(1 for t in tasks if q.enqueue(t))
    print(f"積みました: {added} 件 / 計画 {len(tasks)} 件（残りは既に表にあります）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
