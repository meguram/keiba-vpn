"""開催日の予測ワークフローを起動する CLI（各レースの発走45分前に予測）。

モード:
  plan       起動時刻の一覧を表示するだけ（何も実行・登録しない）
  enqueue    全レースを Cloud Tasks に予約配信として一括登録（開催日の朝に1回）
  run-local  このプロセスが発走-45分まで待って順に実行（VPS・開発用）
  single     指定した race_id を今すぐ1件予測

例:
  python -m src.scripts.cloud_jobs.predict_race_day --mode plan --date 20261004
  python -m src.scripts.cloud_jobs.predict_race_day --mode enqueue --date 20261004
  python -m src.scripts.cloud_jobs.predict_race_day --mode run-local --date 20261004
  python -m src.scripts.cloud_jobs.predict_race_day --mode single --race-id 202605030811
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

_JST = ZoneInfo("Asia/Tokyo")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["plan", "enqueue", "run-local", "single"], required=True)
    parser.add_argument("--date", default="", help="開催日 YYYYMMDD（省略時は今日 JST）")
    parser.add_argument("--race-id", default="", help="single モードの race_id")
    parser.add_argument("--lead-minutes", type=int, default=45, help="発走の何分前に予測するか（既定45）")
    args = parser.parse_args(argv)

    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()

    from src.pipeline.inference import race_day_workflow as wf
    from src.scraper.storage import HybridStorage

    storage = HybridStorage(".")
    date_fmt = args.date or datetime.now(_JST).strftime("%Y%m%d")

    if args.mode == "single":
        if not args.race_id:
            parser.error("--mode single には --race-id が必要です")
        res = wf.predict_race(args.race_id, storage, source="t45_cli")
        print(json.dumps({k: res.get(k) for k in ("race_id", "status", "error", "model_version", "total_horses", "elapsed_sec")}, ensure_ascii=False))
        return 0 if res.get("status") == "success" else 1

    if args.mode == "plan":
        plan = wf.plan_race_day(storage, date_fmt, lead_minutes=args.lead_minutes)
        for r in plan:
            print(f"{r['run_at'].strftime('%H:%M')} 起動 / {r['post_time'].strftime('%H:%M')} 発走  {r['race_id']}  {r.get('venue', '')}{r.get('round', '')}R")
        print(f"計 {len(plan)} レース")
        return 0

    if args.mode == "enqueue":
        results = wf.enqueue_race_day_tasks(storage, date_fmt, lead_minutes=args.lead_minutes)
    else:
        results = wf.run_race_day_local(storage, date_fmt, lead_minutes=args.lead_minutes)

    summary: dict[str, int] = {}
    for r in results:
        summary[r["status"]] = summary.get(r["status"], 0) + 1
    print(json.dumps({"date": date_fmt, "mode": args.mode, "summary": summary}, ensure_ascii=False))
    return 1 if summary.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
