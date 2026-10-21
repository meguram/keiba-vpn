"""VPS 実行役 → GCP のイベント。VPS は ML を持たない（250M）ので、『馬体重が出た』を GCS に置くだけにして、予測は GCP が起動する。

VPS 実行役（``fetch_agent``）は馬体重を検知すると ``gs://BUCKET/ops/events/weights_announced/<日>/<race_id>.json`` を create-only で置く。
ここ（Scheduler → Run Job ``event-pump``。開催日の日中に毎分）がそれを見つけて、まだ起動していないレースの予測タスク
``predict-<race_id>-w`` を Cloud Tasks に即時（数秒後）で積み、``ops/events/predict_triggered/<日>/<race_id>.json`` に済みの印を置く。

- 積むのに失敗したレースは印を置かない（次の毎分で再試行）。Cloud Tasks のタスク名が固定なので、印が無くても二重には動かない。
- 発走 45 分前のフォールバック予測（``predict-<race_id>``）は ``reservations.reserve_race_day`` が前夜に予約済み。
- 対象の日付は今日（JST）と、日をまたいだ直後のために昨日。

    python -m src.ops.events pump [--date YYYYMMDD]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta
from typing import Any, Callable
from zoneinfo import ZoneInfo

from src.ops.gcs_io import BlobIO, read_json, write_json

JST = ZoneInfo("Asia/Tokyo")
ANNOUNCED = "ops/events/weights_announced"
TRIGGERED = "ops/events/predict_triggered"
_RACE_ID = re.compile(r"^\d{12}$")


def announced_races(io: BlobIO, date: str) -> list[str]:
    out = []
    for k in io.list(f"{ANNOUNCED}/{date}/"):
        rid = k.rsplit("/", 1)[-1][: -len(".json")]
        if _RACE_ID.match(rid):
            out.append(rid)
    return sorted(out)


def pump(io: BlobIO, *, now: datetime | None = None, dates: list[str] | None = None,
         enqueue_predict: Callable[[dict[str, Any]], str] | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    from src.ops import reservations as R

    now = now or datetime.now(JST)
    dates = dates or [(now - timedelta(days=1)).strftime("%Y%m%d"), now.strftime("%Y%m%d")]
    out: dict[str, Any] = {"seen": 0, "triggered": 0, "skipped": 0, "error": 0}
    for date in dates:
        for rid in announced_races(io, date):
            out["seen"] += 1
            mark = f"{TRIGGERED}/{date}/{rid}.json"
            if io.exists(mark):
                out["skipped"] += 1
                continue
            try:
                spec = R.predict_task_spec(rid, date, now + timedelta(seconds=5), suffix="-w", env=env)
                result = (enqueue_predict or R.enqueue_run_job)(spec)
                write_json(io, mark, {"race_id": rid, "date": date, "triggered_at": now.isoformat(timespec="seconds"), "task": spec["task_id"], "result": result,
                                      "detected_at": (read_json(io, f"{ANNOUNCED}/{date}/{rid}.json") or {}).get("detected_at")})
                out["triggered"] += 1
            except Exception as exc:  # noqa: BLE001
                out["error"] += 1
                out.setdefault("errors", []).append(f"{rid}: {type(exc).__name__}: {exc}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pump")
    p.add_argument("--date", default=None)
    ns = ap.parse_args(argv)
    from src.ops.gcs_io import default_io
    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()
    res = pump(default_io(), dates=[ns.date] if ns.date else None)
    print(json.dumps(res, ensure_ascii=False))
    return 1 if res["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
