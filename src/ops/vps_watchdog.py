"""VPS と取得の流れの監視（GCP 側。Scheduler → Run Job で 5 分おき）。VPS 自身は Slack の秘密を持たず、死んでも GCP 側が気づく。

見るもの:
1. **VPS ハートビート**（``ops/heartbeat/latest.json``）: 15 分途絶=緊急、MemAvailable < 1.0GB が 3 回連続=注意、< 600MB=緊急、
   swap > 200MB=注意、ユニット failed・再起動多発=注意、ディスク > 80%=注意。評価は ``heartbeat_schema.evaluate``（受信 API の
   ``/ops/heartbeat/check`` と同じ関数）。同じ異常は 6 時間たつまで再送しない。復旧は 1 回だけ通知。
2. **取得タスクの滞留**（``ops/fetch_tasks/pending``）: 実行時刻を過ぎても pending のまま（実行役が止まっている）。T-60/T-15 のオッズは緊急。
3. **取得タスクの失敗**（``ops/fetch_tasks/failed``）: 直近 30 分に failed になったもの。
4. **予約の鮮度**: ``reserve-fetch-tasks`` の最終成功が 2 時間以上前（定期タスクが積まれなくなる）。

通知は ``alerting.notify``（``slack_notify.send_payload`` 経由。環境名・実行場所が付く）で、同じ内容を ``ops/alerts/`` に記録する。
CLI: ``python -m src.ops.vps_watchdog [--send] [--dry-run]``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Callable
from zoneinfo import ZoneInfo

from src.ops import fetch_task as FT
from src.ops import heartbeat_schema as HS
from src.ops import heartbeat_store as HST
from src.ops.gcs_io import BlobIO, read_json, write_json

JST = ZoneInfo("Asia/Tokyo")
STATE_KEY = "ops/watchdog/state.json"
OVERDUE_GRACE_S = 180            # 実行役は 60 秒おきに確認する。3 分過ぎても pending なら止まっている
FAILED_LOOKBACK_S = 30 * 60
RESERVE_STALE_S = 2 * 3600
Notify = Callable[..., Any]


def check_heartbeat(backend: HST.Backend, now: float, notify: Notify) -> dict[str, Any]:
    sent: list[dict] = []
    recovered: list[str] = []

    def _fn(rec: dict) -> dict:
        alerts = HS.evaluate(rec, now)
        to_send, rec_keys, new_state = HS.plan_notifications(alerts, rec.get("alert_state") or {}, now)
        sent.extend(to_send)
        recovered.extend(rec_keys)
        return {**rec, "alert_state": new_state}

    HST.update(backend, _fn)
    for a in sent:
        notify(a["title"], [a["detail"]] if a["detail"] else [], severity=a["severity"])
    if recovered:
        notify("VPS の異常が解消しました", [", ".join(recovered)], severity="normal")
    return {"sent": [a["key"] for a in sent], "recovered": recovered}


def fetch_queue_alerts(io: BlobIO, now: datetime) -> list[dict[str, str]]:
    """滞留（pending のまま実行時刻を過ぎた）と、直近の failed。"""
    out: list[dict[str, str]] = []
    q = FT.TaskQueue(_IoStore(io))
    late: list[FT.FetchTask] = []
    for tid in q.list_ids("pending"):
        t = q.load("pending", tid)
        if t is None:
            continue
        try:
            if FT.parse_ts(t.not_before) + timedelta(seconds=OVERDUE_GRACE_S) < now and FT.parse_ts(t.deadline) + timedelta(minutes=30) > now:
                late.append(t)
        except ValueError:
            continue
    if late:
        crit = [t for t in late if t.task in FT.RACE_ODDS_TASKS and t.task in ("race-odds-t60", "race-odds-t15")]
        out.append({"key": "fetch_overdue", "severity": "critical" if crit else "warning",
                    "title": f"取得タスクが実行時刻を過ぎても未実行（{len(late)} 件）",
                    "detail": ", ".join(sorted(t.task_id for t in late)[:8]) + "（VPS の実行役が止まっている疑い）"})
    failed = []
    for tid in q.list_ids("failed"):
        t = q.load("failed", tid)
        if t is None or not t.finished_at:
            continue
        try:
            if now - FT.parse_ts(t.finished_at) <= timedelta(seconds=FAILED_LOOKBACK_S):
                failed.append(t)
        except ValueError:
            continue
    if failed:
        crit = [t for t in failed if t.task in ("race-odds-t60", "race-odds-t15")]
        out.append({"key": "fetch_failed_recent", "severity": "critical" if crit else "warning",
                    "title": f"取得タスクが失敗（直近 30 分に {len(failed)} 件）",
                    "detail": ", ".join(f"{t.task_id}:{(t.outcome or {}).get('reason', '?')}" for t in failed[:6])})
    return out


def reservation_alerts(io: BlobIO, now: datetime) -> list[dict[str, str]]:
    rec = read_json(io, "ops/job_runs/reserve-fetch-tasks/latest.json")
    if not rec:
        return [{"key": "reserve_never", "severity": "warning", "title": "定期タスクの予約（reserve-fetch-tasks）の実行記録がありません", "detail": "Scheduler の g-hourly を確認"}]
    try:
        age = (now - datetime.fromisoformat(rec["finished_at"])).total_seconds()
    except (KeyError, ValueError):
        age = RESERVE_STALE_S + 1
    if rec.get("status") not in ("ok", "partial") or age > RESERVE_STALE_S:
        return [{"key": "reserve_stale", "severity": "critical", "title": "定期タスクの予約が止まっています",
                 "detail": f"最終 {rec.get('status')}・{int(age // 60)} 分前（VPS の定期取得が積まれなくなる）"}]
    return []


class _IoStore:
    """fetch_task.TaskStore の読み取り専用アダプタ（監視は書き込まない）。"""

    def __init__(self, io: BlobIO) -> None:
        self.io = io

    def get(self, key: str):
        raw = self.io.get_bytes(key)
        return None if raw is None else (raw.decode("utf-8"), 0)

    def create(self, key: str, text: str) -> bool:      # pragma: no cover - 使わない
        raise RuntimeError("read-only")

    def delete(self, key: str) -> bool:                 # pragma: no cover - 使わない
        raise RuntimeError("read-only")

    def list(self, prefix: str) -> list[str]:
        return self.io.list(prefix)


def run(io: BlobIO, backend: HST.Backend, notify: Notify, *, now: datetime | None = None, clock: float | None = None, state_key: str = STATE_KEY) -> dict[str, Any]:
    now = now or datetime.now(JST)
    t = time.time() if clock is None else clock
    res: dict[str, Any] = {"heartbeat": check_heartbeat(backend, t, notify)}
    alerts = fetch_queue_alerts(io, now) + reservation_alerts(io, now)
    state = read_json(io, state_key) or {}
    to_send, recovered, new_state = HS.plan_notifications(alerts, state, t)
    for a in to_send:
        notify(a["title"], [a["detail"]] if a["detail"] else [], severity=a["severity"])
    if recovered:
        notify("取得の流れの異常が解消しました", [", ".join(recovered)], severity="normal")
    write_json(io, state_key, new_state)
    res["flow"] = {"sent": [a["key"] for a in to_send], "recovered": recovered, "active": [a["key"] for a in alerts]}
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--send", action="store_true", help="（互換用）通知は KEIBA_SLACK_NOTIFY=1 のときだけ送られる")
    ap.add_argument("--dry-run", action="store_true", help="通知せず、見つかった異常だけを出す")
    ns = ap.parse_args(argv)
    from src.ops import alerting
    from src.ops.gcs_io import default_io
    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()
    io = default_io()
    backend = HST.GcsBackend()
    if ns.dry_run:
        rec, _ = backend.read()
        now = datetime.now(JST)
        print(json.dumps({"heartbeat": [a["key"] for a in HS.evaluate(rec or HS.new_record(), time.time())],
                          "flow": [a["key"] for a in fetch_queue_alerts(io, now) + reservation_alerts(io, now)]}, ensure_ascii=False))
        return 0

    def notify(title: str, lines: list[str], *, severity: str = "warning") -> Any:
        return alerting.notify(title, lines, severity=severity, job="vps-watchdog", io=io)

    print(json.dumps(run(io, backend, notify), ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
