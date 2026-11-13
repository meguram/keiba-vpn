"""定期実行の記録の集め方（src/monitor/schedule_sources.py）: 共用の Run Job の割り当て・中のジョブの失敗・未登録・VPS の記録の時刻。"""

import json
from datetime import datetime, timedelta, timezone

from src.monitor import schedule_sources as S
from src.ops.gcs_io import MemoryIO

JST = timezone(timedelta(hours=9))


def ts(s):
    return datetime.fromisoformat(s).replace(tzinfo=JST).timestamp()


NOW = ts("2026-11-13 09:00:00")
REG = {"jobs": [
    {"id": "a", "where": "gcp", "kind": "scheduler", "status": "active", "schedule": {"cron": "30 7 * * *"}, "grace_min": 30, "weight": "med",
     "ref": {"scheduler": "keiba-g-a", "run_job": "rj-s", "group": "ga"}, "what": "x"},
    {"id": "b", "where": "gcp", "kind": "scheduler", "status": "active", "schedule": {"cron": "0 5 * * *"}, "grace_min": 30, "weight": "med",
     "ref": {"scheduler": "keiba-g-b", "run_job": "rj-s", "group": "gb"}, "what": "x"},
    {"id": "c", "where": "gcp", "kind": "scheduler", "status": "retired", "schedule": {"cron": "0 6 * * *"}, "grace_min": 30, "weight": "med",
     "ref": {"scheduler": "keiba-g-c", "run_job": "rj-l", "group": "gc"}, "what": "x"},
]}
MAP = {"scheduler_groups": {"ga": {"jobs": ["ja"]}, "gb": {"jobs": []}, "gc": {"jobs": []}}}


class Api:
    def __init__(self, ex, sched=None):
        self.ex, self.sched = ex, sched or []

    def scheduler_jobs(self):
        return self.sched

    def executions(self, rj):
        return [dict(e) for e in self.ex.get(rj, [])]


def ex(start, minutes=3, ok=True):
    s = ts(start)
    return {"start": s, "end": s + minutes * 60, "ok": ok, "code": None}


def test_shared_run_job_is_split_by_each_groups_slots():
    g = S.gather(MemoryIO(), REG, MAP, api=Api({"rj-s": [ex("2026-11-13 07:30:10"), ex("2026-11-13 05:00:05"), ex("2026-11-13 03:00:00")]}), now=NOW)
    assert [round(r["start"]) for r in g["obs"]["a"]["runs"]] == [round(ts("2026-11-13 07:30:10"))]
    assert [round(r["start"]) for r in g["obs"]["b"]["runs"]] == [round(ts("2026-11-13 05:00:05"))]   # 03:00 の手動の実行はどちらにも入れない


def test_inner_failure_marks_only_that_run_and_ignores_old_records():
    io = MemoryIO()
    io.put_bytes("ops/job_runs/ja/latest.json", json.dumps({"status": "refused", "started_at": "2026-11-13T07:30:20+09:00"}).encode())
    api = Api({"rj-s": [ex("2026-11-13 07:30:10"), ex("2026-11-13 05:00:05")]})
    g = S.gather(io, REG, MAP, api=api, now=NOW)
    assert g["obs"]["a"]["runs"][0]["ok"] is False and g["obs"]["b"]["runs"][0]["ok"] is True   # 他のグループの回は変えない
    io.put_bytes("ops/job_runs/ja/latest.json", json.dumps({"status": "failed", "started_at": "2026-11-12T07:30:20+09:00"}).encode())
    assert S.gather(io, REG, MAP, api=api, now=NOW)["obs"]["a"]["runs"][0]["ok"] is True   # 前の日の失敗は今日の回に付けない


def test_runs_of_a_job_nobody_schedules_and_unknown_schedulers_are_flagged():
    api = Api({"rj-l": [ex("2026-11-13 06:00:00")]}, sched=[{"name": "keiba-g-c", "state": "ENABLED"}, {"name": "mystery", "state": "ENABLED"}])
    names = [u["name"] for u in S.gather(MemoryIO(), REG, MAP, api=api, now=NOW)["unregistered"]]
    assert any(n.startswith("keiba-g-c") and "廃止" in n for n in names) and "mystery" in names and any(n.startswith("rj-l の実行") for n in names)


def test_vps_record_carries_its_own_time_and_errors():
    io = MemoryIO()
    io.put_bytes("ops/heartbeat/vps/status.json", json.dumps({"written_at": NOW - 1800, "jobs": {}, "error": "systemctl show が失敗"}).encode())
    s = S.gather(io, REG, MAP, api=Api({}), now=NOW)["sources"]["vps"]
    assert s["as_of"] == NOW - 1800 and "systemd を読めない" in s["error"]
