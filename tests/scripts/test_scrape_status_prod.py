"""scrape_status_notify の prod の材料（GCS の実行記録）。"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from src.ops.gcs_io import MemoryIO, write_json
from src.scripts.data import scrape_status_notify as N

JST = ZoneInfo("Asia/Tokyo")
NOW = datetime(2026, 10, 30, 9, 0, tzinfo=JST)          # 金曜 09:00


def _doc():
    return {"scheduler_groups": {"g-0700": {"cron_jst": "0 7 * * *"}, "g-0730": {"cron_jst": "30 7 * * *"}},
            "jobs": [
                {"id": "export-tables-daily", "destination": "gcp", "enabled": True, "ready": True, "group": "g-0700", "desc": "履歴テーブル"},
                {"id": "lap-forecast-morning", "destination": "gcp", "enabled": True, "ready": True, "group": "g-0730"},
                {"id": "not-ready", "destination": "gcp", "enabled": True, "ready": False, "group": "g-0700"},
                {"id": "entries-sync", "destination": "vps", "enabled": True, "ready": True, "schedule_jst": "*/10 8-22 * * *",
                 "fetch_task": {"task": "entries-sync"}},
                {"id": "race-odds-per-race", "destination": "vps", "enabled": True, "ready": True, "reserved_by": "reserve-race-day",
                 "fetch_task": {"task": "race-odds-t60"}, "extra_tasks": ["race-odds-t15"]},
            ]}


def test_collect_prod_reads_job_runs_and_fetch_tasks():
    io = MemoryIO()
    write_json(io, "ops/job_runs/export-tables-daily/20261029T070001.json", {"status": "failed", "started_at": "2026-10-29T07:00:01+09:00", "reason": "x"})
    write_json(io, "ops/job_runs/export-tables-daily/20261030T070001.json", {"status": "failed", "started_at": "2026-10-30T07:00:01+09:00", "reason": "step rc=1"})
    write_json(io, "ops/job_runs/export-tables-daily/latest.json", {"status": "failed"})
    # lap-forecast-morning は 07:30 の記録が無い → 予定から 90 分で未開始
    write_json(io, "ops/fetch_tasks_shadow/done/entries-sync-20261030T0850.json",
               {"claimed_at": "2026-10-30T08:50:30+09:00", "outcome": {"rc": 0, "result": {}}})
    write_json(io, "ops/fetch_tasks_shadow/done/entries-sync-20261030T0900.json",
               {"claimed_at": "2026-10-30T09:00:10+09:00", "outcome": {"rc": 0, "result": {"skipped": "budget_exhausted"}}})
    write_json(io, "ops/fetch_tasks_shadow/failed/race-odds-t15-202605040101.json",
               {"claimed_at": "2026-10-30T08:40:00+09:00", "outcome": {"rc": 3, "access_restriction": True}})
    jobs = {j["id"]: j for j in N.collect_prod(NOW, io, doc=_doc(), shadow=True)}
    assert set(jobs) == {"export-tables-daily", "lap-forecast-morning", "entries-sync", "race-odds-per-race"}
    e = jobs["export-tables-daily"]
    assert e["consecutive_failures"] == 2 and e["cause"] == "step rc=1" and e["overdue_min"] is None
    assert jobs["lap-forecast-morning"]["overdue_min"] == 90
    es = jobs["entries-sync"]
    assert es["consecutive_failures"] == 0 and es["started_at"].startswith("2026-10-30T08:50")      # 上限でスキップした回は数えない
    assert jobs["race-odds-per-race"]["cause_id"] == "403"
    an = N.detect_anomalies(list(jobs.values()), None, None, {})
    assert "cron:export-tables-daily:fail" in an and "cron:lap-forecast-morning:overdue" in an and "net:cause" in an


def test_run_with_prod_source_uses_injected_jobs_and_skips_host_only_inputs(tmp_path):
    res = N.run("summary", NOW, out_dir=tmp_path, source="prod", inputs={"jobs": [], "maintenance": False, "raceday": lambda d: False},
                env={"KEIBA_ENV": "prod"})
    assert res["code"] == 0 and res["payload"]


def test_when_jst_narrows_the_expected_fire():
    doc = {"scheduler_groups": {"g-5min": {"cron_jst": "*/5 * * * *"}},
           "jobs": [{"id": "daily-at-7", "destination": "gcp", "enabled": True, "ready": True, "group": "g-5min", "when_jst": "0 7 * * *"}]}
    io = MemoryIO()
    write_json(io, "ops/job_runs/daily-at-7/20261029T070000.json", {"status": "ok", "started_at": "2026-10-29T07:00:05+09:00"})
    assert N.collect_prod(datetime(2026, 10, 30, 7, 30, tzinfo=JST), io, doc=doc)[0]["overdue_min"] == 30     # グループの 07:30 ではなく 07:00 が予定
    write_json(io, "ops/job_runs/daily-at-7/20261030T070000.json", {"status": "ok", "started_at": "2026-10-30T07:00:04+09:00"})
    assert N.collect_prod(datetime(2026, 10, 30, 7, 30, tzinfo=JST), io, doc=doc)[0]["overdue_min"] is None
