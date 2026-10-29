"""開催日・定期の予約（取得タスク + Cloud Tasks の予測）、VPS のイベント処理、GCP 側の監視。外部接続なし。"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from src.ops import events as EV, fetch_task as FT, heartbeat_schema as HS, heartbeat_store as HST, reservations as R, vps_watchdog as W
from src.ops.gcs_io import MemoryIO, read_json, write_json

JST = ZoneInfo("Asia/Tokyo")
EVE = datetime(2026, 10, 23, 19, 10, tzinfo=JST)          # 前夜
SCHED = [{"race_id": "202605040101", "post_time": datetime(2026, 10, 24, 10, 5, tzinfo=JST)},
         {"race_id": "202605040102", "post_time": datetime(2026, 10, 24, 10, 35, tzinfo=JST)}]
ENV = {"GCS_PROJECT_ID": "proj", "CLOUD_TASKS_OAUTH_SERVICE_ACCOUNT": "keiba-scheduler@proj.iam.gserviceaccount.com", "CLOUD_TASKS_LOCATION": "asia-northeast1"}
OWNED = {"race-odds-t60", "race-odds-t15", "race-odds-t10", "race-odds-t05", "race-odds-t02", "race-weight-check", "entries-sync", "daily-race-lists"}


def queue() -> FT.TaskQueue:
    return FT.TaskQueue(FT.MemoryTaskStore())


# ── 開催日の予約 ──

def test_race_day_reserves_fetch_and_predict_tasks_idempotently():
    q, calls = queue(), []
    res = R.reserve_race_day("20261024", object(), q, now=EVE, load_schedule=lambda s, d: SCHED,
                             enqueue_predict=lambda spec: calls.append(spec) or "created", env={**ENV, "KEIBA_ODDS_LATE_SNAPSHOTS": "1"}, owned=OWNED)
    assert res["races"] == 2 and res["fetch"]["enqueued"] == res["fetch_tasks"] > 20
    kinds = {FT.FetchTask.from_json(q.store.get(k)[0]).task for k in q.store.list("ops/fetch_tasks/pending/")}
    assert {"race-odds-t60", "race-odds-t15", "race-odds-t10", "race-odds-t05", "race-odds-t02", "race-weight-check"} <= kinds
    assert [c["task_id"] for c in calls] == ["predict-202605040101", "predict-202605040102"]
    spec = calls[0]
    assert spec["http_request"]["url"] == "https://run.googleapis.com/v2/projects/proj/locations/asia-northeast1/jobs/keiba-dispatch-l:run"
    body = json.loads(spec["http_request"]["body"])
    assert body["overrides"]["containerOverrides"][0]["args"] == ["-m", "src.ops.job_dispatch", "job", "predict-race", "--arg", "race_id=202605040101", "--arg", "date=20261024"]
    assert spec["http_request"]["oauth_token"]["service_account_email"].startswith("keiba-scheduler@")
    assert spec["schedule_time"] == datetime(2026, 10, 24, 9, 20, tzinfo=JST)           # 発走 45 分前のフォールバック
    again = R.reserve_race_day("20261024", object(), q, now=EVE, load_schedule=lambda s, d: SCHED, enqueue_predict=lambda spec: "exists",
                               env={**ENV, "KEIBA_ODDS_LATE_SNAPSHOTS": "1"}, owned=OWNED)
    assert again["fetch"]["enqueued"] == 0 and again["fetch"]["exists"] == res["fetch"]["enqueued"] and again["predict"]["exists"] == 2


def test_late_snapshots_only_when_flag_on():
    q = queue()
    R.reserve_race_day("20261024", object(), q, now=EVE, load_schedule=lambda s, d: SCHED, enqueue_predict=lambda s: "created", env=ENV, owned=OWNED)
    kinds = {FT.FetchTask.from_json(q.store.get(k)[0]).task for k in q.store.list("ops/fetch_tasks/pending/")}
    assert "race-odds-t10" not in kinds and "race-odds-t60" in kinds


def test_only_agent_owned_tasks_are_enqueued():
    q = queue()
    res = R.reserve_race_day("20261024", object(), q, now=EVE, load_schedule=lambda s, d: SCHED, enqueue_predict=lambda s: "created", env=ENV, owned=set())
    assert res["fetch"] == {"enqueued": 0, "exists": 0, "error": 0} and "race-odds-t60" in res["not_owned_tasks"]     # 担当が空の間はホストの cron が全部やる
    assert res["predict"]["created"] == 2                                      # 予測の予約(GCP)は担当に関係なく積む


def test_past_races_are_not_reserved_and_dry_run_has_no_side_effects():
    q = queue()
    now = datetime(2026, 10, 24, 10, 20, tzinfo=JST)        # 1 レース目は発走済み
    res = R.reserve_race_day("20261024", object(), q, now=now, load_schedule=lambda s, d: SCHED, enqueue_predict=lambda s: "created", env=ENV, owned=OWNED, dry_run=True)
    assert res["past"] == 1 and res["predict_tasks"] == 1 and q.store.list("ops/fetch_tasks/") == []


def test_predict_spec_requires_settings_and_defaults_run_job():
    with pytest.raises(ValueError):
        R.predict_task_spec("202605040101", "20261024", EVE, env={})
    spec = R.predict_task_spec("202605040101", "20261024", EVE, suffix="-w", env=ENV)
    assert spec["task_id"] == "predict-202605040101-w" and spec["queue"] == "keiba-predict"
    assert R.predict_run_job() == "keiba-dispatch-l"


# ── 定期の予約 ──

def test_recurring_expands_cron_and_respects_ownership():
    q = queue()
    owned = {"entries-sync", "daily-race-lists"}
    res = R.reserve_recurring(q, now=datetime(2026, 10, 24, 6, 55, tzinfo=JST), horizon_hours=1, owned=owned)
    res2 = R.reserve_recurring(q, now=datetime(2026, 10, 24, 10, 55, tzinfo=JST), horizon_hours=1, owned=owned)
    ids = sorted(k.rsplit("/", 1)[-1][:-5] for k in q.store.list("ops/fetch_tasks/pending/"))
    assert "daily-race-lists-morning-20261024T0700" in ids
    assert [i for i in ids if i.startswith("entries-sync")] == [f"entries-sync-20261024T11{m:02d}" for m in range(0, 60, 10)]    # 08:00〜22:59 の 10 分おき
    assert not any(i.startswith("jra-baba") for i in ids)                                       # 担当でない
    skipped = {s["job"]: s["reason"] for s in res["skipped_jobs"]}
    assert "jra-baba-morning" in skipped and "担当ではない" in skipped["jra-baba-morning"]
    assert "raceday-evening" in skipped and "担当ではない" in skipped["raceday-evening"]          # ready でも担当（agent_tasks）でなければ積まない
    assert R.reserve_recurring(q, now=datetime(2026, 10, 24, 10, 55, tzinfo=JST), horizon_hours=1, owned=owned)["fetch"]["enqueued"] == 0
    assert res2["tasks"] == 6


def test_recurring_tasks_validate_against_allowlist():
    tasks, _ = R.recurring_tasks(__import__("src.ops.job_map", fromlist=["load"]).load(), datetime(2026, 10, 24, 0, 0, tzinfo=JST), 24,
                                 owned=set(FT.DEFAULT_AGENT_TASKS))
    assert tasks and all(t.task in FT.ALLOWED_TASKS for t in tasks)


# ── イベント ──

def _announce(io, date, rid):
    write_json(io, f"ops/events/weights_announced/{date}/{rid}.json", {"race_id": rid, "date": date, "detected_at": "2026-10-24T09:10:00+09:00"})


def test_pump_triggers_prediction_once_per_announced_race():
    io, calls = MemoryIO(), []
    now = datetime(2026, 10, 24, 9, 11, tzinfo=JST)
    _announce(io, "20261024", "202605040101")
    _announce(io, "20261024", "202605040102")
    res = EV.pump(io, now=now, enqueue_predict=lambda s: calls.append(s["task_id"]) or "created", env=ENV)
    assert res["triggered"] == 2 and sorted(calls) == ["predict-202605040101-w", "predict-202605040102-w"]
    assert read_json(io, "ops/events/predict_triggered/20261024/202605040101.json")["result"] == "created"
    res2 = EV.pump(io, now=now, enqueue_predict=lambda s: calls.append("again"), env=ENV)
    assert res2["triggered"] == 0 and res2["skipped"] == 2 and "again" not in calls


def test_pump_retries_failed_enqueue_next_time():
    io = MemoryIO()
    _announce(io, "20261024", "202605040101")
    now = datetime(2026, 10, 24, 9, 11, tzinfo=JST)

    def boom(spec):
        raise RuntimeError("tasks api down")
    r = EV.pump(io, now=now, enqueue_predict=boom, env=ENV)
    assert r["error"] == 1 and not io.exists("ops/events/predict_triggered/20261024/202605040101.json")
    assert EV.pump(io, now=now, enqueue_predict=lambda s: "created", env=ENV)["triggered"] == 1


def test_pump_ignores_junk_keys_and_looks_at_yesterday_too():
    io = MemoryIO()
    io.put_bytes("ops/events/weights_announced/20261024/not-a-race.json", b"{}")
    _announce(io, "20261023", "202605040101")
    r = EV.pump(io, now=datetime(2026, 10, 24, 0, 5, tzinfo=JST), enqueue_predict=lambda s: "created", env=ENV)
    assert r["seen"] == 1 and r["triggered"] == 1


# ── 監視 ──

def _payload(avail=1400, swap=0, disk=40.0, sent=1_760_000_000, units=None):
    return HS.validate_payload({"v": 1, "host": "vps", "sent_at": sent, "mem": {"total_mb": 1980, "available_mb": avail, "used_mb": 380},
                                "swap": {"total_mb": 1024, "used_mb": swap}, "disk": {"root_used_pct": disk}, "units": units or {"nginx": {"active": "active", "restarts": 0, "mem_mb": 10}},
                                "checks": {"nextjs_http": 200}})


def _backend(tmp_path, received, **kw):
    b = HST.FileBackend(tmp_path / "hb.json")
    HST.update(b, lambda r: HS.append_history(r, _payload(**kw), received))
    return b


def test_watchdog_silent_when_healthy(tmp_path):
    sent = []
    t = 1_760_000_000.0
    res = W.run(MemoryIO(), _backend(tmp_path, t), lambda *a, **k: sent.append((a, k)), now=datetime(2026, 10, 24, 12, 0, tzinfo=JST), clock=t + 60)
    # 予約記録が無い warning だけ（ハートビートは正常）
    assert res["heartbeat"]["sent"] == [] and [a[0][0] for a in sent] == ["定期タスクの予約（reserve-fetch-tasks）の実行記録がありません"]


def test_watchdog_critical_on_stale_heartbeat_and_dedup(tmp_path):
    sent = []
    t = 1_760_000_000.0
    b = _backend(tmp_path, t)
    io = MemoryIO()
    write_json(io, "ops/job_runs/reserve-fetch-tasks/latest.json", {"status": "ok", "finished_at": datetime(2026, 10, 24, 11, 50, tzinfo=JST).isoformat()})
    now = datetime(2026, 10, 24, 12, 0, tzinfo=JST)
    W.run(io, b, lambda *a, **k: sent.append((a, k)), now=now, clock=t + 16 * 60)
    assert [(a[0], k["severity"]) for a, k in sent] == [("VPS のハートビートが途絶えています", "critical")]
    W.run(io, b, lambda *a, **k: sent.append((a, k)), now=now, clock=t + 17 * 60)
    assert len(sent) == 1                                                          # 同じ異常は再送しない


def test_watchdog_memory_swap_disk_thresholds(tmp_path):
    t = 1_760_000_000.0
    sent = []
    b = _backend(tmp_path, t, avail=500, swap=300, disk=85.0)
    io = MemoryIO()
    write_json(io, "ops/job_runs/reserve-fetch-tasks/latest.json", {"status": "ok", "finished_at": datetime(2026, 10, 24, 11, 50, tzinfo=JST).isoformat()})
    W.run(io, b, lambda *a, **k: sent.append((a, k)), now=datetime(2026, 10, 24, 12, 0, tzinfo=JST), clock=t + 30)
    sev = {a[0]: k["severity"] for a, k in sent}
    assert sev["VPS の空きメモリが非常に少ない"] == "critical" and "VPS の swap 使用量が多い" in sev and "VPS のディスク使用率が高い" in sev


def _put_task(io, state, task, tid, not_before, finished=None, outcome=None):
    nb = not_before
    t = FT.FetchTask(task_id=tid, task=task, args={"race_id": "202605040101", "date": "20261024"} if task.startswith("race-") else {},
                     not_before=nb.isoformat(), deadline=(nb + timedelta(minutes=30)).isoformat(), finished_at=finished.isoformat() if finished else None, outcome=outcome)
    io.put_bytes(f"ops/fetch_tasks/{state}/{tid}.json", t.to_json().encode())


def test_fetch_queue_alerts_overdue_and_failed():
    io = MemoryIO()
    now = datetime(2026, 10, 24, 9, 30, tzinfo=JST)
    _put_task(io, "pending", "race-odds-t60", "race-odds-t60-x", now - timedelta(minutes=10))
    _put_task(io, "failed", "entries-sync", "entries-sync-y", now - timedelta(minutes=20), finished=now - timedelta(minutes=10), outcome={"reason": "deadline_exceeded"})
    _put_task(io, "failed", "entries-sync", "entries-sync-old", now - timedelta(hours=5), finished=now - timedelta(hours=4))
    al = {a["key"]: a for a in W.fetch_queue_alerts(io, now)}
    assert al["fetch_overdue"]["severity"] == "critical" and "race-odds-t60-x" in al["fetch_overdue"]["detail"]
    assert al["fetch_failed_recent"]["severity"] == "warning" and "deadline_exceeded" in al["fetch_failed_recent"]["detail"] and "old" not in al["fetch_failed_recent"]["detail"]


def test_reservation_staleness():
    io = MemoryIO()
    now = datetime(2026, 10, 24, 12, 0, tzinfo=JST)
    write_json(io, "ops/job_runs/reserve-fetch-tasks/latest.json", {"status": "ok", "finished_at": (now - timedelta(hours=3)).isoformat()})
    assert W.reservation_alerts(io, now)[0]["key"] == "reserve_stale"
    write_json(io, "ops/job_runs/reserve-fetch-tasks/latest.json", {"status": "ok", "finished_at": (now - timedelta(minutes=20)).isoformat()})
    assert W.reservation_alerts(io, now) == []


def test_race_day_does_not_reserve_predicts_while_predict_race_is_not_ready():
    # 積んでも Job が『準備未完了』で何もせず終わるだけ（開催日に約 36 回の無駄な起動）
    q = FT.TaskQueue(FT.MemoryTaskStore())
    called = []
    res = R.reserve_race_day("20261024", object(), q, now=EVE, load_schedule=lambda s, d: SCHED,
                             enqueue_predict=lambda spec: called.append(spec) or "created", env=ENV, owned=OWNED, predict_ready=False)
    assert called == [] and res["predict"]["skipped"] == res["predict_tasks"] > 0
    assert res["fetch"]["enqueued"] > 0                                         # 取得のタスクは今までどおり積む
