"""切替判定（cron_cutover_status）・適用スクリプトの生成（cutover_plan）・crontab の停止／復帰（host_cron_switch.sh）・担当の切替（job_owner + cron_run.sh）。"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.ops import job_map, job_owner
from src.ops.gcs_io import MemoryIO, write_json
from src.scripts.ops import cron_cutover_status as CS, cutover_plan as CP

JST = ZoneInfo("Asia/Tokyo")
NOW = datetime(2026, 11, 10, 12, 0, tzinfo=JST)
ROOT = Path(__file__).resolve().parents[2]


def doc(jobs):
    return {"cutover_defaults": {"min_consecutive_success": 3, "max_start_delay_sec": 600, "max_start_delay_sec_vps": 120, "window_days": 14, "require_parity": True,
                                 "parity_min_days": 2, "raceday_min_days": 2, "max_alerts": 0}, "jobs": jobs}


def gj(jid="j1", parity="json_tree", **kw):
    j = {"id": jid, "destination": "gcp", "enabled": True, "ready": True, "blockers": [], "origin": {"cron_job_id": jid + "_c", "tag": "T"}, "wave": 2, "parity": {"kind": parity}}
    j.update(kw)
    return j


def put_run(io, jid, i, status="ok", delay=30):
    sched = NOW - timedelta(days=i + 1)
    write_json(io, f"ops/job_runs/{jid}/{(sched + timedelta(seconds=delay)).strftime('%Y%m%dT%H%M%S')}.json",
               {"job": jid, "status": status, "scheduled_at": sched.isoformat(), "started_at": (sched + timedelta(seconds=delay)).isoformat(), "reason": "boom" if status == "failed" else None})


def put_parity(io, jid, i, ok=True):
    write_json(io, f"ops/parity/{(NOW - timedelta(days=i + 1)).strftime('%Y%m%d')}/{jid}.json", {"job": jid, "date": (NOW - timedelta(days=i + 1)).strftime("%Y%m%d"), "ok": ok})


def state_of(d, jid="j1", io=None):
    res = CS.evaluate_all(d, CS.Records(io), NOW)
    return next(r for r in res["rows"] if r["job"] == jid), res


def test_gcp_job_passes_with_clean_history_and_parity():
    io = MemoryIO()
    for i in range(3):
        put_run(io, "j1", i)
    for i in range(2):
        put_parity(io, "j1", i)
    row, res = state_of(doc([gj()]), io=io)
    assert row["state"] == "合格" and res["cut_units"]["j1_c"]["ready_to_stop"] is True


def test_failure_blocks():
    io = MemoryIO()
    for i in range(3):
        put_run(io, "j1", i)
    for i in range(2):
        put_parity(io, "j1", i)
    put_run(io, "j1", -1, "failed")          # いちばん新しい実行が失敗
    row, _ = state_of(doc([gj()]), io=io)
    assert row["state"] == "不合格" and "consecutive_success" in row["reason"]


def test_insufficient_runs_late_start_parity_and_alerts_each_block():
    d = doc([gj()])
    io = MemoryIO()
    put_run(io, "j1", 0)
    assert "データ不足" in state_of(d, io=io)[0]["reason"]
    io = MemoryIO()
    for i in range(3):
        put_run(io, "j1", i, delay=900 if i == 1 else 30)
        put_parity(io, "j1", i)
    assert "start_delay" in state_of(d, io=io)[0]["reason"]
    io = MemoryIO()
    for i in range(3):
        put_run(io, "j1", i)
    put_parity(io, "j1", 0)
    put_parity(io, "j1", 1, ok=False)
    assert "parity" in state_of(d, io=io)[0]["reason"]
    io = MemoryIO()
    for i in range(3):
        put_run(io, "j1", i)
    for i in range(2):
        put_parity(io, "j1", i)
    write_json(io, f"ops/alerts/{(NOW - timedelta(days=1)).strftime('%Y%m%d')}/120000-j1-critical.json", {"job": "j1", "severity": "critical", "title": "x"})
    assert "no_alerts" in state_of(d, io=io)[0]["reason"]


def test_skipped_and_waiting_runs_are_not_counted():
    io = MemoryIO()
    for i in range(3):
        put_run(io, "j1", i)
        put_run(io, "j1", i + 10, "skipped")
    for i in range(2):
        put_parity(io, "j1", i)
    assert state_of(doc([gj()]), io=io)[0]["state"] == "合格"


def test_job_without_parity_needs_no_parity_check():
    io = MemoryIO()
    for i in range(3):
        put_run(io, "j1", i)
    assert state_of(doc([gj(parity="none")]), io=io)[0]["state"] == "合格"


def test_host_unready_disabled_and_retire_states():
    d = doc([gj("h", destination="host", reason="自宅 IP"), gj("u", ready=False, blockers=["サイズ未確認"]), gj("x", enabled=False), gj("r", destination="retire", reason="PG 廃止")])
    res = CS.evaluate_all(d, CS.Records(MemoryIO()), NOW)
    st = {r["job"]: r["state"] for r in res["rows"]}
    assert st == {"h": "N/A", "u": "未着手", "x": "無効", "r": "対象外"}
    assert res["cut_units"].get("h_c") is None                    # ホストに残すジョブは止める単位に載らない


def _vps_done(io, tid, nb, delay=5, state="done", date="20261109", task="entries-sync", race="202605040101"):
    args = {"race_id": race, "date": date} if task.startswith("race-") else {}
    write_json(io, f"ops/fetch_tasks/{state}/{tid}.json", {"task_id": tid, "task": task, "args": args, "not_before": nb.isoformat(), "claimed_at": (nb + timedelta(seconds=delay)).isoformat(), "outcome": {"reason": "x"}})


def test_vps_job_uses_claim_delay():
    j = {"id": "entries-sync", "destination": "vps", "enabled": True, "ready": True, "blockers": [], "origin": {"cron_job_id": "entries-sync"}, "fetch_task": {"task": "entries-sync"}, "parity": {"kind": "none"},
         "cutover": {"min_consecutive_success": 3}}
    io = MemoryIO()
    for i in range(3):
        _vps_done(io, f"entries-sync-2026110{i}T1000", NOW - timedelta(hours=i + 1), delay=10)
    assert state_of(doc([j]), "entries-sync", io)[0]["state"] == "合格"
    _vps_done(io, "entries-sync-20261109T1100", NOW - timedelta(minutes=30), delay=400)
    assert "start_delay" in state_of(doc([j]), "entries-sync", io)[0]["reason"]


def test_race_odds_requires_all_races_t60_and_t15():
    j = {"id": "race-odds-per-race", "destination": "vps", "enabled": True, "ready": True, "blockers": [], "origin": {"cron_job_id": "raceday-runner"}, "reserved_by": "reserve-race-day",
         "fetch_task": {"task": "race-odds-t60"}, "extra_tasks": ["race-odds-t15"], "parity": {"kind": "none"}, "cutover": {"raceday_min_days": 2}}
    io = MemoryIO()
    base = NOW - timedelta(days=3)
    for date in ("20261107", "20261108"):
        for race in ("202605040101", "202605040102"):
            for t in ("race-odds-t60", "race-odds-t15"):
                _vps_done(io, f"{t}-{date}-{race}", base, task=t, date=date, race=race)
    assert state_of(doc([j]), "race-odds-per-race", io)[0]["state"] == "合格"
    write_json(io, "state/data/page_reference/race_day_schedule/20261108.json", {"slots": [{}, {}, {}]})       # 3 レース開催のはずが 2 レース分しかない
    r = state_of(doc([j]), "race-odds-per-race", io)[0]
    assert r["state"] == "不合格" and "2/3" in r["reason"]
    write_json(io, "state/data/page_reference/race_day_schedule/20261108.json", {"slots": [{}, {}]})
    _vps_done(io, "race-odds-t15-20261108-202605040102", base, task="race-odds-t15", date="20261108", race="202605040102", state="failed")
    # 同じ id が done と failed の両方にある不整合は失敗として扱う
    assert state_of(doc([j]), "race-odds-per-race", io)[0]["state"] == "不合格"


def test_cut_unit_requires_all_jobs_in_the_unit():
    a = {"id": "dl-m", "destination": "vps", "enabled": True, "ready": True, "blockers": [], "origin": {"cron_job_id": "dl"}, "fetch_task": {"task": "daily-race-lists"}, "parity": {"kind": "none"}, "cutover": {"min_consecutive_success": 1}}
    b = dict(a, id="dl-e")
    io = MemoryIO()
    _vps_done(io, "dl-m-20261109T0700", NOW - timedelta(hours=2), task="daily-race-lists")
    res = CS.evaluate_all(doc([a, b]), CS.Records(io), NOW)
    assert res["cut_units"]["dl"] == {"ready_to_stop": False, "jobs": ["dl-m", "dl-e"], "waiting_on": ["dl-e"]}


def test_real_job_map_evaluates_without_data_and_renders():
    res = CS.evaluate_all(job_map.load(), CS.Records(MemoryIO()), NOW)
    assert res["summary"].get("合格", 0) == 0                     # 記録が無いのに合格するジョブは無い
    assert "ホストの cron を止めてよい" in CS.render_text(res) and "| 状態 |" in CS.render_markdown(res)
    names = {r["job"] for r in res["rows"]}
    assert names == {j["id"] for j in job_map.load()["jobs"]}      # 全ジョブが表に載る


# ── cutover_plan ──

def test_plan_refuses_unpassed_unit_and_host_units(tmp_path):
    assert CP.main(["make", "--unit", "update_jockey_trainer_stats", "--out-dir", str(tmp_path)]) == 3
    assert CP.main(["make", "--unit", "backfill-2026-fast", "--force", "--out-dir", str(tmp_path)]) == 2
    status = tmp_path / "s.json"
    status.write_text(json.dumps({"cut_units": {"update_jockey_trainer_stats": {"ready_to_stop": True, "waiting_on": []}}}))
    assert CP.main(["make", "--unit", "update_jockey_trainer_stats", "--status-file", str(status), "--out-dir", str(tmp_path)]) == 0
    names = sorted(p.name for p in tmp_path.glob("cutover_*"))
    assert names == ["cutover_update_jockey_trainer_stats_1_owner.sh", "cutover_update_jockey_trainer_stats_2_cron_off.sh", "cutover_update_jockey_trainer_stats_rollback.sh"]
    for p in tmp_path.glob("cutover_*"):
        assert subprocess.run(["bash", "-n", str(p)]).returncode == 0 and os.access(p, os.X_OK)


def test_generated_scripts_order_owner_before_cron_and_reverse_for_rollback():
    files = CP.render("daily-race-lists")
    s1, s2, rb = (files[f"cutover_daily-race-lists_{k}.sh"] for k in ("1_owner", "2_cron_off", "rollback"))
    assert "job_owner set daily-race-lists vps" in s1 and "agent_tasks" in s1 and "daily-race-lists" in s1       # 先に担当を渡す（VPS は fetch_owner も）
    assert "host_cron_switch.sh disable --job-id daily-race-lists --tag KEIBA-VPN-ALL" in s2
    assert rb.index("host_cron_switch.sh enable") < rb.index("job_owner unset")                                   # 戻すときは行を戻してから担当を外す
    assert "cron_backups" in s1 and "set -euo pipefail" in s2


def test_multi_part_units_move_both_sides():
    s1 = CP.render("raceday-runner")["cutover_raceday-runner_1_owner.sh"]
    assert "job_owner set raceday-runner gcp" in s1 and "race-odds-t60" in s1 and "race-odds-t02" in s1       # reserve-race-day(GCP) + race-odds(VPS)


# ── job_owner と cron_run.sh ──

def test_owner_switch_roundtrip_and_guards(tmp_path, monkeypatch):
    f = tmp_path / "o.json"
    f.write_text('{"version":1,"owners":{}}')
    monkeypatch.setenv("KEIBA_JOB_OWNER_FILE", str(f))
    monkeypatch.delenv("KEIBA_RUN_LOCATION", raising=False)         # 他のテストが環境に残した値の影響を受けない
    assert job_owner.main(["check", "update_jockey_trainer_stats"]) == 0
    assert job_owner.main(["can-stop", "update_jockey_trainer_stats"]) == 3                 # 担当が渡る前は止められない
    assert job_owner.main(["set", "update_jockey_trainer_stats", "gcp"]) == 0
    assert job_owner.main(["check", "update_jockey_trainer_stats"]) == job_owner.SKIP_EXIT
    assert job_owner.main(["can-stop", "update_jockey_trainer_stats"]) == 0
    monkeypatch.setenv("KEIBA_RUN_LOCATION", "gcp")                                          # 担当側(GCP)では常に実行してよい
    assert job_owner.main(["check", "update_jockey_trainer_stats"]) == 0
    monkeypatch.delenv("KEIBA_RUN_LOCATION")
    assert job_owner.main(["set", "backfill-2026-fast", "gcp"]) == 2                          # ホストに残すジョブは渡せない
    assert job_owner.main(["set", "unknown-job", "gcp"]) == 2
    assert job_owner.main(["unset", "update_jockey_trainer_stats"]) == 0
    assert job_owner.main(["check", "update_jockey_trainer_stats"]) == 0
    f.write_text("{broken")
    assert job_owner.host_must_skip("update_jockey_trainer_stats") is False                  # 壊れていたら誰も奪わない


def test_cron_run_skips_owned_job_but_records_success(tmp_path):
    owner = tmp_path / "o.json"
    owner.write_text('{"owners":{"demo_job":"gcp"}}')
    env = {**{k: v for k, v in os.environ.items() if k != "KEIBA_RUN_LOCATION"}, "KEIBA_CRON_RUNS_DIR": str(tmp_path / "runs"), "KEIBA_JOB_OWNER_FILE": str(owner)}
    r = subprocess.run(["bash", str(ROOT / "scripts/cron/cron_run.sh"), "demo_job", "--", "bash", "-c", "echo SHOULD-NOT-RUN; exit 7"], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "SHOULD-NOT-RUN" not in r.stdout and "GCP / VPS の担当" in r.stdout
    assert (tmp_path / "runs/demo_job.tsv").read_text().split("\t")[2] == "0"               # 監視ポータルには成功として残る
    r2 = subprocess.run(["bash", str(ROOT / "scripts/cron/cron_run.sh"), "other_job", "--", "bash", "-c", "echo RAN; exit 3"], env=env, capture_output=True, text=True)
    assert r2.returncode == 3 and "RAN" in r2.stdout


# ── host_cron_switch.sh（偽の crontab コマンドで） ──

def _fake_cron(tmp_path, content):
    f = tmp_path / "crontab.txt"
    f.write_text(content)
    cmd = tmp_path / "fakecron"
    cmd.write_text('#!/usr/bin/env bash\nF="$FAKE_CRON_FILE"\nif [ "${1:-}" = "-l" ]; then cat "$F"; else cp "$1" "$F"; fi\n')
    cmd.chmod(0o755)
    return f, {**{k: v for k, v in os.environ.items() if k != "KEIBA_RUN_LOCATION"}, "FAKE_CRON_FILE": str(f), "CRONTAB_CMD": str(cmd), "KEIBA_CRON_BACKUP_DIR": str(tmp_path / "bk"), "KEIBA_PROJECT_DIR": str(ROOT)}


CRON = """PATH=/opt/conda/bin:/usr/bin
0 1 * * 1 TZ=Asia/Tokyo bash scripts/cron/cron_run.sh drift-monitor-weekly -- bash scripts/cron/drift_monitor_weekly.sh keiba-vpn  # KEIBA-DRIFT-MONITOR
30 1 * * 1 TZ=Asia/Tokyo bash scripts/cron/cron_run.sh frozen-tracking-weekly -- bash x  # KEIBA-FROZEN-TRACKING
0 22 * * * TZ=Asia/Tokyo bash scripts/cron/cron_run.sh daily-race-lists -- bash y  # KEIBA-VPN-ALL
0 8 * * * TZ=Asia/Tokyo bash scripts/cron/cron_run.sh daily-race-lists -- bash y  # KEIBA-VPN-ALL
"""


def sw(env, *args):
    return subprocess.run(["bash", str(ROOT / "scripts/ops/cutover/host_cron_switch.sh"), *args], env=env, capture_output=True, text=True)


def test_switch_script_disable_enable_roundtrip_with_backup(tmp_path):
    f, env = _fake_cron(tmp_path, CRON)
    r = sw(env, "disable", "--job-id", "drift-monitor-weekly", "--tag", "KEIBA-DRIFT-MONITOR", "--no-owner-check")
    assert r.returncode == 0, r.stderr
    after = f.read_text().splitlines()
    assert after[1].startswith("#KEIBA-CUTOVER:drift-monitor-weekly# 0 1 * * 1") and after[2] == CRON.splitlines()[2]       # 他の行は一文字も変えない
    assert len(list((tmp_path / "bk").glob("crontab_before_cutover_disable_*"))) == 1
    assert "停止中 1" in sw(env, "status", "--job-id", "drift-monitor-weekly").stdout
    assert sw(env, "enable", "--job-id", "drift-monitor-weekly").returncode == 0
    assert f.read_text() == CRON                                                                                                # 完全に元どおり


def test_switch_script_guards(tmp_path):
    f, env = _fake_cron(tmp_path, CRON)
    assert sw(env, "disable", "--job-id", "drift-monitor-weekly", "--tag", "KEIBA-DRIFT-MONITOR").returncode == 5               # 担当が渡る前は止めない
    assert sw(env, "disable", "--job-id", "nothing-here", "--no-owner-check").returncode == 3
    assert sw(env, "disable", "--job-id", "daily-race-lists", "--max-lines", "1", "--no-owner-check").returncode == 4         # 想定より多い行は止めない
    assert sw(env, "enable", "--job-id", "drift-monitor-weekly").returncode == 3
    assert sw(env, "disable", "--job-id", "drift;rm", "--no-owner-check").returncode == 2
    assert f.read_text() == CRON                                                                                                # 失敗時は crontab を変えない
    d = sw(env, "disable", "--job-id", "daily-race-lists", "--no-owner-check", "--dry-run")
    assert d.returncode == 0 and "dry-run" in d.stdout and f.read_text() == CRON


def test_switch_script_with_real_owner_check(tmp_path):
    f, env = _fake_cron(tmp_path, CRON)
    owner = tmp_path / "o.json"
    owner.write_text('{"owners":{}}')
    env["KEIBA_JOB_OWNER_FILE"] = str(owner)
    assert sw(env, "disable", "--job-id", "drift-monitor-weekly").returncode == 5
    job_owner_set = subprocess.run(["python3", "-m", "src.ops.job_owner", "set", "drift-monitor-weekly", "gcp"], env={**env, "PYTHONPATH": str(ROOT)}, cwd=ROOT, capture_output=True, text=True)
    assert job_owner_set.returncode == 0, job_owner_set.stderr
    assert sw(env, "disable", "--job-id", "drift-monitor-weekly").returncode == 0
