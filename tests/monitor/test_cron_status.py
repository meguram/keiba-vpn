"""cron_status（監視ポータルの cron 状態表示のもと）のテスト。合成の crontab・実行記録・ログで確かめる。"""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.monitor import cron_status as C

UTC = timezone.utc
JST = timezone(timedelta(hours=9))
ROOT = Path(__file__).resolve().parents[2]

CRONTAB = """\
PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin
# コメント
0 22 * * * cd /p && TZ=Asia/Tokyo bash /p/scripts/cron/cron_run.sh daily-race-lists -- bash /p/scripts/cron/yield_to_rescrape.sh daily-race-lists -- bash /p/scripts/cron/run_auto_scrape_logged.sh /p daily-race-lists logs/daily_race_lists_am.log # KEIBA-VPN-ALL
0 8 * * * cd /p && TZ=Asia/Tokyo bash /p/scripts/cron/cron_run.sh daily-race-lists -- bash /p/scripts/cron/yield_to_rescrape.sh daily-race-lists -- bash /p/scripts/cron/run_auto_scrape_logged.sh /p daily-race-lists logs/daily_race_lists_pm.log # KEIBA-VPN-ALL
30 22 * * * cd /p && TZ=Asia/Tokyo bash /p/scripts/cron/run_auto_scrape_logged.sh /p raceday-runner logs/raceday_runner.log # KEIBA-VPN-ALL
0 9 * * 5 cd /p && bash /p/scripts/cron/cron_run.sh weekly-update -- bash /p/scripts/cron/run_auto_scrape_logged.sh /p weekly-update logs/weekly_update.log # KEIBA-VPN-ALL
*/3 * * * * bash /p/scripts/cron/cron_run.sh server_watchdog -- /p/scripts/server/server_watchdog.sh # KEIBA-VPN-WATCHDOG
@reboot sleep 15 && bash /p/scripts/cron/cron_run.sh server_watchdog -- /p/scripts/server/server_watchdog.sh # KEIBA-VPN-WATCHDOG
0 15 * * *   cd /p && bash /p/scripts/cron/cron_run.sh backfill-2026-fast -- bash /p/scripts/cron/yield_to_rescrape.sh backfill -- /opt/conda/bin/python3 -m src.scraper.backfill --year 2026 --phase fast --max-dates 7 >> /p/logs/backfill_2026.log 2>&1 # KEIBA_BACKFILL
"""
NOW = datetime(2026, 10, 16, 8, 0, tzinfo=UTC)          # 金曜 17:00 JST


def status(tmp_path, ps=None, **kw):
    d = C.build_status(CRONTAB, runs_dir=tmp_path / "runs", log_dir=tmp_path / "logs", now=NOW, server_tz=UTC, ps_lines=ps if ps is not None else [], **kw)
    return d, {j["id"]: j for g in d["groups"] for j in g["jobs"]}


def record(runs, job, start, end, ec, tail="", fail=None):
    runs.mkdir(parents=True, exist_ok=True)
    with open(runs / f"{job}.tsv", "a") as f:
        f.write(f"{int(start.timestamp())}\t{int(end.timestamp())}\t{ec}\t123\n")
    (runs / f"{job}.tail.txt").write_text(tail)
    if fail:
        (runs / f"{job}.fail.txt").write_text(fail)


# ── cron 式 ──

@pytest.mark.parametrize("expr,after,expected", [
    ("*/10 20-23 * * *", datetime(2026, 10, 16, 8, 0, tzinfo=UTC), datetime(2026, 10, 16, 20, 0, tzinfo=UTC)),
    ("0 9 * * 5", datetime(2026, 10, 16, 8, 0, tzinfo=UTC), datetime(2026, 10, 16, 9, 0, tzinfo=UTC)),          # 同じ金曜のあと 1 時間
    ("0 9 * * 5", datetime(2026, 10, 16, 9, 0, tzinfo=UTC), datetime(2026, 10, 23, 9, 0, tzinfo=UTC)),          # ちょうどの時刻は含まず、翌週
    ("0 17 * * 1,4", datetime(2026, 10, 16, 8, 0, tzinfo=UTC), datetime(2026, 10, 19, 17, 0, tzinfo=UTC)),       # 月・木
    ("30 22 * * *", datetime(2026, 10, 31, 23, 0, tzinfo=UTC), datetime(2026, 11, 1, 22, 30, tzinfo=UTC)),       # 月またぎ
    ("0 0 31 * *", datetime(2026, 11, 1, 0, 0, tzinfo=UTC), None),                                                # 8 日先までに無い
    ("0 12 * * 0", datetime(2026, 10, 16, 8, 0, tzinfo=UTC), datetime(2026, 10, 18, 12, 0, tzinfo=UTC)),         # 日曜 = 0
    ("0 12 * * 7", datetime(2026, 10, 16, 8, 0, tzinfo=UTC), datetime(2026, 10, 18, 12, 0, tzinfo=UTC)),         # 日曜 = 7
])
def test_next_fire(expr, after, expected):
    assert C.next_fire(expr, after) == expected


def test_next_run_is_returned_in_jst(tmp_path):
    _, j = status(tmp_path)
    assert j["weekly-update"]["next_run"] == "2026-10-16T18:00:00+09:00"           # UTC 09:00 金 = JST 18:00 金
    assert j["raceday-runner"]["next_run"] == "2026-10-17T07:30:00+09:00"          # 今日の UTC 22:30 = JST 翌日 07:30


# ── 一覧 ──

def test_jobs_with_the_same_id_are_merged_and_schedule_is_in_jst(tmp_path):
    d, j = status(tmp_path)
    assert set(j) == {"daily-race-lists", "raceday-runner", "weekly-update", "server_watchdog", "backfill-2026-fast"}
    assert j["daily-race-lists"]["schedule"] == "毎日 07:00・17:00 JST" and len(j["daily-race-lists"]["cron_exprs"]) == 2
    assert j["weekly-update"]["schedule"] == "金 18:00 JST"
    assert "システム起動時" in j["server_watchdog"]["schedule"]
    assert j["daily-race-lists"]["yields"] and not j["raceday-runner"]["yields"]
    assert d["groups"][0]["key"] == "live" and any(g["key"] == "backfill" for g in d["groups"])


def test_job_ids_match_the_wrapper_argument():
    for ln in C._lines(CRONTAB):
        assert C.derive_job_id(ln["cmd"]) == ln["id"], ln["line"]
    assert C.derive_job_id("TZ=Asia/Tokyo /opt/conda/bin/python3 -m src.monitor.page_quality_check >> x.log 2>&1") == "page_quality_check"
    assert C.derive_job_id("/p/scripts/cron/export_tables_daily.sh /p") == "export_tables_daily"
    assert C.derive_job_id("python -m src.scraper.backfill --phase horse") == "backfill-all-horse"


# ── 状態と結果 ──

def test_without_any_record_the_job_is_waiting_with_unknown_result(tmp_path):
    d, j = status(tmp_path)
    assert j["weekly-update"]["state"] == "waiting" and j["weekly-update"]["last_run"] is None
    assert d["summary"]["waiting"] == 5 and d["summary"]["running"] == 0 and d["summary"]["unknown"] == 5


def test_success_failure_warning_and_skipped_from_records(tmp_path):
    runs = tmp_path / "runs"
    t0 = NOW - timedelta(hours=3)
    record(runs, "weekly-update", t0, t0 + timedelta(minutes=5), 0, "all good\n")
    record(runs, "raceday-runner", t0, t0 + timedelta(seconds=9), 1, "Traceback (most recent call last):\nModuleNotFoundError: No module named 'bs4'\n",
           fail="# 2026-10-16T05:00:00+09:00 exit=1\nModuleNotFoundError: No module named 'bs4'\n")
    record(runs, "backfill-2026-fast", t0, t0 + timedelta(seconds=1), 0, "[2026-10-16T02:00:01+09:00] SKIP backfill: 優先処理(取り直し)が動いているため見送り\n")
    record(runs, "daily-race-lists", t0, t0 + timedelta(seconds=30), 0, "PARSE_AUDIT_FAIL: 必須項目の欠損があります\n")
    d, j = status(tmp_path)
    assert j["weekly-update"]["last_run"]["result"] == "success" and j["weekly-update"]["cause"] is None
    assert j["weekly-update"]["last_run"]["duration_sec"] == 300 and j["weekly-update"]["last_run"]["source"] == "record"
    f = j["raceday-runner"]
    assert f["last_run"]["result"] == "failed" and f["cause"]["pattern_id"] == "missing_module" and "bs4" in f["cause"]["label"]
    assert f["last_failure"]["cause"]["pattern_id"] == "missing_module" and f["last_failure"]["at"] == "2026-10-16T05:00:00+09:00"
    assert j["backfill-2026-fast"]["last_run"]["result"] == "skipped" and j["backfill-2026-fast"]["cause"]["kind"] == "skipped"
    assert j["daily-race-lists"]["last_run"]["result"] == "warning"
    assert (d["summary"]["success"], d["summary"]["failed"], d["summary"]["skipped"], d["summary"]["warning"]) == (1, 1, 1, 1)
    assert f["output_tail"][-1].startswith("ModuleNotFoundError")


def test_last_failure_is_remembered_after_a_later_success(tmp_path):
    runs = tmp_path / "runs"
    record(runs, "weekly-update", NOW - timedelta(hours=1), NOW, 0, "fine\n", fail="# 2026-10-09T18:00:00+09:00 exit=143\nstopped\n")
    _, j = status(tmp_path)
    assert j["weekly-update"]["last_run"]["result"] == "success"
    assert j["weekly-update"]["last_failure"]["cause"]["pattern_id"] == "sigterm"


def test_running_job_is_detected_from_the_marker_and_stale_marker_is_ignored(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    p = subprocess.Popen(["bash", "-c", "sleep 20; echo cron_run.sh raceday-runner"])
    try:
        (runs / "raceday-runner.running").write_text(f"{p.pid} {int(time.time()) - 600}\n")
        (runs / "weekly-update.running").write_text("999999 1\n")                    # 存在しない pid（クラッシュで残った）
        _, j = status(tmp_path)
        assert j["raceday-runner"]["state"] == "running" and j["raceday-runner"]["running_pid"] == p.pid and j["raceday-runner"]["running_since"]
        assert j["weekly-update"]["state"] == "waiting"
    finally:
        p.kill()
        p.wait()


def test_job_without_records_is_found_by_ps(tmp_path):
    ps = [(4242, "/opt/conda/bin/python3 -m src.scraper.backfill --year 2026 --phase fast --max-dates 7"), (4243, "python -m src.scraper.auto_scrape --task weekly-update")]
    _, j = status(tmp_path, ps=ps)
    assert j["backfill-2026-fast"]["state"] == "running" and j["backfill-2026-fast"]["running_pid"] == 4242
    assert j["weekly-update"]["state"] == "running"
    assert j["raceday-runner"]["state"] == "waiting"


# ── ログへの切り替え ──

def test_falls_back_to_the_cron_boundary_lines_in_the_log(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "raceday_runner.log").write_text(
        "===== 2026-10-15T07:30:01+09:00 cron START task=raceday-runner pid=11 cwd=/p =====\nok\n"
        "===== 2026-10-15T07:30:03+09:00 cron END   task=raceday-runner exit=0 =====\n\n"
        "===== 2026-10-16T07:30:01+09:00 cron START task=raceday-runner pid=12 cwd=/p =====\n"
        "2026-10-16 07:30:02 [ERROR] scraper.jra_calendar: 9月取得失敗: HTTPSConnectionPool(host='www.jra.go.jp', port=443)\n"
        "===== 2026-10-16T07:30:05+09:00 cron END   task=raceday-runner exit=1 =====\n")
    _, j = status(tmp_path)
    r = j["raceday-runner"]
    assert r["last_run"]["source"] == "log" and r["last_run"]["exit_code"] == 1 and r["last_run"]["result"] == "failed"
    assert r["last_run"]["started_at"] == "2026-10-16T07:30:01+09:00" and r["last_run"]["duration_sec"] == 4
    assert r["cause"]["pattern_id"] == "jra_unreachable" and r["log_path"] == "logs/raceday_runner.log"
    assert r["last_failure"]["at"] == "2026-10-16T07:30:05+09:00"


def test_log_without_boundaries_gives_only_the_modification_time(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "backfill_2026.log").write_text("something\n")
    _, j = status(tmp_path)
    lr = j["backfill-2026-fast"]["last_run"]
    assert lr["source"] == "mtime" and lr["result"] == "unknown" and lr["exit_code"] is None


def test_record_wins_over_the_log(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "raceday_runner.log").write_text("===== 2026-10-15T07:30:03+09:00 cron END   task=raceday-runner exit=1 =====\n")
    record(tmp_path / "runs", "raceday-runner", NOW - timedelta(minutes=5), NOW, 0, "ok\n")
    _, j = status(tmp_path)
    assert j["raceday-runner"]["last_run"]["source"] == "record" and j["raceday-runner"]["last_run"]["result"] == "success"


def test_api_shape_is_json_serialisable(tmp_path):
    import json
    d, _ = status(tmp_path)
    json.dumps(d)
    assert set(d["summary"]) == {"running", "waiting", "success", "failed", "warning", "skipped", "unknown"}
    j = d["groups"][0]["jobs"][0]
    for k in ("id", "name", "desc", "dest", "schedule", "cron_exprs", "yields", "next_run", "state", "running_since", "running_pid",
              "last_run", "cause", "last_failure", "output_tail", "log_path"):
        assert k in j


def test_crontab_source_falls_back_to_the_planned_one(monkeypatch):
    monkeypatch.delenv("KEIBA_CRONTAB_FILE", raising=False)
    monkeypatch.setattr(C.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout="" if cmd[0] == "crontab" else "0 9 * * 5 bash /p/scripts/cron/cron_run.sh weekly-update -- x\n"))
    text, src = C.load_crontab_text()
    assert "weekly-update" in text and src.startswith("setup_all_cron.sh show")
    assert os.environ.get("KEIBA_CRONTAB_FILE") is None


def test_cron_timezone_ignores_the_process_tz_env(monkeypatch):
    """監視サーバは TZ=Asia/Tokyo で動くが、cron のデーモンはシステムの時刻(UTC)で動く。
    プロセスの TZ を使うと『次回の実行』が 9 時間ずれた（2026-10-17: jra-baba-morning の次回が 20:00 と出た）。"""
    monkeypatch.setenv("TZ", "Asia/Tokyo")
    time.tzset()
    monkeypatch.setenv("KEIBA_CRON_TZ", "UTC")
    try:
        assert C.cron_timezone().key == "UTC"
        assert datetime.now().astimezone().utcoffset() == timedelta(hours=9)       # プロセスの時刻は日本時間
        # TZ を使わない評価: UTC の金曜 15:45 から見て、'0 22 * * *'（UTC 22:00 = JST 07:00）は 6 時間 15 分後
        after = datetime(2026, 10, 16, 15, 45, tzinfo=C.cron_timezone())
        nxt = C.next_fire("0 22 * * *", after)
        assert nxt.astimezone(JST).strftime("%m/%d %H:%M") == "10/17 07:00"
        monkeypatch.setenv("KEIBA_CRON_TZ", "Asia/Tokyo")
        assert C.cron_timezone().key == "Asia/Tokyo"
    finally:
        monkeypatch.undo()
        time.tzset()
