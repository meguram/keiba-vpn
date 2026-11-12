"""毎日の定時ジョブの追いつき実行（src/ops/cron_catchup.py）: 逃した予定の分だけ 1 回走り、二度は走らないこと。"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from src.ops import cron_catchup as cc

ROOT = Path(__file__).resolve().parents[2]
CRONTAB = """SHELL=/bin/bash
PATH=/opt/conda/bin:/usr/bin:/bin
0 22 * * * bash /x/scripts/cron/cron_run.sh export_tables_daily -- /x/scripts/cron/export_tables_daily.sh /x # KEIBA-VPN-ALL
40 22 * * * cd /x && bash /x/scripts/cron/cron_run.sh feature-accumulate -- bash /x/scripts/cron/run_auto_scrape_logged.sh /x feature-accumulate logs/f.log # KEIBA-VPN-ALL
30 22 * * * cd /x && bash /x/scripts/cron/cron_run.sh raceday-runner -- bash /x/r.sh # KEIBA-VPN-ALL
0 8 * * * bash /x/scripts/cron/cron_run.sh daily-race-lists -- bash /x/a.sh evening # KEIBA-VPN-ALL
PATH='/late/bin'
0 22 * * * bash /x/scripts/cron/cron_run.sh daily-race-lists -- bash /x/a.sh morning # KEIBA-VPN-ALL
"""
CFG = {"jobs": ["export_tables_daily", "feature-accumulate", "daily-race-lists"], "max_late_hours": 18, "grace_min": 20, "settle_min": 5}
NOW = datetime(2026, 11, 12, 0, 30, tzinfo=timezone.utc)           # 09:30 JST。予定は 07:00・07:40 JST
BOOT = datetime(2026, 11, 11, 23, 46, tzinfo=timezone.utc).timestamp()


def _run(tmp_path, *, now=NOW, cron_started=BOOT, down=(), runner=None, ps=lambda j: False, cfg=CFG):
    calls = []
    items = cc.run(now, crontab_text=CRONTAB, cfg=cfg, runs_dir=tmp_path / "runs", mark_dir=tmp_path / "mark",
                   cron_started=cron_started, down=list(down), boot=BOOT - 60, ps=ps,
                   runner=runner or (lambda it: calls.append(it["id"]) or 0), log=lambda m: None)
    return items, calls


def test_missed_jobs_run_once_in_schedule_order_with_crontab_env(tmp_path):
    (tmp_path / "runs").mkdir()
    seen = []
    items, _ = _run(tmp_path, runner=lambda it: seen.append(it) or 0)
    assert [i["id"] for i in items] == ["export_tables_daily", "daily-race-lists", "feature-accumulate"]   # 許可リストの外（raceday-runner）は走らせない
    env = cc.job_env(seen[0])
    assert env["PATH"] == "/opt/conda/bin:/usr/bin:/bin" and env["SHELL"] == "/bin/bash" and "PYTHONPATH" not in env
    assert "cron_run.sh export_tables_daily" in seen[0]["cmd"] and "KEIBA" not in seen[0]["cmd"]
    drl = seen[1]                                                   # 1 日 2 回の行は直前の予定（22:00 UTC）の行のコマンドと、その行より前の環境
    assert drl["cmd"].endswith("morning") and drl["env"]["PATH"] == "/late/bin"
    items, calls = _run(tmp_path)                                    # 同じ予定の分は二度と走らせない（印）
    assert items == [] and calls == []
    mark = json.loads((tmp_path / "mark" / "export_tables_daily.json").read_text())
    assert mark["pf"] == "2026-11-11T22:00:00+00:00" and mark["rc"] == 0


def test_failed_catchup_is_not_retried_and_one_failure_does_not_stop_the_rest(tmp_path):
    (tmp_path / "runs").mkdir()

    def boom(it):
        if it["id"] == "export_tables_daily":
            raise OSError("x")
        return 1

    items, _ = _run(tmp_path, runner=boom)
    assert [i["rc"] for i in items] == [-1, 1, 1]
    assert _run(tmp_path) == ([], [])


def test_rechecks_right_before_each_job(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()

    def runner(it):                                                  # 1 件目を実行している間に、本来の cron が 2 件目を始めた
        if it["id"] == "export_tables_daily":
            (runs / "daily-race-lists.tsv").write_text(f"{NOW.timestamp()}\t{NOW.timestamp() + 5}\t0\t1\n")
        return 0

    items, _ = _run(tmp_path, runner=runner)
    assert [i["id"] for i in items] == ["export_tables_daily", "feature-accumulate"]


def test_not_run_when_cron_was_up_or_job_already_started_or_running(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    early_boot = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc).timestamp()
    assert _run(tmp_path, cron_started=early_boot)[0] == []         # cron は動いていた → 追いつかない（未開始は通知で人が見る）
    pf = datetime(2026, 11, 11, 22, 0, tzinfo=timezone.utc).timestamp()
    cfg1 = {**CFG, "jobs": ["export_tables_daily"]}
    assert [i["id"] for i in _run(tmp_path, cron_started=early_boot, down=[(pf - 600, pf + 600)], cfg=cfg1, runner=lambda it: 0)[0]] == ["export_tables_daily"]
    for p in (tmp_path / "mark").iterdir():
        p.unlink()
    (runs / "export_tables_daily.tsv").write_text(f"{pf + 5}\t{pf + 200}\t0\t1\n")   # 予定どおり始まっている
    (runs / "feature-accumulate.running").write_text(f"1 {BOOT}\n")                 # PID 1 は生きている＝実行中
    assert [i["id"] for i in _run(tmp_path)[0]] == ["daily-race-lists"]
    assert _run(tmp_path, ps=lambda j: True)[0] == []              # ps で cron_run.sh <job> が起動中なら走らせない


def test_stale_running_marker_from_before_the_boot_is_ignored(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "export_tables_daily.running").write_text(f"1 {BOOT - 3600}\n")   # PC の停止で残った印（PID 1 は再利用されて生きている）
    assert "export_tables_daily" in [i["id"] for i in _run(tmp_path, cfg={**CFG, "jobs": ["export_tables_daily"]})[0]]


def test_respects_grace_settle_and_jst_day(tmp_path):
    (tmp_path / "runs").mkdir()
    assert _run(tmp_path, now=datetime(2026, 11, 11, 22, 10, tzinfo=timezone.utc), cron_started=BOOT - 7200)[0] == []   # 予定の 10 分後: まだ待つ
    assert _run(tmp_path, now=datetime(2026, 11, 11, 23, 48, tzinfo=timezone.utc))[0] == []   # cron の起動から 2 分: 落ち着くまで待つ
    late = datetime(2026, 11, 12, 15, 30, tzinfo=timezone.utc)      # 翌 00:30 JST: 日付が変わったので翌日の予定に任せる
    assert _run(tmp_path, now=late)[0] == []


def test_crontab_env_parsing():
    ents = cc.crontab_entries("FOO = 'a b'\nlower=x\n0 1 * * * bash /x/scripts/cron/cron_run.sh j1 -- echo 1\n")
    assert ents[0]["env"] == {"FOO": "a b", "lower": "x"}


def test_config_jobs_are_daily_and_registered_by_repo_scripts():
    import subprocess

    cfg = cc.load_config()
    show = subprocess.run(["bash", str(ROOT / "scripts/cron/setup_all_cron.sh"), "show"], capture_output=True, text=True, check=True).stdout
    scripts = show + "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "scripts/cron").glob("install_*.sh"))
    for job in cfg["jobs"]:
        assert re.search(rf"cron_run\.sh {re.escape(job)} --", scripts), f"{job} を登録する行がリポジトリに無い"
        assert not job.startswith("raceday-")                     # 開催の時刻に合わせて動くものは入れない
    for e in cc.crontab_entries(show):
        if e["id"] in cfg["jobs"]:
            assert re.fullmatch(r"\d+ \d+ \* \* \*", e["expr"]), (e["id"], e["expr"])   # 毎日の定時だけ


def test_install_script_line_and_safety():
    body = (ROOT / "scripts/cron/install_cron_catchup.sh").read_text(encoding="utf-8")
    assert "cron_run.sh cron-catchup --" in body and "src.ops.cron_catchup run" in body and "*/15 * * * *" in body
    assert "crontab -l に失敗したので中止" in body                     # 読み取りに失敗したら crontab 全体を置き換えない
