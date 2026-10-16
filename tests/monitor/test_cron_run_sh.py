"""scripts/cron/cron_run.sh（cron の実行記録ラッパー）のテスト。実際に bash で動かす。"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "cron" / "cron_run.sh"
pytestmark = pytest.mark.skipif(not Path("/bin/bash").exists(), reason="bash が必要")


def run(job, args, runs_dir, **kw):
    env = {**os.environ, "KEIBA_CRON_RUNS_DIR": str(runs_dir)}
    return subprocess.run(["bash", str(SCRIPT), job, "--", *args], capture_output=True, text=True, env=env, **kw)


def test_success_is_recorded_and_output_passes_through(tmp_path):
    r = run("okjob", ["bash", "-c", "echo hello; echo oops >&2"], tmp_path)
    assert r.returncode == 0 and "hello" in r.stdout and "oops" in r.stdout
    s, e, ec, _pid = (tmp_path / "okjob.tsv").read_text().strip().split("\t")
    assert ec == "0" and int(e) >= int(s)
    assert "hello" in (tmp_path / "okjob.tail.txt").read_text()
    assert not (tmp_path / "okjob.running").exists() and not (tmp_path / "okjob.fail.txt").exists()


def test_failure_keeps_the_exit_code_and_saves_the_tail(tmp_path):
    r = run("badjob", ["bash", "-c", "for i in $(seq 1 200); do echo line$i; done; exit 7"], tmp_path)
    assert r.returncode == 7
    tail = (tmp_path / "badjob.tail.txt").read_text().splitlines()
    assert len(tail) == 80 and tail[-1] == "line200"
    fail = (tmp_path / "badjob.fail.txt").read_text().splitlines()
    assert fail[0].endswith("exit=7") and fail[-1] == "line200"
    assert (tmp_path / "badjob.tsv").read_text().strip().split("\t")[2] == "7"


def test_failure_note_survives_a_later_success(tmp_path):
    run("j", ["bash", "-c", "echo boom; exit 1"], tmp_path)
    run("j", ["bash", "-c", "echo fine"], tmp_path)
    assert "boom" in (tmp_path / "j.fail.txt").read_text() and "fine" in (tmp_path / "j.tail.txt").read_text()
    assert len((tmp_path / "j.tsv").read_text().splitlines()) == 2


def test_running_marker_exists_while_running_and_is_removed_after(tmp_path):
    env = {**os.environ, "KEIBA_CRON_RUNS_DIR": str(tmp_path)}
    p = subprocess.Popen(["bash", str(SCRIPT), "slow", "--", "sleep", "3"], env=env)
    try:
        for _ in range(50):
            if (tmp_path / "slow.running").exists():
                break
            time.sleep(0.1)
        pid, start = (tmp_path / "slow.running").read_text().split()
        assert int(pid) == p.pid and int(start) > 0
    finally:
        p.terminate()
        p.wait(timeout=10)
    assert not (tmp_path / "slow.running").exists()


def test_sigterm_is_forwarded_and_recorded_as_143(tmp_path):
    env = {**os.environ, "KEIBA_CRON_RUNS_DIR": str(tmp_path)}
    p = subprocess.Popen(["bash", str(SCRIPT), "term", "--", "sleep", "30"], env=env)
    for _ in range(50):
        if (tmp_path / "term.running").exists():
            break
        time.sleep(0.1)
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=10) == 143
    assert (tmp_path / "term.tsv").read_text().strip().split("\t")[2] == "143"
    assert not (tmp_path / "term.running").exists()


def test_unwritable_runs_dir_does_not_stop_the_command(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    r = run("x", ["bash", "-c", "echo still-runs; exit 3"], blocker / "sub")
    assert r.returncode == 3 and "still-runs" in r.stdout


def test_tsv_is_kept_small(tmp_path):
    (tmp_path / "many.tsv").write_text("".join(f"1\t2\t0\t{i}\n" for i in range(400)))
    run("many", ["true"], tmp_path)
    lines = (tmp_path / "many.tsv").read_text().splitlines()
    assert len(lines) <= 261 and lines[-1].split("\t")[2] == "0"
    run("many", ["true"], tmp_path)
    assert len((tmp_path / "many.tsv").read_text().splitlines()) <= 261


def test_job_id_is_made_safe_for_file_names(tmp_path):
    run("a/b c", ["true"], tmp_path)
    assert (tmp_path / "a_b_c.tsv").exists()


def test_missing_command_is_a_usage_error(tmp_path):
    r = subprocess.run(["bash", str(SCRIPT), "x"], capture_output=True, text=True, env={**os.environ, "KEIBA_CRON_RUNS_DIR": str(tmp_path)})
    assert r.returncode == 2
