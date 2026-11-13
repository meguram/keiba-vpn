"""段 1: VPS が取ったレース一覧・時刻表を ops/mirror_vps/ 経由でホストに渡す（ユニット・cron）。"""

from __future__ import annotations

import configparser
import subprocess
from pathlib import Path

T = Path("scripts/ops/vps/templates")


def _unit(name):
    cp = configparser.ConfigParser(strict=False, interpolation=None)
    cp.optionxform = str
    cp.read(T / name, encoding="utf-8")
    return cp


def test_push_unit_runs_push_vps_in_fetch_slice():
    sv = _unit("keiba-mirror-vps-push.service")["Service"]
    assert sv["ExecStart"].endswith("-m src.ops.gcs_mirror push-vps") and sv["User"] == "keibafetch"
    assert sv["Slice"] == "keiba-fetch.slice" and int(sv["MemoryMax"].rstrip("M")) <= 120
    assert _unit("keiba-mirror-vps-push.timer")["Timer"]["Unit"] == "keiba-mirror-vps-push.service"


def test_host_pull_cron_script_parses_and_gates_by_owner():
    s = Path("scripts/cron/install_vps_mirror_pull_cron.sh").read_text(encoding="utf-8")
    assert subprocess.run(["bash", "-n", "scripts/cron/install_vps_mirror_pull_cron.sh"]).returncode == 0
    assert "pull-vps --trees race_lists,race_day_schedule" in s and "cushion" not in s.split("LINES=")[1] and "--force" not in s
