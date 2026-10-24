"""VPS の systemd ユニット・スクリプトの静的検査（実機なし）。メモリ予算が architecture_target.md 3.2 の表とずれないことを守る。"""
import configparser
import re
import subprocess
from pathlib import Path

T = Path("scripts/ops/vps/templates")


def load(name):
    cp = configparser.ConfigParser(strict=False, interpolation=None)
    cp.optionxform = str
    cp.read(T / name, encoding="utf-8")
    return cp


def mb(s):
    m = re.fullmatch(r"(\d+)M", s)
    assert m, s
    return int(m.group(1))


def test_memory_budget_matches_design():
    caps = {
        "agent": load("keiba-fetch-agent.service")["Service"],
        "slice": load("keiba-fetch.slice")["Slice"],
        "nginx": load("dropin-nginx-memory.conf")["Service"],
        "cf": load("dropin-cloudflared-prod-memory.conf")["Service"],
        "ui": load("dropin-keiba-ui-memory.conf")["Service"],
    }
    assert (mb(caps["agent"]["MemoryHigh"]), mb(caps["agent"]["MemoryMax"])) == (220, 250)
    assert (mb(caps["slice"]["MemoryHigh"]), mb(caps["slice"]["MemoryMax"])) == (220, 250)
    assert (mb(caps["nginx"]["MemoryMax"]), mb(caps["cf"]["MemoryMax"]), mb(caps["ui"]["MemoryMax"])) == (48, 128, 200)
    for k, v in caps.items():
        assert mb(v["MemoryHigh"]) < mb(v["MemoryMax"]), k
    assert sum(mb(caps[k]["MemoryMax"]) for k in ("agent", "nginx", "cf", "ui")) <= 640   # OS 約 400MB と合わせて約 1.0GB


def test_agent_unit_is_hardened_and_off_by_default():
    svc = load("keiba-fetch-agent.service")
    s = svc["Service"]
    assert s["OOMScoreAdjust"] == "500" and s["Slice"] == "keiba-fetch.slice" and s["Restart"] == "on-failure"
    assert "KEIBA_ENV=prod" in s["Environment"] and "KEIBA_RUN_LOCATION=vps" in s["Environment"]
    assert s["NoNewPrivileges"] == "true" and s["ProtectSystem"] == "strict"
    text = (T / "keiba-fetch-agent.service").read_text(encoding="utf-8")
    assert "PASSWORD" not in text and "SECRET" not in text      # 秘密はユニットに書かない


def test_pull_timers_sit_inside_maintenance_window():
    for kind, hhmm in (("ui", "03:10"), ("fetch", "03:20")):
        t = load(f"keiba-release-pull-{kind}.timer")
        assert t["Timer"]["OnCalendar"].startswith(f"*-*-* {hhmm}")
        assert t["Timer"]["Unit"] == f"keiba-release-pull@{kind}.service"
    assert "--kind %i" in (T / "keiba-release-pull@.service").read_text(encoding="utf-8")


def test_shell_scripts_parse():
    for f in ("vps_03_units.sh", "measure_memory.sh", "publish_release.sh", "vps_01_base_setup.sh"):
        assert subprocess.run(["bash", "-n", f"scripts/ops/vps/{f}"]).returncode == 0, f


def test_vps_03_dry_run_changes_nothing():
    r = subprocess.run(["bash", "scripts/ops/vps/vps_03_units.sh", "--dry-run"], capture_output=True, text=True)
    assert r.returncode == 0 and "完了" in r.stdout
    assert "eyJ" not in r.stdout
