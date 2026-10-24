"""setup_all_cron.sh が作る全 cron 行が cron_run.sh 経由で、渡す job_id が監視側の算出（derive_job_id）と一致するかのテスト。"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from src.monitor.cron_status import _lines, derive_job_id

ROOT = Path(__file__).resolve().parents[2]


def _generated() -> str:
    return subprocess.run(["bash", str(ROOT / "scripts/cron/setup_all_cron.sh"), "show"], capture_output=True, text=True, check=True).stdout


def test_every_generated_cron_line_goes_through_cron_run():
    lines = [ln for ln in _generated().splitlines() if re.match(r"^(@reboot|[\d*/,\-]+\s+[\d*/,\-]+\s+[\d*/,\-]+\s+[\d*/,\-]+\s+[\d*/,\-]+\s+)", ln.strip()) and not ln.lstrip().startswith("#")]
    assert len(lines) == 38
    assert [ln for ln in lines if "cron_run.sh" not in ln] == []


def test_job_id_passed_to_cron_run_equals_the_derived_one():
    parsed = _lines(_generated())
    assert len(parsed) == 38
    for ln in parsed:
        explicit = re.search(r"cron_run\.sh\s+(\S+)\s+--", ln["cmd"]).group(1)
        assert explicit == derive_job_id(ln["cmd"]) == ln["id"], ln["line"]
        assert re.fullmatch(r"[A-Za-z0-9._\-]+", explicit)


def test_known_ids_cover_the_whole_schedule():
    ids = {ln["id"] for ln in _lines(_generated())}
    assert {"daily-race-lists", "raceday-eve", "weekly-update", "jra-baba-morning", "export_tables_daily", "sync_pg_from_gcs", "server_watchdog",
            "ensure_postgres", "rotate_logs", "page_quality_check", "update_jockey_trainer_stats", "backfill-2026-fast", "backfill-all-horse"} <= ids


def test_setup_script_still_has_the_markers_the_test_command_checks():
    out = _generated()
    for needle in ("PATH=/opt/conda", "raceday-runner", "raceday-result-runner", "raceday-evening", "weekly-update", "jra-baba-morning", "KEIBA-VPN-ALL", "KEIBA_BACKFILL"):
        assert needle in out
    assert re.search(r"30 22.*raceday-runner", out) and "git_pull_hourly" not in out
