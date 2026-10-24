"""scripts/ops/cron_prune_apply.sh: 完了済みのバックフィル等を止め、2026 の補填を週1にする。他の行は変えない。"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/ops/cron_prune_apply.sh"


def _line(sched: str, name: str) -> str:
    return f"{sched} cd /x && TZ=Asia/Tokyo bash /x/scripts/cron/cron_run.sh {name} -- bash /x/run.sh # TAG"


SAMPLE = "\n".join([
    "PATH=/usr/bin",
    _line("0 16 * * *  ", "backfill-2025-fast"),
    _line("0 15 * * *  ", "backfill-2026-fast"),
    _line("50 22 * * * ", "backfill-2026-full"),
    _line("0 17 * * 1,4", "backfill-2021-fast"),
    _line("0 21 * * *  ", "backfill-all-horse"),
    _line("0 18 * * *  ", "horse-remarks-daily"),
    _line("0 13 * * *  ", "sporthorse-births-nightly"),
    _line("30 22 * * * ", "raceday-runner"),
    "# 0 1 * * * old commented line",
]) + "\n"


def _run(tmp_path, text, *extra):
    src, out = tmp_path / "in.txt", tmp_path / "out.txt"
    src.write_text(text, encoding="utf-8")
    r = subprocess.run(["bash", str(SCRIPT), "--crontab-file", str(src), "--out", str(out), *extra], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return out.read_text(encoding="utf-8"), r.stdout


def test_prunes_completed_and_reschedules(tmp_path):
    out, log = _run(tmp_path, SAMPLE)
    lines = out.splitlines()
    assert any(l.startswith("# PRUNED") and "backfill-2025-fast" in l for l in lines)
    assert any(l.startswith("# PRUNED") and "horse-remarks-daily" in l for l in lines)
    assert any(l.startswith("0 15 * * 0 ") and "backfill-2026-fast" in l for l in lines)
    assert any(l.startswith("50 22 * * 0 ") and "backfill-2026-full" in l for l in lines)
    assert any(l.startswith("# RESCHEDULED") for l in lines)
    # 進行中のバックフィル・開催日の取得・研究 cron（既定では）は変えない
    for keep in ("backfill-2021-fast", "backfill-all-horse", "raceday-runner", "sporthorse-births-nightly"):
        assert any(keep in l and not l.startswith("#") for l in lines), keep
    assert "変更 4 行" in log


def test_with_research_stops_sporthorse(tmp_path):
    out, _ = _run(tmp_path, SAMPLE, "--with-research")
    assert any(l.startswith("# PRUNED") and "sporthorse-births-nightly" in l for l in out.splitlines())


def test_idempotent(tmp_path):
    once, _ = _run(tmp_path, SAMPLE)
    twice, log = _run(tmp_path, once)
    assert "変更 0 行" in log
    assert [l for l in twice.splitlines() if not l.startswith("# RESCHEDULED")] == [l for l in once.splitlines() if not l.startswith("# RESCHEDULED")]
