"""開発者向け git pull（Next.js UI / Flask API から手動実行）。"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/cron/git_pull_hourly.sh"
LOG_FILE = ROOT / "logs" / "git_pull.log"


def _git_short_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return (out.stdout or "").strip() or "unknown"
    except Exception:
        return "unknown"


def _last_log_line() -> str:
    if not LOG_FILE.is_file():
        return ""
    try:
        lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        return lines[-1] if lines else ""
    except OSError:
        return ""


def run_git_pull() -> dict[str, Any]:
    """``git_pull_hourly.sh`` を実行し結果を返す。"""
    before = _git_short_head()
    try:
        proc = subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": "git pull がタイムアウトしました（180秒）",
            "before": before,
            "after": _git_short_head(),
            "exit_code": -1,
        }

    after = _git_short_head()
    line = _last_log_line()
    if "ERROR:" in line:
        status = "error"
    elif "skip (" in line:
        status = "skipped"
    elif "ok updated" in line:
        status = "updated"
    elif "ok (already up to date" in line:
        status = "up_to_date"
    elif proc.returncode == 0:
        status = "ok"
    else:
        status = "error"

    message = line.split("git_pull: ", 1)[-1] if "git_pull:" in line else (line or proc.stderr.strip())

    return {
        "status": status,
        "message": message or "git pull finished",
        "before": before,
        "after": after,
        "exit_code": proc.returncode,
    }
