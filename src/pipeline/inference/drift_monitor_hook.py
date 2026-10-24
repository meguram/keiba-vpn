"""ドリフト週次監視の自動起動（KEIBA_DRIFT_MONITOR=1 のときだけ）。金曜の weekly-update の後に呼ばれ、別プロセスで監視を回す。

本体の処理(auto_scrape)は止めない・待たない: 失敗しても例外を投げず、ログを残すだけ。重い特徴量の抜き取りは含めない（--no-features）。
特徴量の PSI まで見るときは scripts/cron/drift_monitor_weekly.sh（抜き取り込み）を cron で回す。手動実行は環境変数に依らない。
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

ENV_FLAG = "KEIBA_DRIFT_MONITOR"
logger = logging.getLogger("pipeline.inference.drift_monitor_hook")
REPO_ROOT = Path(__file__).resolve().parents[3]


def enabled() -> bool:
    return (os.environ.get(ENV_FLAG) or "").strip().lower() in ("1", "true", "on", "yes")


def refresh_for_dates(dates: list[str], *, reason: str = "") -> dict:
    if not enabled():
        return {"status": "disabled"}
    try:
        log = REPO_ROOT / "logs" / "drift_monitor.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["nice", "-n", "19", sys.executable, "-m", "src.scripts.data.drift_monitor", "--root", str(REPO_ROOT), "--no-features"]
        with open(log, "ab") as fh:
            proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=fh, stderr=subprocess.STDOUT, start_new_session=True, env=dict(os.environ))
        logger.info("ドリフト監視を起動 (%s): pid=%s", reason, proc.pid)
        return {"status": "started", "pid": proc.pid, "dates": sorted(dates)[-3:]}
    except Exception as exc:  # noqa: BLE001
        logger.warning("ドリフト監視を起動できない (%s): %s", reason, exc)
        return {"status": "error", "error": str(exc)[:200]}
