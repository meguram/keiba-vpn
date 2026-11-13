"""取得を VPS に渡したあと、ホストに残す「後処理だけ」のジョブ（docs/operations/vps_only_fetch.md の段 4）。

取得のジョブ（raceday-eve・raceday-evening・recent-retry など）は、取得のついでに後処理（前夜の計算・馬場速度・予測の成績）をしていた。
VPS の取得は後処理をしない（KEIBA_RUN_LOCATION=vps・KEIBA_EVE_FETCH_ONLY=1 で飛ばす。以前は GCP Jobs に任せる作りだったが停止中）。
そこで、対応する取得の担当が VPS に移ったジョブだけ、ホストで後処理を動かす。移っていなければ何もせずに終わる（ホストの取得ジョブが
今どおり後処理もするので二重にしない。GCS にも触れない）。

  python -m src.ops.host_postprocess eve-precompute [--force]
  python -m src.ops.host_postprocess track-speed-daily
  python -m src.ops.host_postprocess prediction-results-refresh

中身は GCP Jobs と同じ既存の入口（config/gcp_job_map.json の eve-precompute・export-tables-daily の 2 段目・prediction-results-refresh）。
"""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
JST = timezone(timedelta(hours=9))

# owners: このどれかが VPS の担当になったら動く。steps: python の引数（{today} などは JST の日付 YYYYMMDD）
JOBS: dict[str, dict[str, Any]] = {
    "eve-precompute": {
        "owners": {"raceday-eve-fetch"},
        "steps": [["-m", "src.ops.jobs.eve_precompute", "--date", "{tomorrow}"]],
        "env": {"KEIBA_TRACKING_V2": "1"},
        "timeout": 2400,
    },
    "track-speed-daily": {
        "owners": {"raceday-evening", "recent-retry"},
        "steps": [["-m", "src.scripts.data.track_speed_daily", "--from", "{days_ago_10}", "--to", "{today}"]],
        "timeout": 1800,
    },
    "prediction-results-refresh": {
        "owners": {"recent-retry"},
        "timeout": 900,
        "steps": [["-m", "src.scripts.data.prediction_results", "--from", "{days_ago_10}", "--to", "{today}", "--skip-final"],
                  ["-m", "src.scripts.data.prediction_results", "--from", "{days_ago_10}", "--to", "{today}", "--add-target-bets"]],
    },
}


def params(now: datetime | None = None) -> dict[str, str]:
    d = (now or datetime.now(JST)).astimezone(JST)
    f = lambda x: x.strftime("%Y%m%d")  # noqa: E731
    return {"today": f(d), "tomorrow": f(d + timedelta(days=1)), "days_ago_10": f(d - timedelta(days=10))}


def should_run(name: str, owned: set[str] | None = None) -> bool:
    if owned is None:
        from src.ops.fetch_owner import agent_tasks

        owned = agent_tasks()
    return bool(JOBS[name]["owners"] & owned)


def run(name: str, *, force: bool = False, owned: set[str] | None = None, now: datetime | None = None,
        runner: Callable[[list[str], dict[str, str]], int] | None = None) -> int:
    job = JOBS[name]
    if not force and not should_run(name, owned):
        print(f"{name}: 対応する取得（{', '.join(sorted(job['owners']))}）の担当は VPS に移っていない。ホストの取得ジョブが後処理もするので何もしない")
        return 0
    p = params(now)
    env = {**os.environ, **job.get("env", {})}
    limit = int(job.get("timeout", 900))           # GCP Jobs の timeout_sec と同じ。止まったまま cron が重ならないように

    def _default(argv: list[str], e: dict[str, str]) -> int:
        try:
            return subprocess.run([sys.executable, *argv], cwd=str(ROOT), env=e, timeout=limit).returncode
        except subprocess.TimeoutExpired:
            print(f"{name}: 上限時間 {limit} 秒を超えたので止めた")
            return 124

    # 同じジョブが重ならないように（前の回が長引いたとき・ほかの cron と同じ分に起動したとき）
    lock_dir = Path(os.environ.get("KEIBA_HOST_POSTPROCESS_LOCK_DIR") or "/tmp")
    with open(lock_dir / f"keiba_host_postprocess_{name}.lock", "w") as lf:
        try:
            fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"{name}: 前の回が動いているので見送る")
            return 0
        for step in job["steps"]:
            argv = [a.format(**p) for a in step]
            r = (runner or _default)(argv, env)
            print(f"{name}: {' '.join(argv)} → rc={r}")
            if r:
                return r                            # 前の段が失敗したら次の段は動かさない
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="取得を VPS に渡したあとのホストの後処理だけのジョブ")
    ap.add_argument("name", choices=sorted(JOBS))
    ap.add_argument("--force", action="store_true", help="担当に関係なく動かす（手動の確認用）")
    a = ap.parse_args(argv)
    return run(a.name, force=a.force)


if __name__ == "__main__":
    sys.exit(main())
