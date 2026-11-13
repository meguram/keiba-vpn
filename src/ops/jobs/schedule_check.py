"""定期実行ジョブの毎朝の点検（Jobs ``schedule-check``、g-0730-shadow の中で 1 日 1 回）。最後の砦。

    python -m src.ops.jobs.schedule_check [--dry-run]

監視ポータル（/schedules）と同じ判定（src/ops/schedule_rules.py）で GCP・VPS の全ジョブを評価し、赤があれば Slack に 1 通送る。
VPS の見張り（15 分おき）が止まっていれば「VPS の記録の途絶」（45 分）として赤になるので、VPS とホストが同時に止まっても翌朝には気づける。
読むのは GCS 数件（ops/heartbeat/vps/status.json・ops/job_runs/<job>/latest.json）と Scheduler・Run Admin の API（読み取り無料）だけ。Cloud Run Jobs の中なので転送料なし。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any


def run(io: Any = None, *, now: float | None = None, api: Any = None, notify: Any = None, dry_run: bool = False) -> dict[str, Any]:
    from src.monitor import schedule_sources
    from src.ops import schedule_rules as R

    if io is None:
        from src.ops.gcs_io import default_io

        io = default_io()
    now = time.time() if now is None else now
    v = schedule_sources.build_view(io, now=now, api=api)
    reds = R.red_keys(v["result"])
    blind = [a["title"] for a in v["alerts"] if a["level"] == "unknown"]   # 読めない場所（権限なし・API の失敗）も黙って終えない
    out = {"overall": v["overall"], "counts": v["counts"], "red": list(reds.values()), "blind": blind, "sent": False}
    if (reds or blind) and not dry_run:
        if notify is None:
            from src.utils.slack_notify import notify_event

            notify = notify_event
        title = "定期実行（GCP・VPS）に異常があります（毎朝の点検）" if reds else "定期実行の毎朝の点検: 一部を確かめられませんでした"
        r = notify(title, (list(reds.values()) + blind)[:20], severity="critical" if reds else "warning")
        out["sent"] = not (isinstance(r, dict) and not r.get("ok", True))
        if not out["sent"]:
            out["notify_error"] = (r or {}).get("error") or (r or {}).get("reason")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    print(json.dumps(run(dry_run=a.dry_run), ensure_ascii=False))
    return 0   # 異常の有無はジョブの失敗にしない（Slack で知らせる。ジョブの失敗は別の通知になる）


if __name__ == "__main__":
    sys.exit(main())
