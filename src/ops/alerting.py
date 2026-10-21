"""運用の異常通知。Slack へは ``src.utils.slack_notify`` の ``build_message`` / ``send_payload`` で送り（環境名・実行場所が自動で付く）、
同じ内容を ``ops/alerts/<日付>/…json`` に記録する（切替判定の『Slack 異常通知ゼロ』はこの記録を数える）。

- Slack は ``KEIBA_SLACK_NOTIFY=1`` のときだけ送る（既存の通知と同じ規則）。記録は常に書く（IO が渡されたとき）。
- 送信・記録の失敗は本処理を止めない。
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from src.ops.gcs_io import BlobIO, write_json

logger = logging.getLogger("ops.alerting")
JST = ZoneInfo("Asia/Tokyo")
ALERT_PREFIX = "ops/alerts"


def alert_key(now: datetime, job: str | None, severity: str) -> str:
    return f"{ALERT_PREFIX}/{now.strftime('%Y%m%d')}/{now.strftime('%H%M%S')}-{job or 'ops'}-{severity}.json"


def notify(title: str, lines: list[str], *, severity: str = "warning", job: str | None = None, io: BlobIO | None = None,
           channel: str = "alert", now: datetime | None = None, send: Any = None) -> dict[str, Any]:
    """異常を通知して記録する。``send`` はテスト用の差し替え（``send_payload`` と同じ引数）。"""
    from src.utils import slack_notify as SN

    now = now or datetime.now(JST)
    result: dict[str, Any] = {"slack": None, "recorded": False}
    if os.environ.get("KEIBA_SLACK_NOTIFY", "") == "1":
        blocks = [SN.section(f"*[{SN.SEVERITY_LABEL.get(severity, severity)}] {SN.esc(title)}*")]
        if lines:
            blocks.append(SN.section("\n".join("• " + SN.esc(x) for x in lines)))
        payload = SN.build_message(title, blocks, severity=severity, fallback=f"[{SN.SEVERITY_LABEL.get(severity, severity)}] {title}")
        result["slack"] = (send or SN.send_payload)(payload, channel=channel)
    if io is not None:
        try:
            write_json(io, alert_key(now, job, severity), {"title": title, "lines": lines, "severity": severity, "job": job, "at": now.isoformat(timespec="seconds"),
                                                           "slack_ok": bool((result["slack"] or {}).get("ok"))})
            result["recorded"] = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("異常の記録に失敗: %s", exc)
    return result
