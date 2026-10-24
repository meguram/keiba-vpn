"""
Slack Incoming Webhook への軽量通知ヘルパー。

``SLACK_WEBHOOK_URL`` が未設定の場合は何もしない（既存の動作を壊さない）。
cronジョブ失敗通知等、"送れなくても本体の処理は止めない" 用途を想定しているため、
送信失敗（ネットワークエラー・Slack側障害等）は握りつぶしてログにのみ残す。
"""

from __future__ import annotations

import logging
import os

import requests

_log = logging.getLogger("utils.notify")


def notify_slack(message: str, *, timeout: float = 5.0) -> bool:
    """Slack Incoming Webhook にテキストを1件送信する。

    Returns:
        送信を試みて成功したら True。``SLACK_WEBHOOK_URL`` 未設定や送信失敗時は False
        （呼び出し元はこの戻り値で処理を分岐する必要はない想定）。
    """
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
    if not webhook_url:
        return False
    try:
        resp = requests.post(webhook_url, json={"text": message}, timeout=timeout)
        resp.raise_for_status()
        return True
    except Exception as e:
        _log.warning("Slack通知の送信に失敗: %s", e)
        return False
