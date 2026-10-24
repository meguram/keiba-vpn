"""メンテナンスの開始/終了を Slack に 1 通ずつ通知する（既定オフ）。

有効にする条件: config/maintenance.json の notify.slack_start_end が true、かつ KEIBA_SLACK_NOTIFY=1（かつ KEIBA_MAINTENANCE_WINDOW=1）。
送るのは各イベント 1 日 1 回（data/local/meta/slack_sent/maintenance_<日付>_<イベント>.json）。--send を付けない既定は dry-run。

  python -m src.scripts.ops.maintenance_notify start|end [--send] [--now ISO]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from src.utils import maintenance as M

REPO_ROOT = Path(__file__).resolve().parents[3]
SENT_DIR = REPO_ROOT / "data" / "local" / "meta" / "slack_sent"


def build(event: str, st: dict) -> tuple[str, list[str]]:
    if event == "start":
        return "メンテナンスを開始しました", [f"終了予定 {st.get('ends_at_label') or '-'}（日本時間）", "この間、画面は『メンテナンス中』の表示になります"]
    return "メンテナンスが終了しました", ["通常のサービスに戻りました"]


def run(event: str, *, now: datetime | None = None, send: bool = False, env: dict[str, str] | None = None, cfg: dict | None = None,
        sent_dir: Path | None = None, notify: Callable[..., Any] | None = None) -> dict:
    e = os.environ if env is None else env
    cfg = cfg if cfg is not None else M.load_config()
    if not (cfg.get("notify") or {}).get("slack_start_end"):
        return {"code": 0, "reason": "notify.slack_start_end が false"}
    if not M.is_enabled(cfg, e):
        return {"code": 0, "reason": "メンテナンス機能が無効"}
    now = now or datetime.now(M._tz(cfg))
    st = M.compute_status(cfg, now, enabled=True)
    # start は窓の中、end は窓の外のときだけ（cron がずれても誤送信しない）
    if (event == "start") != bool(st["active"]):
        return {"code": 0, "reason": f"状態が合わない（active={st['active']}）"}
    title, lines = build(event, st)
    key = (st.get("window_start") if event == "start" else (st.get("window_end") or ""))
    day = (now.strftime("%Y%m%d"))
    path = (sent_dir or SENT_DIR) / f"maintenance_{day}_{event}.json"
    if send and path.exists():
        return {"code": 0, "reason": "already_sent"}
    res: dict = {"code": 0, "reason": "ok", "title": title, "lines": lines}
    if not send:
        return res
    if str(e.get("KEIBA_SLACK_NOTIFY", "")).strip() != "1":
        return {"code": 13, "reason": "KEIBA_SLACK_NOTIFY!=1"}
    if notify is None:
        from src.utils import slack_notify as SN

        notify = SN.notify_event
    r = notify(title, lines, severity="normal", channel="report")
    res["send"] = r
    if isinstance(r, dict) and not r.get("ok", True):
        res["code"] = 14
        return res
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"sent_at": now.isoformat(timespec="seconds"), "key": key}), encoding="utf-8")
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("event", choices=["start", "end"])
    ap.add_argument("--send", action="store_true")
    ap.add_argument("--now")
    a = ap.parse_args(argv)
    now = datetime.fromisoformat(a.now) if a.now else None
    res = run(a.event, now=now, send=a.send)
    print(json.dumps({k: v for k, v in res.items() if k != "send"}, ensure_ascii=False))
    return res["code"]


if __name__ == "__main__":
    sys.exit(main())
