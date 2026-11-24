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
    reds = dict(R.red_keys(v["result"]))
    blind = [a["title"] for a in v["alerts"] if a["level"] == "unknown"]   # 読めない場所（権限なし・API の失敗）も黙って終えない
    blind += list(v.get("notes") or [])   # VPS の見張りの Slack の宛先が未設定など、気づかないと困る設定の抜け
    lr_red, lr_blind = login_rotate_alerts(io, now)   # 共有ログインの毎月の切り替え（Job keiba-login-rotate）が通知の前に落ちたとき
    for msg in lr_red:
        reds[f"login-rotate:{msg}"] = msg
    blind += lr_blind
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


def login_rotate_alerts(io: Any, now: float) -> tuple[list[str], list[str]]:
    """ops/login_rotate/status.json（秘密の無い結果。src/scripts/ops/login_rotate.py）から (異常, 確かめられなかったこと) を返す。
    ファイルが無ければ（新方式がまだ動いていない）何も言わない。読むのは 1 件だけ（東京の中・Class B 1 回）。ここで例外は出さない（点検全体を止めない）。"""
    from datetime import datetime, timedelta, timezone

    try:
        raw = io.get_bytes("ops/login_rotate/status.json")
    except Exception:  # noqa: BLE001
        return [], ["共有ログインの切り替えの結果（ops/login_rotate/status.json）を読めませんでした"]
    if not raw:
        return [], []
    try:
        st = json.loads(raw.decode("utf-8"))
        if not isinstance(st, dict):
            raise ValueError
        ts = float(st.get("ts") or 0)
        t = datetime.fromtimestamp(now, timezone(timedelta(hours=9)))
        this_month = t.strftime("%Y-%m")
        name = {"prepare": "準備", "draft": "下書き", "switch": "切り替え"}.get(st.get("step"), st.get("step"))
        if st.get("result") == "failed" and (now - ts < 36 * 3600 or (st.get("step") == "switch" and st.get("month") == this_month)):
            return [f"共有ログインの{name}が失敗したままです（{st.get('month')}）: {st.get('error') or ''}"], []
        if t.day == 1 and t.hour >= 7 and not (st.get("step") == "switch" and st.get("month") == this_month and st.get("result") in ("switched", "ok")):
            return [f"共有ログインの {this_month} の切り替えが記録されていません（06:00 の Job が動かなかったか、通知の前に落ちた）"], []
        return [], []
    except (UnicodeDecodeError, ValueError, TypeError):
        return [], ["共有ログインの切り替えの結果のファイルが壊れています"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    print(json.dumps(run(dry_run=a.dry_run), ensure_ascii=False))
    return 0   # 異常の有無はジョブの失敗にしない（Slack で知らせる。ジョブの失敗は別の通知になる）


if __name__ == "__main__":
    sys.exit(main())
