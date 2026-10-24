"""スクレイピング対策のイベントログ（data/local/meta/anti_scraping/events-YYYYMMDD.jsonl）の日次サマリーと急増の警報。

既定は標準出力に表示するだけ（dry-run）。--send で Slack へ（KEIBA_SLACK_NOTIFY=1 のときだけ。値は扱わない）。
Usage: python -m src.scripts.maintenance.anti_scraping_report [--date YYYYMMDD] [--hours 24] [--send] [--alert-only]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIR = REPO_ROOT / "data" / "local" / "meta" / "anti_scraping"
JST = 9 * 3600


def load_events(log_dir: Path, since: float, until: float) -> list[dict]:
    out: list[dict] = []
    days = {time.strftime("%Y%m%d", time.gmtime(t + JST)) for t in (since, until)}
    for day in sorted(days):
        p = log_dir / f"events-{day}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if since <= float(r.get("ts", 0)) <= until:
                out.append(r)
    return out


def summarize(events: list[dict], hours: float) -> dict:
    by_kind = Counter(f"{e.get('kind')}:{e.get('action')}" for e in events)
    blocks = [e for e in events if e.get("action") == "block"]
    ips = Counter(e.get("ip") for e in events if e.get("ip"))
    uas = Counter((e.get("ua") or "")[:60] for e in events if e.get("ua"))
    reasons = Counter(str(e.get("reason", "")).split(":")[0] for e in events)
    return {
        "hours": hours,
        "total": len(events),
        "blocks": len(blocks),
        "by_kind": dict(by_kind.most_common()),
        "reasons": dict(reasons.most_common(8)),
        "top_ips": ips.most_common(5),
        "top_ua": uas.most_common(5),
    }


def render_lines(s: dict) -> list[str]:
    lines = [f"直近 {s['hours']:g} 時間: 記録 {s['total']} 件（ブロック {s['blocks']} 件）"]
    if s["by_kind"]:
        lines.append("種類別: " + ", ".join(f"{k}={v}" for k, v in list(s["by_kind"].items())[:8]))
    if s["reasons"]:
        lines.append("理由: " + ", ".join(f"{k}={v}" for k, v in s["reasons"].items()))
    if s["top_ips"]:
        lines.append("多い接続元: " + ", ".join(f"{ip}({n})" for ip, n in s["top_ips"]))
    if s["top_ua"]:
        lines.append("多い UA: " + ", ".join(f"{ua or '(なし)'}({n})" for ua, n in s["top_ua"]))
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--send", action="store_true")
    ap.add_argument("--alert-only", action="store_true", help="1 時間あたりのブロック数が閾値を超えたときだけ通知する")
    ap.add_argument("--threshold", type=int, default=None, help="警報の閾値（既定は config/anti_scraping.json の notify.blocks_per_hour_alert）")
    a = ap.parse_args(argv)
    now = time.time()
    ev = load_events(Path(a.dir), now - a.hours * 3600, now)
    s = summarize(ev, a.hours)
    thr = a.threshold
    if thr is None:
        try:
            thr = int(json.loads((REPO_ROOT / "config" / "anti_scraping.json").read_text(encoding="utf-8"))["notify"]["blocks_per_hour_alert"])
        except (OSError, ValueError, KeyError):
            thr = 50
    recent = summarize([e for e in ev if float(e.get("ts", 0)) >= now - 3600], 1.0)
    alert = recent["blocks"] >= thr
    lines = render_lines(s)
    print("\n".join(lines))
    if alert:
        print(f"警報: 直近 1 時間のブロックが {recent['blocks']} 件（閾値 {thr}）")
    if not a.send:
        return 0
    if os.environ.get("KEIBA_SLACK_NOTIFY") != "1":
        print("KEIBA_SLACK_NOTIFY!=1 のため送信しません", file=sys.stderr)
        return 0
    if a.alert_only and not alert:
        return 0
    from src.utils import slack_notify as SN

    title = "スクレイピング対策: ブロックが急増" if alert else "スクレイピング対策: 日次サマリー"
    r = SN.notify_event(title, lines, severity="warning" if alert else "info", channel="alert" if alert else "report")
    print(json.dumps(r, ensure_ascii=False))
    return 0 if r.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
