#!/usr/bin/env python3
"""試験ログの要約。`gcloud logging read ... --format=json` の出力（またはスモーク本体の標準出力そのもの）を標準入力から読む。

Cloud Logging の形はどれでも読めるようにしてある: textPayload / jsonPayload.message / jsonPayload そのもの。
出力: 表と、合否の 1 行（`RESULT: PASS|FAIL|UNKNOWN ...`）。
"""
from __future__ import annotations

import json
import sys
from collections import Counter


def records(text: str) -> list[dict]:
    out: list[dict] = []

    def take(obj):
        if isinstance(obj, str):
            obj = obj.strip()
            if not obj.startswith("{"):
                return
            try:
                obj = json.loads(obj)
            except ValueError:
                return
        if isinstance(obj, dict):
            if "event" in obj:
                out.append(obj)
            elif "message" in obj:
                take(obj["message"])
            elif "jsonPayload" in obj:
                take(obj["jsonPayload"])
            elif "textPayload" in obj:
                take(obj["textPayload"])

    text = text.strip()
    if text.startswith("["):
        try:
            for e in json.loads(text):
                take(e)
            return out
        except ValueError:
            pass
    for line in text.splitlines():
        take(line)
    return out


def summarize(recs: list[dict]) -> tuple[str, str]:
    recs = sorted(recs, key=lambda r: (r.get("event") != "start", r.get("i", -1)))
    start = next((r for r in recs if r.get("event") == "start"), {})
    reqs = [r for r in recs if r.get("event") == "request"]
    summ = next((r for r in recs if r.get("event") == "summary"), None)
    lines = [f"送信元 IP: {start.get('egress_ip') or (summ or {}).get('egress_ip') or '取得できず'} / 枠: {start.get('slot') or '-'}"]
    lines.append(f"リクエスト数: {len(reqs)}  ステータス: {dict(Counter(r.get('code') for r in reqs))}")
    for h in ("db.netkeiba.com", "race.netkeiba.com"):
        rr = [r for r in reqs if r.get("host") == h]
        if rr:
            secs = sorted(r.get("sec", 0) for r in rr)
            lines.append(f"  {h}: n={len(rr)} 目印あり={sum(bool(r.get('marker_ok')) for r in rr)} 中央値 {secs[len(secs) // 2]}s")
    errs = dict(Counter(r["err"] for r in reqs if r.get("err")))
    if errs:
        lines.append(f"  エラー: {errs}")
    if summ is not None:
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M")
        note = "" if summ["verdict"] == "PASS" else " / ".join(summ.get("reasons") or [])
        lines.append(
            f"記録用の行（docs/operations/gcp_smoke_results.md に貼る）:\n"
            f"| {now} JST | {start.get('slot') or '-'} | {summ.get('egress_ip') or '-'} | {summ['verdict']} | {summ['ok']}/{summ['planned']} | {summ['n403_429']} | {note} |"
        )
    if summ is None:
        return "\n".join(lines), "RESULT: UNKNOWN（summary 行がありません。ログの反映待ちか、コンテナが途中で落ちた可能性）"
    why = "" if summ["verdict"] == "PASS" else " 理由: " + " / ".join(summ.get("reasons") or [])
    return "\n".join(lines), f"RESULT: {summ['verdict']} 成功 {summ['ok']}/{summ['planned']} 403/429={summ['n403_429']}{why}"


def main() -> int:
    body, result = summarize(records(sys.stdin.read()))
    print(body)
    print(result)
    return 0 if result.startswith("RESULT: PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
