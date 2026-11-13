#!/usr/bin/env python3
"""統括役の定期確認用: 直近 N 分に動いた各 Claude セッションの「編集したファイル・commit/install/cron などの操作・最後の発言・最後のツール」を短く出す。

使い方: python3 -I .claude/skills/claude-orchestrator/scripts/recent_activity.py [--minutes 12] [--exclude <自分の session ID の先頭 8 文字>]

- 最後のツールが ExitPlanMode・AskUserQuestion なら「ユーザー待ち」の印を付ける（計画の承認や質問で止まっている）。
- 読むのは手元のトランスクリプトだけ（GCP には触れない）。設定ディレクトリは CLAUDE_CONFIG_DIR（無ければ ~/.claude）。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from pathlib import Path

WATCH_BASH = ("sudo install", "git commit", "git push", "git merge", "cherry-pick", "crontab", "gcloud", "gsutil", "systemctl", "ssh ")


def project_dir() -> Path:
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / "CLAUDE.md").exists() and (p / ".git").exists():
            cwd = p
            break
    base = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")
    return base / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(cwd))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=12)
    ap.add_argument("--exclude", default="")
    ap.add_argument("--project-dir")
    a = ap.parse_args()
    pdir = Path(a.project_dir) if a.project_dir else project_dir()
    since = time.time() - a.minutes * 60
    ts_since = time.strftime("%Y-%m-%dT%H:%M", time.gmtime(since))
    for f in sorted(glob.glob(str(pdir / "*.jsonl")), key=os.path.getmtime):
        sid = Path(f).name[:8]
        if os.path.getmtime(f) < since or (a.exclude and sid.startswith(a.exclude)):
            continue
        edits, ops, last_text, last_tool, title = [], [], "", "", ""
        for line in open(f, encoding="utf-8", errors="replace"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            title = d.get("aiTitle") or title
            c = d.get("message", {}).get("content") if isinstance(d.get("message"), dict) else None
            if not isinstance(c, list):
                continue
            for b in c:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_use":
                    last_tool = b.get("name", "")
                if d.get("timestamp", "") < ts_since:
                    continue
                if b.get("type") == "tool_use" and b.get("name") in ("Edit", "Write"):
                    edits.append(str(b.get("input", {}).get("file_path", "")))
                elif b.get("type") == "tool_use" and b.get("name") == "Bash":
                    cmd = str(b.get("input", {}).get("command", ""))
                    if any(k in cmd for k in WATCH_BASH):
                        ops.append(cmd[:150].replace("\n", " "))
                elif b.get("type") == "text" and d.get("type") == "assistant" and b.get("text", "").strip():
                    last_text = b["text"].strip()
        wait = "  ← ユーザー待ち" if last_tool in ("ExitPlanMode", "AskUserQuestion") else ""
        print(f"== {sid} {title}  最後のツール: {last_tool}{wait}")
        for e in sorted(set(edits))[-8:]:
            print(f"   編集: {re.sub(r'.*/keiba-vpn/', '', e)}")
        for o in ops[-4:]:
            print(f"   操作: {o}")
        if last_text:
            print(f"   発言: {last_text[-220:].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
