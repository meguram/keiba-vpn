#!/usr/bin/env python3
"""指定日(既定: 今日, Asia/Tokyo)のこのプロジェクトの全 Claude セッションを、要約用に短く抜き出す。

出力は「セッションごと: タイトル・時間帯・ユーザー依頼・各依頼への最終回答・触ったファイル」。
サブエージェントのトランスクリプトは含めない(親セッションの回答に結果が出るため)。

使い方: python3 -I session_digest.py [--date YYYY-MM-DD] [--tz Asia/Tokyo] [--project-dir DIR]
"""
import argparse
import collections
import datetime as dt
import json
import re
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

PROMPT_MAX = 220
REPLY_MAX = 360
PENDING_MAX = 500
MAX_TURNS = 60
HEAD_TURNS = 10


def project_dir_default() -> Path:
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / "CLAUDE.md").exists() and (p / ".git").exists():
            cwd = p
            break
    slug = re.sub(r"[^A-Za-z0-9]", "-", str(cwd))
    return Path.home() / ".claude" / "projects" / slug


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def clean_prompt(raw: str) -> str:
    raw = re.sub(r"<system-reminder>.*?</system-reminder>", "", raw, flags=re.S)
    m = re.search(r"<command-name>(.*?)</command-name>", raw, flags=re.S)
    if m:
        args = re.search(r"<command-args>(.*?)</command-args>", raw, flags=re.S)
        return (m.group(1).strip() + " " + (args.group(1).strip() if args else "")).strip()
    raw = re.sub(r"<[^>]+>", "", raw)
    return " ".join(raw.split())


def shorten(s: str, n: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def head_tail(s: str, n: int) -> str:
    s = " ".join(s.split())
    if len(s) <= n:
        return s
    h = n // 3
    return s[:h] + " … " + s[-(n - h):]


CONT_MARK = "This session is being continued"


def pending_of(summary: str) -> str:
    m = re.search(r"Pending Tasks?:?\**\s*(.*?)(?:\s\d+\.\s+\**[A-Z][A-Za-z ]+:|\Z)", summary, flags=re.S)
    return shorten(m.group(1), PENDING_MAX) if m else ""


def parse_ts(s: str, tz: ZoneInfo):
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(tz)
    except Exception:
        return None


def digest_session(path: Path, day: dt.date, tz: ZoneInfo):
    title = ""
    turns = []  # {"t": time, "prompt": str, "reply": str}
    files = collections.Counter()
    pending = ""
    first = last = None
    for line in path.open(encoding="utf-8", errors="replace"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        typ = d.get("type")
        if typ == "ai-title":
            title = d.get("aiTitle", title)
            continue
        if typ not in ("user", "assistant") or d.get("isSidechain"):
            continue
        ts = parse_ts(d.get("timestamp", ""), tz)
        if ts is None or ts.date() != day:
            continue
        first = first or ts
        last = ts
        msg = d.get("message") or {}
        if typ == "user":
            if (d.get("origin") or {}).get("kind") != "human" and d.get("turnOrigin") != "human":
                continue
            p = clean_prompt(text_of(msg.get("content")))
            if p.startswith(CONT_MARK):
                pending = pending_of(p) or pending
                p = "(コンテキスト引き継ぎ後の作業)"
            if p:
                turns.append({"t": ts, "prompt": p, "reply": ""})
        else:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for b in content:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text" and b.get("text", "").strip() and turns:
                    turns[-1]["reply"] = b["text"]
                elif b.get("type") == "tool_use" and b.get("name") in ("Edit", "Write", "NotebookEdit"):
                    fp = (b.get("input") or {}).get("file_path")
                    if fp:
                        files[fp] += 1
    if not turns and first is None:
        return None
    return {"id": path.stem, "title": title, "first": first, "last": last, "turns": turns, "files": files, "pending": pending}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    ap.add_argument("--tz", default="Asia/Tokyo")
    ap.add_argument("--project-dir")
    a = ap.parse_args()
    tz = ZoneInfo(a.tz)
    day = dt.date.fromisoformat(a.date) if a.date else dt.datetime.now(tz).date()
    pdir = Path(a.project_dir) if a.project_dir else project_dir_default()
    if not pdir.is_dir():
        sys.exit(f"プロジェクトのトランスクリプトが見つかりません: {pdir}")

    sessions = []
    for p in pdir.glob("*.jsonl"):
        if dt.datetime.fromtimestamp(p.stat().st_mtime, tz).date() < day:
            continue
        s = digest_session(p, day, tz)
        if s:
            sessions.append(s)
    sessions.sort(key=lambda s: s["first"])

    print(f"# {day.isoformat()} ({a.tz}) セッション {len(sessions)} 件  [{pdir}]")
    for s in sessions:
        t0, t1 = s["first"].strftime("%H:%M"), s["last"].strftime("%H:%M")
        print(f"\n## session {s['id'][:8]}  {t0}-{t1}  title: {s['title'] or '(なし)'}")
        turns = s["turns"]
        if len(turns) > MAX_TURNS:
            skipped = len(turns) - MAX_TURNS
            turns = turns[:HEAD_TURNS] + [None] + turns[-(MAX_TURNS - HEAD_TURNS):]
        else:
            skipped = 0
        for t in turns:
            if t is None:
                print(f"  …(中略 {skipped} 件の依頼)…")
                continue
            print(f"- [{t['t'].strftime('%H:%M')}] 依頼: {shorten(t['prompt'], PROMPT_MAX)}")
            if t["reply"]:
                print(f"    回答: {head_tail(t['reply'], REPLY_MAX)}")
        if s["pending"]:
            print(f"  引き継ぎ要約の未了事項: {s['pending']}")
        if s["files"]:
            top = ", ".join(f"{Path(f).name}×{n}" for f, n in s["files"].most_common(8))
            print(f"  編集ファイル({len(s['files'])}): {top}")


if __name__ == "__main__":
    main()
