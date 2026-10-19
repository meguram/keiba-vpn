#!/usr/bin/env python3
"""発走直前オッズ（T-10 / T-5 / T-2）の貯まり方を確認する。

  python -m src.scripts.data.late_odds_report --date 20261024 [--date 20261025 ...]
  python -m src.scripts.data.late_odds_report --all-local          # ローカルにある race_odds 全体（dev でも動く）

出力: 日付ごとの (1) レース数と時点別の取得数・欠損率 (2) 単勝オッズの変化（t15→t02、t05→t02）の分布 (3) 取得時刻のずれ
（snapshot_at と、発走の T-x の差。許容は 120 秒）。データが無くても動く（0 件と表示する）。読み取り専用。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

PHASES = ("t60", "t15", "t10", "t05", "t02", "final")
LEADS = {"t60": 60, "t15": 15, "t10": 10, "t05": 5, "t02": 2}


def phase_docs(doc: dict | None) -> dict[str, dict]:
    """race_odds 文書から phase → {snapshot_at, entries} を取り出す（トップが最新、snapshots が履歴）。"""
    out: dict[str, dict] = {}
    if not isinstance(doc, dict):
        return out
    for s in doc.get("snapshots") or []:
        if isinstance(s, dict) and s.get("phase"):
            out[str(s["phase"])] = s
    if doc.get("phase"):
        out[str(doc["phase"])] = doc
    return out


def win_odds(snap: dict | None) -> dict[int, float]:
    res: dict[int, float] = {}
    for e in (snap or {}).get("entries") or []:
        try:
            o = float(e.get("win_odds"))
            n = int(e.get("horse_number"))
        except (TypeError, ValueError):
            continue
        if o > 0:
            res[n] = o
    return res


def log_change(a: dict[int, float], b: dict[int, float]) -> list[float]:
    """b/a の対数変化（馬ごと）。負なら b でオッズが下がった。"""
    return [math.log(b[n] / a[n]) for n in a if n in b and a[n] > 0 and b[n] > 0]


def quantiles(xs: list[float]) -> dict[str, float | None]:
    if not xs:
        return {"n": 0, "p05": None, "p25": None, "median": None, "p75": None, "p95": None}
    s = sorted(xs)

    def q(p: float) -> float:
        return s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]

    return {"n": len(s), "p05": q(0.05), "p25": q(0.25), "median": q(0.5), "p75": q(0.75), "p95": q(0.95)}


def timing_error_sec(snap: dict, post: datetime, lead_min: int) -> float | None:
    try:
        at = datetime.fromisoformat(str(snap.get("snapshot_at")))
    except (TypeError, ValueError):
        return None
    return (at - post).total_seconds() + lead_min * 60.0


def analyze(rows: list[tuple[str, datetime | None, dict | None]]) -> dict[str, Any]:
    """rows: (race_id, 発走時刻, race_odds 文書)。"""
    n = len(rows)
    have: Counter = Counter()
    changes: dict[str, list[float]] = defaultdict(list)
    timing: dict[str, list[float]] = defaultdict(list)
    for _rid, post, doc in rows:
        ph = phase_docs(doc)
        for p in PHASES:
            if p in ph:
                have[p] += 1
        for a, b in (("t15", "t02"), ("t05", "t02"), ("t10", "t02"), ("t15", "final")):
            if a in ph and b in ph:
                changes[f"{a}→{b}"] += log_change(win_odds(ph[a]), win_odds(ph[b]))
        if post is not None:
            for p, lead in LEADS.items():
                if p in ph:
                    err = timing_error_sec(ph[p], post, lead)
                    if err is not None:
                        timing[p].append(err)
    return {
        "races": n,
        "have": {p: have.get(p, 0) for p in PHASES},
        "missing_rate": {p: (None if n == 0 else round(1 - have.get(p, 0) / n, 4)) for p in PHASES},
        "log_odds_change": {k: quantiles(v) for k, v in changes.items()},
        "timing_error_sec": {p: quantiles(v) for p, v in timing.items()},
    }


def _fmt(a: dict[str, Any]) -> str:
    lines = [f"レース数 {a['races']}"]
    for p in PHASES:
        mr = a["missing_rate"][p]
        lines.append(f"  {p:5s} 取得 {a['have'][p]:4d}  欠損率 {'-' if mr is None else f'{mr:.1%}'}")
    for k, q in a["log_odds_change"].items():
        if q["n"]:
            lines.append(f"  対数オッズ変化 {k}: n={q['n']} 中央値 {q['median']:+.4f} 5%点 {q['p05']:+.4f} 95%点 {q['p95']:+.4f}")
    for p, q in a["timing_error_sec"].items():
        if q["n"]:
            lines.append(f"  取得時刻のずれ {p}: n={q['n']} 中央値 {q['median']:+.0f}秒 95%点 {q['p95']:+.0f}秒（+ は遅れ）")
    return "\n".join(lines)


def _schedule_and_docs(storage: Any, date_fmt: str) -> list[tuple[str, datetime | None, dict | None]]:
    from src.scraper.auto_scrape_queue import _fetch_race_schedule_storage

    rows = []
    for r in _fetch_race_schedule_storage(storage, date_fmt):
        rid = r["race_id"]
        rows.append((rid, r.get("post_time"), storage.load("race_odds", rid)))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", action="append", default=[], help="YYYYMMDD（複数可）")
    ap.add_argument("--all-local", action="store_true", help="ローカルの race_odds 文書全体（発走時刻なし）")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    from src.scraper.run import REPO_ROOT
    from src.scraper.storage import HybridStorage

    storage = HybridStorage(str(REPO_ROOT))
    result: dict[str, Any] = {}
    if args.all_local:
        rows = []
        for rid in storage.list_keys("race_odds"):
            rows.append((rid, None, storage.load("race_odds", rid)))
        result["all-local"] = analyze(rows)
    for d in args.date:
        result[d] = analyze(_schedule_and_docs(storage, d))
    if not result:
        print("--date か --all-local を指定してください", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=1))
    else:
        for k, a in result.items():
            print(f"== {k}\n{_fmt(a)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
