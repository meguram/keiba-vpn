"""定期実行ジョブの一覧（``config/scheduled_jobs.json``）の読み込み・検証と、予定時刻の計算。

一覧は監視ポータル（``/schedules``）・VPS の見張り（``src/ops/sched_status.py``）・毎朝の点検（``src.ops.jobs.schedule_check``）・
docs のページ（``src/scripts/docs/build_scheduled_jobs_html.py``）が共通に読む。cron は JST の 5 項目（分 時 日 月 曜日。曜日は 0=日〜6=土、7 も日）。
croniter は入れていないので、必要な分だけ自前で計算する（さかのぼるのは最大 8 日）。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

JST = timezone(timedelta(hours=9))
REPO = Path(__file__).resolve().parents[2]
PATH = REPO / "config" / "scheduled_jobs.json"

WHERE = {"gcp", "vps", "host"}
STATUS = {"active", "paused", "planned", "retired"}
WEIGHT = {"high", "med", "low"}
KINDS = {"timer", "scheduler", "run_job", "daemon", "cron"}


def load(path: Path | None = None) -> dict[str, Any]:
    d = json.loads(Path(path or PATH).read_text(encoding="utf-8"))
    problems = validate(d)
    if problems:
        raise ValueError("config/scheduled_jobs.json が不正: " + "; ".join(problems[:5]))
    return d


def validate(d: dict[str, Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for j in d.get("jobs") or []:
        jid = j.get("id") or "?"
        if jid in seen:
            out.append(f"{jid}: id が重複")
        seen.add(jid)
        if j.get("where") not in WHERE:
            out.append(f"{jid}: where")
        if j.get("status") not in STATUS:
            out.append(f"{jid}: status")
        if j.get("weight") not in WEIGHT:
            out.append(f"{jid}: weight")
        if j.get("kind") not in KINDS:
            out.append(f"{jid}: kind")
        sch = j.get("schedule")
        if sch is not None:
            if "cron" in sch:
                try:
                    parse_cron(sch["cron"])
                except ValueError as exc:
                    out.append(f"{jid}: cron {exc}")
            elif not isinstance(sch.get("every_min"), (int, float)) or sch["every_min"] <= 0:
                out.append(f"{jid}: schedule")
            if not isinstance(j.get("grace_min"), (int, float)):
                out.append(f"{jid}: grace_min")
        if not j.get("what"):
            out.append(f"{jid}: what が空")
    return out


# ───────────────────────── cron ─────────────────────────

_RANGES = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 7)]


def _field(spec: str, lo: int, hi: int) -> set[int]:
    vals: set[int] = set()
    for part in spec.split(","):
        step = 1
        stepped = "/" in part
        if stepped:
            part, s = part.split("/", 1)
            step = int(s)
            if step <= 0:
                raise ValueError("step")
        if part == "*":
            a, b = lo, hi
        elif "-" in part:
            a, b = (int(x) for x in part.split("-", 1))
        else:
            a = int(part)
            b = hi if stepped else a   # 「5/1」「5/10」は 5〜最大値（cron の慣習）
        if a < lo or b > hi or a > b:
            raise ValueError(f"範囲外 {part}")
        vals.update(range(a, b + 1, step))
    return vals


def parse_cron(expr: str) -> tuple[set[int], set[int], set[int], set[int], set[int], bool, bool]:
    f = expr.split()
    if len(f) != 5:
        raise ValueError("5 項目ではない")
    mi, h, dom, mon, dow = (_field(s, *r) for s, r in zip(f, _RANGES))
    if 7 in dow:
        dow = (dow - {7}) | {0}
    return mi, h, dom, mon, dow, f[2] == "*", f[4] == "*"


def _day_ok(d: datetime, dom: set[int], mon: set[int], dow: set[int], dom_any: bool, dow_any: bool) -> bool:
    if d.month not in mon:
        return False
    wd = (d.weekday() + 1) % 7   # cron: 0=日
    if dom_any and dow_any:
        return True
    if dom_any:
        return wd in dow
    if dow_any:
        return d.day in dom
    return d.day in dom or wd in dow   # cron の慣習: 両方指定なら OR


def slots_before(expr: str, now: float, n: int = 3, days: int = 8, day_filter: Any = None) -> list[float]:
    """now 以前の予定時刻を新しい順に n 個（epoch 秒）。day_filter(date_iso) が False の日は飛ばす。"""
    mi, h, dom, mon, dow, dom_any, dow_any = parse_cron(expr)
    cur = datetime.fromtimestamp(now, JST)
    out: list[float] = []
    for back in range(days + 1):
        d = (cur - timedelta(days=back)).replace(hour=0, minute=0, second=0, microsecond=0)
        if not _day_ok(d, dom, mon, dow, dom_any, dow_any):
            continue
        if day_filter is not None and not day_filter(d.date().isoformat()):
            continue
        for hh in sorted(h, reverse=True):
            for mm in sorted(mi, reverse=True):
                t = d.replace(hour=hh, minute=mm).timestamp()
                if t <= now:
                    out.append(t)
                    if len(out) >= n:
                        return out
    return out


def interval_min(job: dict[str, Any]) -> float | None:
    """おおよその実行間隔（分）。cron は直近 2 回の差。"""
    sch = job.get("schedule") or {}
    if "every_min" in sch:
        return float(sch["every_min"])
    if "cron" in sch:
        s = slots_before(sch["cron"], datetime.now(JST).timestamp(), n=2)
        if len(s) == 2:
            return (s[0] - s[1]) / 60.0
    return None
