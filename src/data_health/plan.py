"""不足（gaps）から、スクレイピングすべき対象の計画（キュー投入用 spec）を作る。

spec は ``ScrapeJobQueue.bulk_add_jobs`` / ``POST /api/scrape/enqueue`` と同じ形。
既存データを上書きしないよう ``smart_skip=True`` / ``overwrite=False`` を明示する。
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

DATE_ALL_THRESHOLD = 6          # 1 開催日でこの数以上のレースに不足があれば date_all 1 本にまとめる
RECENT_DAYS = 14                # これより新しい日付は VPS の当日取得 cron、古いものは学習PCで補完


def runner_for(date: str | None, now: datetime) -> str:
    if not date:
        return "学習PC"
    d = datetime.strptime(date, "%Y%m%d").date()
    return "VPS cron（当日取得）" if d >= now.date() - timedelta(days=RECENT_DAYS) else "学習PC（過去分の補完）"


def _base(kind: str, target: str, tasks: list[str], reason: str, runner: str, priority: int, *, status: str = "missing",
          categories: list[str] | None = None, confidence: str = "confirmed") -> dict[str, Any]:
    """キュー投入用 spec（bulk_add_jobs と同形式）＋ 絞り込み用の付加情報。

    付加情報（キューは無視する）: status = missing（不足）/ invalid（スキーマ不適合）/ calendar（race_lists 不完全）、
    categories = 不足しているカテゴリ名、confidence = confirmed（race_lists で確認）/ present / inferred（連番から推定）。
    """
    return {"job_kind": kind, "target_id": target, "tasks": tasks, "smart_skip": True, "overwrite": False,
            "priority": priority, "reason": reason, "runner": runner, "status": status,
            "categories": categories or [], "confidence": confidence}


def build_plan(env: str, race_cov: dict, horse_cov: dict, calendar_anomalies: list[dict], now: datetime, *,
               include_optional: bool = False) -> dict[str, Any]:
    want = {"required", "recommended"} | ({"optional"} if include_optional else set())
    specs: list[dict] = []
    notes: list[str] = []
    if env == "dev":
        return {"env": env, "generated_at": now.isoformat(timespec="seconds"),
                "counts": {"jobs": 0, "by_runner": {}}, "specs": [],
                "notes": ["dev はスクレイピング不要（データ不足は make dev-mock で解消）"]}

    for a in calendar_anomalies:
        if "不完全" in a["problem"] or "読めない" in a["problem"]:
            specs.append(_base("date", a["date"], ["race_list"], a["problem"], runner_for(a["date"], now), 5,
                               status="calendar", categories=["race_lists"]))

    by_race: dict[str, dict] = {}
    invalid_by_race: dict[str, dict] = {}
    for g in race_cov.get("gaps", []):
        if g["level"] not in want:
            continue
        if g.get("status") == "invalid":
            if g["task"]:
                r = invalid_by_race.setdefault(g["race_id"], {"date": g["date"], "tasks": set(), "cats": set(), "issues": set(),
                                                              "confidence": g["confidence"]})
                r["tasks"].add(g["task"])
                r["cats"].add(g["category"])
                r["issues"].update(g.get("issues", []))
            else:
                notes.append(f"{g['category']} は生成物でスキーマ不適合（再生成が必要）: {g['race_id']}")
            continue
        if not g["task"]:
            notes.append(f"{g['category']} は生成物（スクレイプ対象外）: {g['race_id']}")
            continue
        r = by_race.setdefault(g["race_id"], {"date": g["date"], "tasks": set(), "cats": set(), "confidence": g["confidence"]})
        r["tasks"].add(g["task"])
        r["cats"].add(g["category"])

    by_date: dict[str | None, list[str]] = defaultdict(list)
    for rid, r in by_race.items():
        by_date[r["date"]].append(rid)
    for date, rids in by_date.items():
        soon = bool(date) and datetime.strptime(date, "%Y%m%d").date() >= now.date()
        prio = 10 if soon else 0
        if date and len(rids) >= DATE_ALL_THRESHOLD:
            cats = sorted({c for rid in rids for c in by_race[rid]["cats"]})
            specs.append(_base("date", date, ["date_all"], f"{len(rids)} レースに不足（{', '.join(cats)}）",
                               runner_for(date, now), prio, categories=cats))
        else:
            for rid in sorted(rids):
                r = by_race[rid]
                tag = "（連番から推定した欠損）" if r["confidence"] == "inferred" else ""
                specs.append({**_base("race", rid, sorted(r["tasks"]), f"不足: {', '.join(sorted(r['cats']))}{tag}",
                                      runner_for(date, now), prio, categories=sorted(r["cats"]), confidence=r["confidence"]),
                              "date": date or ""})

    # スキーマ不適合は「存在する」ので smart_skip では直らない → 上書き再取得（overwrite）で作り直す
    for rid, r in sorted(invalid_by_race.items()):
        issues = ", ".join(sorted(r["issues"])[:3])
        specs.append({**_base("race", rid, sorted(r["tasks"]), f"スキーマ不適合（{', '.join(sorted(r['cats']))}: {issues}）を上書き再取得",
                              runner_for(r["date"], now), 5, status="invalid", categories=sorted(r["cats"]), confidence=r["confidence"]),
                      "date": r["date"] or "", "smart_skip": False, "overwrite": True})

    horse_tasks: dict[str, set[str]] = defaultdict(set)
    horse_cats: dict[str, set[str]] = defaultdict(set)
    for g in horse_cov.get("gaps", []):
        if g["level"] in want and g["task"]:
            horse_tasks[g["horse_id"]].add(g["task"])
            horse_cats[g["horse_id"]].add(g["category"])
    for hid, tasks in sorted(horse_tasks.items()):
        specs.append(_base("horse", hid, sorted(tasks), "出走予定/直近出走馬のデータ不足", "VPS cron（当日取得）", 8,
                           categories=sorted(horse_cats[hid])))

    by_runner: dict[str, int] = defaultdict(int)
    for s in specs:
        by_runner[s["runner"]] += 1
    return {"env": env, "generated_at": now.isoformat(timespec="seconds"), "counts": {"jobs": len(specs), "by_runner": dict(by_runner)},
            "specs": specs, "notes": sorted(set(notes))[:50]}
