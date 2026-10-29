"""過去の開催日の出馬表で、確定しないまま（馬番が全頭 0 または出馬表なし）残っているレースを見つけて再取得をキューに入れる。

entries-sync は未来のレースだけを取り直すので、確定の取得に失敗した日（登録段階の出馬表のまま）はここで拾う。
同じレースを何度も投入しないよう、試行回数を data/local/meta/shutuba_audit_state.json に残して上限で止める。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from src.scraper import entries_phase as EP
from src.scraper.missing_races import is_jra_race_id

logger = logging.getLogger("scraper.shutuba_audit")

ROOT = Path(__file__).resolve().parents[2]
MAX_ATTEMPTS = 3
AUDIT_JOB_PRIORITY = -100  # entries-sync と同じ: 開催日ランナーのジョブより後
DIRECT_MAX_PER_RUN = 12    # 直接再取得の 1 回あたりの上限（VPS の 1 起動で長く掴まない。残りは次の回）


def state_path() -> Path:
    return Path(os.environ.get("KEIBA_SHUTUBA_AUDIT_STATE") or ROOT / "data" / "local" / "meta" / "shutuba_audit_state.json")


def _load_state() -> dict[str, int]:
    try:
        return {str(k): int(v) for k, v in json.loads(state_path().read_text(encoding="utf-8")).items()}
    except (OSError, ValueError):
        return {}


def _save_state(st: dict[str, int]) -> None:
    p = state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, p)


def is_stale(card: dict | None) -> bool:
    """出馬表なし、または馬番が 1 頭にも入っていない（登録・出走馬決定の段階のまま）。"""
    if not card:
        return True
    cls = EP.classify_entries(card, card.get("date"))
    return cls["n_entries"] == 0 or cls["n_numbered"] == 0


def past_dates(race_dates: list[str], today, days: int, since: str | None = None, until: str | None = None) -> list[str]:
    """今日より前で、直近 days 日以内（since/until 指定ならその範囲）の開催日（YYYYMMDD、昇順）。"""
    lo = EP.parse_date(since) if since else today - timedelta(days=days)
    hi = EP.parse_date(until) if until else today - timedelta(days=1)
    out = [d.strftime("%Y%m%d") for d in (EP.parse_date(s) for s in race_dates) if d and lo <= d <= hi]
    return sorted(set(out))


def find_stale(storage: Any, dates: list[str]) -> list[dict]:
    rows: list[dict] = []
    for ymd in dates:
        rl = storage.load("race_lists", ymd) or {}
        for r in rl.get("races", []) or []:
            rid = str(r.get("race_id") or "").strip()
            if not rid or not is_jra_race_id(rid):
                continue
            card = storage.load("race_shutuba", rid, bypass_cache=True)
            if is_stale(card):
                rows.append({"race_id": rid, "date": ymd, "venue": str(r.get("venue") or ""), "round": int(r.get("round") or 0),
                             "race_name": str(r.get("race_name") or "")})
    return rows


def _direct_fetcher() -> Any:
    """ScraperRunner で出馬表だけをその場で取り直す関数 (race_id, date_fmt) -> エラー文字列のリスト。"""
    from src.scraper import auto_scrape_queue as _asq
    from src.scraper.run import ScraperRunner

    runner = ScraperRunner()
    return lambda race_id, date_fmt: _asq._run_race_tasks_direct(runner, race_id, date_fmt, ["race_shutuba"])


def _refetch_direct(todo: list[dict], attempts: dict[str, int], fetch: Any, cap: int) -> dict[str, Any]:
    """古い出馬表を 1 レースずつ直接取り直す。試行は取る前に数えて保存する（途中で落ちても上限に効く）。アクセス制限の疑いで止める。"""
    fetched: list[str] = []
    errors: list[str] = []
    for r in todo[:cap]:
        attempts[r["race_id"]] = attempts.get(r["race_id"], 0) + 1
        _save_state(attempts)
        errs = fetch(r["race_id"], r["date"])
        if errs:
            errors.extend(errs)
            if any("access_restriction" in e for e in errs):
                break
        else:
            fetched.append(r["race_id"])
    return {"fetched": fetched, "errors": errors, "deferred": max(0, len(todo) - cap)}


def run_audit(*, days: int = 7, since: str | None = None, until: str | None = None, enqueue: bool = False,
              storage: Any = None, queue: Any = None, now: datetime | None = None,
              direct: bool = False, fetch: Any = None, direct_cap: int = DIRECT_MAX_PER_RUN) -> dict[str, Any]:
    """direct=True は、ローカルのキューに積まず、その場で出馬表を取り直す（キューを消化する常駐が無い VPS 用）。"""
    now = now or datetime.now(EP.JST)
    if storage is None:
        from src.scraper import auto_scrape_queue as _asq

        storage = _asq._storage()
    dates = past_dates(list(storage.list_keys("race_lists")), now.date(), days, since, until)
    stale = find_stale(storage, dates)
    attempts = _load_state()
    todo = [r for r in stale if attempts.get(r["race_id"], 0) < MAX_ATTEMPTS]
    giveup = [r["race_id"] for r in stale if attempts.get(r["race_id"], 0) >= MAX_ATTEMPTS]
    result: dict[str, Any] = {"dates": dates, "stale": len(stale), "to_enqueue": len(todo), "gave_up": giveup,
                              "race_ids": [r["race_id"] for r in stale]}
    if direct and todo:
        result["direct"] = _refetch_direct(todo, attempts, fetch or _direct_fetcher(), direct_cap)
    elif enqueue and todo:
        if queue is None:
            from src.scraper import auto_scrape_queue as _asq

            queue = _asq._queue()
        specs = [{"job_kind": "race", "target_id": r["race_id"], "date": r["date"], "venue": r["venue"], "round": r["round"],
                  "race_name": r["race_name"], "tasks": ["race_shutuba"], "smart_skip": False, "priority": AUDIT_JOB_PRIORITY}
                 for r in todo]
        result["queue_bulk_add"] = queue.bulk_add_jobs(specs)
        for r in todo:
            attempts[r["race_id"]] = attempts.get(r["race_id"], 0) + 1
        _save_state(attempts)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=int, default=7, help="今日から何日前までの開催日を見るか")
    ap.add_argument("--since", default=None, help="YYYYMMDD（指定すると --days より優先）")
    ap.add_argument("--until", default=None, help="YYYYMMDD")
    ap.add_argument("--enqueue", action="store_true", help="見つかったレースの再取得をキューに入れる（既定は一覧のみ）")
    a = ap.parse_args(argv)
    res = run_audit(days=a.days, since=a.since, until=a.until, enqueue=a.enqueue)
    print(json.dumps({k: v for k, v in res.items() if k != "race_ids"}, ensure_ascii=False))
    if res["race_ids"]:
        print("対象:", ", ".join(res["race_ids"][:30]), "..." if len(res["race_ids"]) > 30 else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
