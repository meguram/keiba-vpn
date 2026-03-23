"""
期間内の開催日（race_lists）からレースを辿り、出走馬 ID をストレージ上のデータから収集する。

前提: 指定期間の race_lists および各レースの race_shutuba または race_result が既に存在すること。
（いずれも無いレースはスキップし、メタ情報で件数を返す）
"""

from __future__ import annotations

from typing import Any

from scraper.missing_races import is_jra_race_id


def collect_horse_ids_for_race_period(
    storage: Any,
    *,
    start_date: str | None,
    end_date: str | None,
    jra_only: bool = True,
) -> tuple[list[str], dict[str, int]]:
    """
    race_lists の開催日キー（YYYYMMDD）を期間で絞り、各 JRA レースの
    race_shutuba → 無ければ race_result の entries から horse_id をユニーク収集。
    """
    meta = {
        "dates_scanned": 0,
        "races_in_lists": 0,
        "races_with_entries": 0,
        "races_no_data": 0,
        "horses_unique": 0,
    }
    dates = sorted(storage.list_keys("race_lists"))
    if start_date:
        dates = [d for d in dates if d >= str(start_date)]
    if end_date:
        dates = [d for d in dates if d <= str(end_date)]

    seen: set[str] = set()

    for date in dates:
        meta["dates_scanned"] += 1
        rl = storage.load("race_lists", date)
        if not rl:
            continue
        for r in rl.get("races", []):
            rid = r.get("race_id")
            if not rid or not isinstance(rid, str):
                continue
            if jra_only and not is_jra_race_id(rid):
                continue
            meta["races_in_lists"] += 1

            card = storage.load("race_shutuba", rid)
            if not card:
                card = storage.load("race_result", rid)
            if not card:
                meta["races_no_data"] += 1
                continue

            entries = card.get("entries") or []
            if not entries:
                meta["races_no_data"] += 1
                continue

            meta["races_with_entries"] += 1
            for e in entries:
                if not isinstance(e, dict):
                    continue
                hid = e.get("horse_id")
                if hid and isinstance(hid, str):
                    hid = hid.strip()
                    if hid:
                        seen.add(hid)

    out = sorted(seen)
    meta["horses_unique"] = len(out)
    return out, meta


def enqueue_horse_tasks_for_race_period(
    storage: Any,
    queue: Any,
    *,
    start_date: str | None,
    end_date: str | None,
    tasks: list[str],
    limit: int,
    dry_run: bool,
    jra_only: bool = True,
) -> dict[str, Any]:
    """
    出走馬 ID を収集し、最大 limit 頭まで馬ジョブをキューに載せる（dry_run なら載せない）。
    """
    horse_ids, meta = collect_horse_ids_for_race_period(
        storage,
        start_date=start_date,
        end_date=end_date,
        jra_only=jra_only,
    )
    total_candidates = len(horse_ids)
    capped = horse_ids[: max(0, limit)]

    if dry_run:
        return {
            "dry_run": True,
            "candidate_horses": total_candidates,
            "would_enqueue": min(total_candidates, limit),
            "tasks": tasks,
            "meta": meta,
            "sample_horse_ids": capped[:25],
        }

    stats = queue.add_horse_jobs_bulk(capped, tasks)
    return {
        "dry_run": False,
        "candidate_horses": total_candidates,
        "enqueued_cap": limit,
        "tasks": tasks,
        "meta": meta,
        **stats,
    }


def collect_jra_race_job_specs_for_period(
    storage: Any,
    *,
    start_date: str | None,
    end_date: str | None,
    jra_only: bool = True,
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    race_lists の開催日を期間で絞り、JRA 各レースをユニークに最大 limit 件まで収集。
    各要素は add_job / bulk_add_jobs 用（tasks は呼び出し側で付与）。
    """
    meta: dict[str, int | bool] = {
        "dates_scanned": 0,
        "races_in_lists": 0,
        "races_unique_enqueued": 0,
        "capped": False,
    }
    dates = sorted(storage.list_keys("race_lists"))
    if start_date:
        dates = [d for d in dates if d >= str(start_date)]
    if end_date:
        dates = [d for d in dates if d <= str(end_date)]

    specs: list[dict[str, Any]] = []
    seen: set[str] = set()

    for date in dates:
        meta["dates_scanned"] += 1
        rl = storage.load("race_lists", date)
        if not rl:
            continue
        for r in rl.get("races", []):
            rid = r.get("race_id")
            if not rid or not isinstance(rid, str):
                continue
            if jra_only and not is_jra_race_id(rid):
                continue
            meta["races_in_lists"] += 1
            rid = rid.strip()
            if not rid or rid in seen:
                continue
            seen.add(rid)
            specs.append(
                {
                    "job_kind": "race",
                    "target_id": rid,
                    "date": date,
                    "venue": str(r.get("venue") or ""),
                    "round": int(r.get("round") or 0),
                    "race_name": str(r.get("race_name") or ""),
                }
            )
            if len(specs) >= limit:
                meta["capped"] = True
                meta["races_unique_enqueued"] = len(specs)
                return specs, meta

    meta["races_unique_enqueued"] = len(specs)
    return specs, meta


def enqueue_race_tasks_for_race_period(
    storage: Any,
    queue: Any,
    *,
    start_date: str | None,
    end_date: str | None,
    tasks: list[str],
    limit: int,
    dry_run: bool,
    jra_only: bool = True,
) -> dict[str, Any]:
    """
    期間内の JRA レース（race_lists）に対し、同一 tasks のレースジョブをキューへ（dry_run なら列挙のみ）。
    """
    from scraper.queue_tasks import normalize_tasks, validate_tasks_for_kind

    tasks_norm = normalize_tasks(tasks)
    if not tasks_norm:
        raise ValueError("tasks に1つ以上のタスクIDを指定してください")
    err = validate_tasks_for_kind("race", tasks_norm)
    if err:
        raise ValueError(err)

    specs, meta = collect_jra_race_job_specs_for_period(
        storage,
        start_date=start_date,
        end_date=end_date,
        jra_only=jra_only,
        limit=limit,
    )
    total = len(specs)
    if dry_run:
        sample: list[dict[str, Any]] = []
        for sp in specs[:25]:
            sample.append(
                {
                    "race_id": sp["target_id"],
                    "date": sp.get("date"),
                    "venue": sp.get("venue"),
                    "round": sp.get("round"),
                    "race_name": sp.get("race_name"),
                }
            )
        return {
            "dry_run": True,
            "candidate_races": total,
            "would_enqueue": total,
            "tasks": tasks_norm,
            "meta": meta,
            "sample_races": sample,
        }

    full = [{**sp, "tasks": tasks_norm} for sp in specs]
    stats = queue.bulk_add_jobs(full)
    return {
        "dry_run": False,
        "candidate_races": total,
        "tasks": tasks_norm,
        "meta": meta,
        **stats,
    }
