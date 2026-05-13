"""
期間（年代）内の JRA レース・出走馬・開催日について、
キューで扱うタスク相当のストレージ有無を検査し、不足分をキュー投入する。

`queue_tasks.TASK_CATALOG` の各 entity（race / horse / date）に対応。
"""

from __future__ import annotations

import logging
import time
from typing import Any

from scraper.missing_races import is_jra_race_id
from scraper.queue_tasks import TASK_CATALOG, normalize_tasks, validate_tasks_for_kind
from scraper.verify_horse_scrape_completeness import (
    enqueue_missing_for_period as enqueue_missing_horses_for_period,
    is_horse_task_satisfied,
    public_verify_payload,
    verify_horses_for_race_period,
    wait_until_queue_idle,
)

logger = logging.getLogger(__name__)

# race_all ジョブが取得する原子的タスク（検証時に展開）
RACE_ALL_ATOMIC: tuple[str, ...] = (
    "race_result",
    "race_result_lap",
    "race_shutuba",
    "race_odds",
    "race_pair_odds",
    "race_index",
    "race_shutuba_past",
    "race_paddock",
    "race_barometer",
    "race_oikiri",
    "race_trainer_comment",
    "smartrc",
)


def year_to_date_range(year: int) -> tuple[str, str]:
    y = int(year)
    if y < 1984 or y > 2100:
        raise ValueError("year must be 1984..2100")
    return f"{y}0101", f"{y}1231"


def default_task_ids_for_entity(entity: str) -> list[str]:
    e = str(entity).strip().lower()
    return [t["id"] for t in TASK_CATALOG if t.get("entity") == e]


def _normalize_horse_task_list(tasks: list[str] | None) -> list[str]:
    if not tasks:
        return default_task_ids_for_entity("horse")
    out: list[str] = []
    for x in tasks:
        s = str(x).strip()
        if s and s not in out:
            out.append(s)
    if "horse_profile" in out and "horse_pedigree" in out:
        out = [x for x in out if x != "horse_pedigree"]
    if not out:
        return default_task_ids_for_entity("horse")
    return out


def _parse_race_tasks_for_verify(user_tasks: list[str] | None) -> tuple[list[str], list[str], bool]:
    """
    Returns: (atomic_tasks_for_inspection, original_user_race_ids, used_race_all)
    - user に race_all が含まれる場合は RACE_ALL_ATOMIC に展開して走査。投入時は race_all 1 本にまとめられる。
    """
    if not user_tasks:
        u = [t["id"] for t in TASK_CATALOG if t.get("entity") == "race"]
    else:
        u = [str(x).strip() for x in user_tasks if str(x).strip()]

    if "race_all" in u:
        return list(RACE_ALL_ATOMIC), u, True

    nt = normalize_tasks(u)
    if not nt:
        return list(RACE_ALL_ATOMIC), u, True
    return list(nt), u, False


def _date_keys_in_range(
    storage: Any,
    *,
    start_date: str | None,
    end_date: str | None,
) -> list[str]:
    keys = sorted(storage.list_keys("race_lists"))
    if start_date:
        keys = [d for d in keys if d >= str(start_date)]
    if end_date:
        keys = [d for d in keys if d <= str(end_date)]
    return keys


def _race_task_storage_category(task: str) -> str:
    t = str(task).strip()
    m: dict[str, str] = {
        "race_result": "race_result",
        "race_result_lap": "race_result_lap",
        "race_shutuba": "race_shutuba",
        "race_odds": "race_odds",
        "race_pair_odds": "race_pair_odds",
        "race_index": "race_index",
        "race_shutuba_past": "race_shutuba_past",
        "race_paddock": "race_paddock",
        "race_barometer": "race_barometer",
        "race_oikiri": "race_oikiri",
        "race_trainer_comment": "race_trainer_comment",
        "smartrc": "smartrc_race",
    }
    if t not in m:
        raise ValueError(f"未サポートのレースタスク: {task}")
    return m[t]


def is_race_task_satisfied(
    storage: Any,
    race_id: str,
    task: str,
    *,
    satisfaction_mode: str = "load_default",
    local_mirror_categories: set[str] | None = None,
) -> bool:
    """キュー上の race タスク1つ分。満足判定の意味は verify_horse と同様（satisfaction_mode）。"""
    rid = str(race_id).strip()
    if not rid:
        return False
    t = str(task).strip()
    mir: set[str] = {str(x) for x in (local_mirror_categories or set()) if str(x).strip()}
    try:
        stc = _race_task_storage_category(t)
    except ValueError:
        return False
    if (
        satisfaction_mode == "mirror_for_selected"
        and mir
        and stc in mir
        and getattr(storage, "local_mirror_exists", None)
        and storage.local_mirror_exists(stc, rid)
    ):
        return True
    if satisfaction_mode == "mirror_for_selected" and mir and stc in mir:
        return False

    def _ld(cat: str) -> Any:
        use = satisfaction_mode == "gcs_strict"
        try:
            return storage.load(cat, rid, bypass_cache=use)  # type: ignore[call-arg]
        except TypeError:
            return storage.load(cat, rid)

    try:
        if t == "race_result":
            d = _ld("race_result")
            if not d or not isinstance(d, dict):
                return False
            return bool(d.get("entries") or [])

        if t == "race_result_lap":
            d = _ld("race_result_lap")
            if not d or not isinstance(d, dict):
                return False
            if d.get("entries_lap") or d.get("entries"):
                return True
            return len(d) > 0

        if t in ("race_shutuba", "race_paddock", "race_oikiri", "race_trainer_comment", "race_index"):
            c = {
                "race_shutuba": "race_shutuba",
                "race_paddock": "race_paddock",
                "race_oikiri": "race_oikiri",
                "race_trainer_comment": "race_trainer_comment",
                "race_index": "race_index",
            }[t]
            d = _ld(c)
            return bool(d) and isinstance(d, dict) and bool(d.get("entries") is not None)

        if t == "race_odds":
            d = _ld("race_odds")
            return bool(d) and isinstance(d, dict) and bool(d.get("entries"))

        if t == "race_pair_odds":
            d = _ld("race_pair_odds")
            if not d or not isinstance(d, dict):
                return False
            for k in ("umaren", "wide", "umatan"):
                v = d.get(k) or []
                if isinstance(v, list) and len(v) > 0:
                    return True
            return False

        if t == "race_shutuba_past":
            d = _ld("race_shutuba_past")
            if not d or not isinstance(d, dict):
                return False
            if d.get("entries") or d.get("training"):
                return True
            return len(d) > 0

        if t == "race_barometer":
            d = _ld("race_barometer")
            if not d or not isinstance(d, dict):
                return False
            return bool(d.get("entries"))

        if t == "smartrc":
            d = _ld("smartrc_race")
            if not d or not isinstance(d, dict):
                return False
            if d.get("runners") and isinstance(d.get("runners"), list) and d["runners"]:
                return True
            h = d.get("horses")
            return bool(h) and isinstance(h, dict) and len(h) > 0
    except Exception as e:
        logger.debug("is_race_task_satisfied %s %s: %s", rid, t, e)
        return False
    return False


def _jra_race_ids_on_date(
    storage: Any,
    date: str,
) -> list[str]:
    rl = storage.load("race_lists", date) or {}
    out: list[str] = []
    for r in rl.get("races", []) or []:
        rid = r.get("race_id")
        if not rid or not isinstance(rid, str):
            continue
        if not is_jra_race_id(rid):
            continue
        rid = rid.strip()
        if rid:
            out.append(rid)
    return out


def is_date_task_satisfied(
    storage: Any,
    date_key: str,
    task: str,
) -> bool:
    """date タスク1つ分（target_id=YYYYMMDD）。"""
    dk = str(date_key).strip()[:8]
    if not dk or len(dk) != 8 or not dk.isdigit():
        return False
    t = str(task).strip()

    def _struct_versions() -> dict[str, Any]:
        fn = getattr(storage, "_load_structure_versions", None)
        if callable(fn):
            try:
                v = fn()
                return v if isinstance(v, dict) else {}
            except Exception:
                return {}
        return {}

    try:
        if t == "race_list":
            d = storage.load("race_lists", dk)
            if not d or not isinstance(d, dict):
                return False
            return "races" in d

        rids = _jra_race_ids_on_date(storage, dk)
        if not rids and t in ("date_results", "date_cards", "date_all"):
            if t == "date_all":
                from scraper.date_complete import DateCompleteRegistry

                reg = DateCompleteRegistry(str(getattr(storage, "_base_dir", ".") or "."))
                return reg.is_complete(dk, _struct_versions())
            return True

        if t == "date_results":
            for rid in rids:
                d = storage.load("race_result", rid)
                if not d or not isinstance(d, dict) or not (d.get("entries") or []):
                    return False
            return True

        if t == "date_cards":
            for rid in rids:
                d = storage.load("race_shutuba", rid)
                if not d or not isinstance(d, dict) or not (d.get("entries") or []):
                    return False
            return True

        if t == "date_all":
            from scraper.date_complete import DateCompleteRegistry

            reg = DateCompleteRegistry(str(getattr(storage, "_base_dir", ".") or "."))
            return reg.is_complete(dk, _struct_versions())
    except Exception as e:
        logger.debug("is_date_task_satisfied %s %s: %s", dk, t, e)
        return False

    raise ValueError(f"未サポートの開催日タスク: {task}")


def verify_races_for_race_period(
    storage: Any,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    jra_only: bool = True,
    user_race_tasks: list[str] | None = None,
    max_sample: int = 20,
    satisfaction_mode: str = "load_default",
    local_mirror_categories: set[str] | None = None,
) -> dict[str, Any]:
    from scraper.period_runners import collect_jra_race_job_specs_for_period

    t0 = time.time()
    expanded, user_raw, _used_race_all = _parse_race_tasks_for_verify(user_race_tasks)
    for tid in expanded:
        err = validate_tasks_for_kind("race", [tid])
        if err:
            return {"ok": False, "error": err, "tasks": expanded, "user_race_tasks": user_raw}
    if jra_only is False:
        return {
            "ok": False,
            "error": "JRA 以外の検証は未サポート（race_lists 周り）",
            "user_race_tasks": user_raw,
        }

    specs, meta = collect_jra_race_job_specs_for_period(
        storage,
        start_date=start_date,
        end_date=end_date,
        jra_only=True,
        limit=None,
    )
    race_ids = [str(s["target_id"]) for s in specs]
    missing: dict[str, list[str]] = {tid: [] for tid in expanded}
    for rid in race_ids:
        for tid in expanded:
            if not is_race_task_satisfied(
                storage,
                rid,
                tid,
                satisfaction_mode=satisfaction_mode,
                local_mirror_categories=local_mirror_categories,
            ):
                missing[tid].append(rid)

    sample = {k: v[: max(0, max_sample)] for k, v in missing.items()}
    counts = {tid: len(v) for tid, v in missing.items()}

    return {
        "ok": True,
        "entity": "race",
        "satisfaction_mode": satisfaction_mode,
        "local_mirror_categories": sorted(
            {str(x) for x in (local_mirror_categories or set()) if str(x).strip()}
        )
        if local_mirror_categories
        else None,
        "user_race_tasks": user_raw,
        "inspection_tasks": expanded,
        "used_race_all_in_request": "race_all" in (user_raw or []),
        "start_date": start_date,
        "end_date": end_date,
        "jra_only": jra_only,
        "total_races": len(race_ids),
        "missing_count_by_task": counts,
        "total_missing_race_task_pairs": int(sum(len(v) for v in missing.values())),
        "sample_ids_by_task": sample,
        "missing_race_ids_by_task": missing,
        "collect_meta": meta,
        "duration_sec": round(time.time() - t0, 3),
    }


def verify_dates_for_race_period(
    storage: Any,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    user_date_tasks: list[str] | None = None,
    max_sample: int = 20,
) -> dict[str, Any]:
    t0 = time.time()
    if not user_date_tasks:
        tlist = default_task_ids_for_entity("date")
    else:
        tlist = [str(x).strip() for x in user_date_tasks if str(x).strip()]
    for tid in tlist:
        err = validate_tasks_for_kind("date", [tid])
        if err:
            return {"ok": False, "error": err, "tasks": tlist}
    dkeys = _date_keys_in_range(
        storage,
        start_date=start_date,
        end_date=end_date,
    )
    n = len(dkeys)
    missing: dict[str, list[str]] = {tid: [] for tid in tlist}
    for dk in dkeys:
        for tid in tlist:
            if not is_date_task_satisfied(storage, dk, tid):
                missing[tid].append(dk)

    sample = {k: v[: max(0, max_sample)] for k, v in missing.items()}
    counts = {tid: len(v) for tid, v in missing.items()}

    return {
        "ok": True,
        "entity": "date",
        "tasks": tlist,
        "start_date": start_date,
        "end_date": end_date,
        "total_dates": n,
        "missing_count_by_task": counts,
        "total_missing_date_task_pairs": int(sum(len(v) for v in missing.values())),
        "sample_ids_by_task": sample,
        "missing_date_ids_by_task": missing,
        "duration_sec": round(time.time() - t0, 3),
    }


def public_strip_entity_payload(d: dict[str, Any]) -> dict[str, Any]:
    strip = {
        "missing_horse_ids_by_task",
        "missing_race_ids_by_task",
        "missing_date_ids_by_task",
    }
    return {a: b for a, b in d.items() if a not in strip}


def verify_combined_for_period(
    storage: Any,
    *,
    start_date: str | None,
    end_date: str | None,
    jra_only: bool = True,
    include_horse: bool = True,
    include_race: bool = True,
    include_date: bool = True,
    tasks_horse: list[str] | None = None,
    user_race_tasks: list[str] | None = None,
    user_date_tasks: list[str] | None = None,
    year: int | None = None,
    satisfaction_mode: str = "load_default",
    local_mirror_categories: list[str] | None = None,
) -> dict[str, Any]:
    t0 = time.time()
    out: dict[str, Any] = {
        "ok": True,
        "start_date": start_date,
        "end_date": end_date,
        "jra_only": jra_only,
        "satisfaction_mode": satisfaction_mode,
    }
    if year is not None:
        out["year"] = int(year)
    mir: set[str] = {
        str(x).strip() for x in (local_mirror_categories or []) if str(x).strip()
    }
    if mir:
        out["local_mirror_categories"] = sorted(mir)
    if include_horse:
        ht = _normalize_horse_task_list(tasks_horse)
        hr = verify_horses_for_race_period(
            storage,
            start_date=start_date,
            end_date=end_date,
            jra_only=jra_only,
            tasks=ht,
            satisfaction_mode=satisfaction_mode,
            local_mirror_categories=mir if mir else None,
        )
        out["verify_horse"] = hr
        if not hr.get("ok"):
            out["ok"] = False
    else:
        out["verify_horse"] = None

    if include_race:
        rr = verify_races_for_race_period(
            storage,
            start_date=start_date,
            end_date=end_date,
            jra_only=jra_only,
            user_race_tasks=user_race_tasks,
            satisfaction_mode=satisfaction_mode,
            local_mirror_categories=mir if mir else None,
        )
        out["verify_race"] = rr
        if not rr.get("ok"):
            out["ok"] = False
    else:
        out["verify_race"] = None

    if include_date:
        dr = verify_dates_for_race_period(
            storage,
            start_date=start_date,
            end_date=end_date,
            user_date_tasks=user_date_tasks,
        )
        out["verify_date"] = dr
        if not dr.get("ok"):
            out["ok"] = False
    else:
        out["verify_date"] = None

    out["duration_total_sec"] = round(time.time() - t0, 3)
    return out


def public_combined_payload(combined: dict[str, Any]) -> dict[str, Any]:
    p = {k: v for k, v in combined.items() if not k.startswith("verify_")}
    for key in ("verify_horse", "verify_race", "verify_date"):
        sub = combined.get(key)
        if not isinstance(sub, dict):
            p[key] = sub
            continue
        if key == "verify_horse":
            p[key] = public_verify_payload(dict(sub))
        else:
            p[key] = public_strip_entity_payload(dict(sub))
    return p


def _enqueue_race_from_verify(
    queue: Any,
    _storage: Any,
    vr: dict[str, Any],
    *,
    per_task_limit: int = 50_000,
    smart_skip: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    verify_races_for_race_period の結果から投入。リクエストに race_all がある場合（または
    既定の全レースタスク＝先頭に race_all を含む場合）は 1 レースあたり race_all 1 本に集約。
    それ以外は不足タスク単位に bulk 投入。
    """
    if not vr.get("ok"):
        return {"ok": False, "enqueued": False, "error": "verify_race 未成功"}
    mbt: dict[str, list[str]] = {
        k: list(v) for k, v in (vr.get("missing_race_ids_by_task") or {}).items() if v
    }
    if not mbt:
        return {
            "ok": True,
            "enqueued": False,
            "per_task": {},
            "total_enqueued_ops": 0,
        }

    plim = max(1, min(100_000, int(per_task_limit)))
    use_bundled = bool(vr.get("used_race_all_in_request"))

    if use_bundled:
        need: set[str] = set()
        for rlist in mbt.values():
            for r in (rlist or [])[:plim]:
                need.add(r)
        if not need:
            return {
                "ok": True, "enqueued": False, "mode": "race_all", "total_enqueued_ops": 0,
            }
        need_list = sorted(need)[:plim]
        full = [
            {
                "job_kind": "race",
                "target_id": rid,
                "tasks": ["race_all"],
                "smart_skip": bool(smart_skip),
                "overwrite": bool(overwrite),
            }
            for rid in need_list
        ]
        st = queue.bulk_add_jobs(full)
        total_ops = int(st.get("created", 0) or 0) + int(st.get("requeued", 0) or 0)
        return {
            "ok": True,
            "enqueued": total_ops > 0,
            "mode": "race_all",
            "races_targeted": len(need_list),
            "queue_stats": st,
            "total_enqueued_ops": total_ops,
        }

    per_out: dict[str, Any] = {}
    tot_ops = 0
    for tid, rlist in mbt.items():
        chunk = rlist[:plim]
        if not chunk:
            per_out[str(tid)] = {"error": "empty", "races_queued": 0, "queue_stats": None}
            continue
        full = [
            {
                "job_kind": "race",
                "target_id": rid,
                "tasks": [tid],
                "smart_skip": bool(smart_skip),
                "overwrite": bool(overwrite),
            }
            for rid in chunk
        ]
        st = queue.bulk_add_jobs(full)
        ops = int(st.get("created", 0) or 0) + int(st.get("requeued", 0) or 0)
        tot_ops += ops
        per_out[str(tid)] = {"races_queued": len(chunk), "queue_stats": st}
    return {
        "ok": True,
        "enqueued": tot_ops > 0,
        "mode": "per_task",
        "per_task": per_out,
        "total_enqueued_ops": tot_ops,
    }


def _enqueue_date_from_verify(
    queue: Any,
    vdate: dict[str, Any],
    *,
    per_task_limit: int = 50_000,
    smart_skip: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not vdate.get("ok"):
        return {"ok": False, "enqueued": False, "error": "verify_date 未成功"}
    mbt: dict[str, list[str]] = {
        k: list(v) for k, v in (vdate.get("missing_date_ids_by_task") or {}).items() if v
    }
    if not mbt:
        return {
            "ok": True, "enqueued": False, "per_task": {}, "total_enqueued_ops": 0,
        }
    plim = max(1, min(100_000, int(per_task_limit)))
    per_out: dict[str, Any] = {}
    tot_ops = 0
    for tid, dlist in mbt.items():
        chunk = dlist[:plim]
        if not chunk:
            continue
        full = [
            {
                "job_kind": "date",
                "target_id": dkey,
                "tasks": [tid],
                "smart_skip": bool(smart_skip),
                "overwrite": bool(overwrite),
            }
            for dkey in chunk
        ]
        st = queue.bulk_add_jobs(full)
        ops = int(st.get("created", 0) or 0) + int(st.get("requeued", 0) or 0)
        tot_ops += ops
        per_out[str(tid)] = {"dates_queued": len(chunk), "queue_stats": st}
    return {
        "ok": True,
        "enqueued": tot_ops > 0,
        "per_task": per_out,
        "total_enqueued_ops": tot_ops,
    }


def enqueue_combined_missing(
    queue: Any,
    storage: Any,
    combined: dict[str, Any],
    *,
    per_task_limit: int = 50_000,
    smart_skip: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "ok": True, "horse": None, "race": None, "date": None,
        "total_enqueued_ops": 0,
    }
    vh = combined.get("verify_horse")
    hm = vh.get("missing_horse_ids_by_task") if isinstance(vh, dict) else None
    if (
        isinstance(vh, dict)
        and vh.get("ok")
        and isinstance(hm, dict)
        and any(hm.values())
    ):
        eh = enqueue_missing_horses_for_period(
            queue,
            storage,
            vh,
            per_task_limit=per_task_limit,
            smart_skip=smart_skip,
            overwrite=overwrite,
        )
        out["horse"] = eh
        if not eh.get("ok", True):
            out["ok"] = False
        out["total_enqueued_ops"] += int(eh.get("total_enqueued_ops", 0) or 0)

    vr = combined.get("verify_race")
    if isinstance(vr, dict) and vr.get("ok"):
        er = _enqueue_race_from_verify(
            queue, storage, vr,
            per_task_limit=per_task_limit,
            smart_skip=smart_skip, overwrite=overwrite,
        )
        out["race"] = er
        if not er.get("ok", True):
            out["ok"] = False
        out["total_enqueued_ops"] += int(er.get("total_enqueued_ops", 0) or 0)

    vd = combined.get("verify_date")
    if isinstance(vd, dict) and vd.get("ok"):
        ed = _enqueue_date_from_verify(
            queue,
            vd,
            per_task_limit=per_task_limit,
            smart_skip=smart_skip,
            overwrite=overwrite,
        )
        out["date"] = ed
        if not ed.get("ok", True):
            out["ok"] = False
        out["total_enqueued_ops"] += int(ed.get("total_enqueued_ops", 0) or 0)
    return out