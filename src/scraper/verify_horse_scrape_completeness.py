"""
期間内の全出走馬について、指定タスク相当の GCS/ローカルデータ有無を検査する。

キュー完了後の抜け漏れ確認用。`verify_horses_for_race_period` で全件走査、
`enqueue_missing_for_period` で不足分のみタスク別にキュー投入。
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_VERIFY_HORSE_TASKS: list[str] = [
    "horse_profile",
    "horse_pedigree_5gen",
    "horse_training",
]


def _horse_task_storage_category(task: str) -> str:
    t = str(task).strip()
    if t in ("horse_profile", "horse_pedigree"):
        return "horse_result"
    if t == "horse_pedigree_5gen":
        return "horse_pedigree_5gen"
    if t == "horse_training":
        return "horse_training"
    raise ValueError(f"未サポートの馬タスク: {task}")


def is_horse_task_satisfied(
    storage: Any,
    horse_id: str,
    task: str,
    *,
    satisfaction_mode: str = "load_default",
    local_mirror_categories: set[str] | None = None,
) -> bool:
    """
    キュー task ID ごとに、永続層に十分なデータがあるか。
    horse_profile / horse_pedigree は horse_result を用いる。

    satisfaction_mode:
      - load_default: HybridStorage.load（メモリ→L2 → GCS）
      - gcs_strict: load(bypass_cache=True) で L1/L2 を飛ばし GCS 実体基準
      - mirror_for_selected: local_mirror_categories のストレージ名に
        当該タスクの保存カテゴリが入っていれば、data/local/mirror/ の
        常設コピー存在のみで満足（GCS 未疎通でもミラーがあれば充足とみなす）
    """
    hid = str(horse_id).strip()
    if not hid:
        return False
    t = str(task).strip()
    try:
        st_cat = _horse_task_storage_category(t)
    except ValueError:
        return False
    mir: set[str] = set(str(x) for x in (local_mirror_categories or set()) if str(x).strip())
    if satisfaction_mode == "mirror_for_selected" and mir and st_cat in mir:
        if getattr(storage, "local_mirror_exists", None) and storage.local_mirror_exists(
            st_cat, hid
        ):
            return True
        return False

    def _ld(cat: str) -> Any:
        use_bypass = satisfaction_mode == "gcs_strict"
        try:
            return storage.load(cat, hid, bypass_cache=use_bypass)  # type: ignore[call-arg]
        except TypeError:
            return storage.load(cat, hid)

    if t in ("horse_profile", "horse_pedigree"):
        try:
            d = _ld("horse_result")
        except Exception as e:
            logger.debug("horse_result load %s: %s", hid, e)
            return False
        if not d or not isinstance(d, dict):
            return False
        if (d.get("horse_name") or "").strip():
            return True
        if d.get("race_history") or d.get("entries"):
            return True
        return False
    if t == "horse_pedigree_5gen":
        try:
            d = _ld("horse_pedigree_5gen")
        except Exception as e:
            logger.debug("horse_pedigree_5gen load %s: %s", hid, e)
            return False
        if not d or not isinstance(d, dict):
            return False
        anc = (d or {}).get("ancestors") or []
        return len(anc) >= 5
    if t == "horse_training":
        try:
            d = _ld("horse_training")
            return bool(d) and isinstance(d, dict)
        except Exception as e:
            logger.debug("horse_training load %s: %s", hid, e)
            return False
    raise ValueError(f"未サポートの馬タスク: {task}")


def _normalize_verify_tasks(tasks: list[str] | None) -> list[str]:
    if not tasks:
        return list(DEFAULT_VERIFY_HORSE_TASKS)
    out: list[str] = []
    for x in tasks:
        s = str(x).strip()
        if s and s not in out:
            out.append(s)
    if "horse_profile" in out and "horse_pedigree" in out:
        out = [x for x in out if x != "horse_pedigree"]
    if not out:
        return list(DEFAULT_VERIFY_HORSE_TASKS)
    return out


def wait_until_queue_idle(
    queue: Any,
    *,
    timeout_sec: float = 3600.0,
    poll_sec: float = 3.0,
) -> tuple[bool, str, dict[str, int | str]]:
    t0 = time.time()
    last: dict[str, int | str] = {}
    while (time.time() - t0) < timeout_sec:
        try:
            st = queue.get_status()
        except Exception as e:
            return False, f"get_status: {e}", last
        qd = st.get("queue") or {}
        p = int(qd.get("pending", 0) or 0)
        r = int(qd.get("running", 0) or 0)
        last = {"pending": p, "running": r}
        if p + r == 0:
            return True, "idle", last
        time.sleep(max(0.5, poll_sec))
    return False, "timeout", last


def verify_horses_for_race_period(
    storage: Any,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    jra_only: bool = True,
    tasks: list[str] | None = None,
    max_sample: int = 20,
    satisfaction_mode: str = "load_default",
    local_mirror_categories: set[str] | None = None,
) -> dict[str, Any]:
    from src.scraper.period_runners import collect_horse_ids_for_race_period
    from src.scraper.queue_tasks import validate_tasks_for_kind

    t0 = time.time()
    tlist = _normalize_verify_tasks(
        [str(x) for x in (tasks or []) if str(x).strip()] or None
    )
    for tid in tlist:
        err = validate_tasks_for_kind("horse", [tid])
        if err:
            return {
                "ok": False,
                "error": err,
                "tasks": tlist,
            }

    horse_ids, meta = collect_horse_ids_for_race_period(
        storage,
        start_date=start_date,
        end_date=end_date,
        jra_only=jra_only,
    )
    n = len(horse_ids)
    missing: dict[str, list[str]] = {tid: [] for tid in tlist}
    for hid in horse_ids:
        for tid in tlist:
            if not is_horse_task_satisfied(
                storage,
                hid,
                tid,
                satisfaction_mode=satisfaction_mode,
                local_mirror_categories=local_mirror_categories,
            ):
                missing[tid].append(hid)

    sample = {tid: v[: max(0, max_sample)] for tid, v in missing.items()}
    counts = {tid: len(v) for tid, v in missing.items()}

    return {
        "ok": True,
        "satisfaction_mode": satisfaction_mode,
        "local_mirror_categories": sorted(
            {str(x) for x in (local_mirror_categories or set()) if str(x).strip()}
        )
        if local_mirror_categories
        else None,
        "start_date": start_date,
        "end_date": end_date,
        "jra_only": jra_only,
        "tasks": tlist,
        "total_horses": n,
        "missing_count_by_task": counts,
        "total_missing_horse_task_pairs": int(sum(len(v) for v in missing.values())),
        "sample_ids_by_task": sample,
        "missing_horse_ids_by_task": missing,
        "collect_meta": meta,
        "duration_sec": round(time.time() - t0, 3),
    }


def public_verify_payload(vr: dict[str, Any]) -> dict[str, Any]:
    """API 用: 大きな missing 全件を除いたコピー。"""
    d = {k: v for k, v in vr.items() if k != "missing_horse_ids_by_task"}
    return d


def enqueue_missing_for_period(
    queue: Any,
    storage: Any,
    verify_result: dict[str, Any],
    *,
    per_task_limit: int = 50_000,
    smart_skip: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    verify_horses_for_race_period の戻り（ok かつ missing_horse_ids_by_task あり）に対し投入。
    """
    if not verify_result.get("ok"):
        return {"ok": False, "enqueued": False, "error": "verify 未成功のため投入スキップ"}
    mbt = verify_result.get("missing_horse_ids_by_task")
    if not isinstance(mbt, dict) or not mbt:
        return {
            "ok": False,
            "enqueued": False,
            "error": "missing_horse_ids_by_task がありません。verify を先に完走させてください。",
        }
    tlist = list(verify_result.get("tasks") or list(mbt.keys()))
    out: dict[str, Any] = {"enqueued": True, "per_task": {}, "total_enqueued_ops": 0}
    for tid in tlist:
        ids = list(mbt.get(tid) or [])
        chunk = ids[: max(0, per_task_limit)]
        if not chunk:
            out["per_task"][tid] = {
                "created": 0, "requeued": 0, "duplicate": 0,
                "skipped_already_complete": 0, "horses_queued": 0,
            }
            continue
        r = queue.add_horse_jobs_bulk(
            chunk,
            [tid],
            smart_skip=smart_skip,
            overwrite=overwrite,
        )
        out["per_task"][tid] = {**r, "horses_queued": len(chunk)}
        out["total_enqueued_ops"] += int(r.get("created", 0) or 0) + int(
            r.get("requeued", 0) or 0
        )
    out["ok"] = True
    return out


if __name__ == "__main__":
    import argparse
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from dotenv import load_dotenv
    load_dotenv()

    from src.scraper.storage import HybridStorage

    p = argparse.ArgumentParser(description="出走馬の取得状況を検査（ストレージ照会のみ）")
    p.add_argument("--start-date", default=None, help="YYYYMMDD")
    p.add_argument("--end-date", default=None, help="YYYYMMDD")
    p.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="例: horse_profile horse_pedigree_5gen horse_training",
    )
    args = p.parse_args()
    st = HybridStorage()
    r = verify_horses_for_race_period(
        st,
        start_date=args.start_date,
        end_date=args.end_date,
        tasks=list(args.tasks) if args.tasks else None,
    )
    import json
    print(json.dumps(public_verify_payload(r), ensure_ascii=False, indent=2))
    sys.exit(0 if r.get("ok") else 1)
