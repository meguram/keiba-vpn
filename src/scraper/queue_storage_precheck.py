"""
待機 (pending) の前段: ストレージ照会 (status=precheck)。

**統一プロトコル**（上書きなし + smart_skip）の新規行は、エンティティ（馬 / レース / 開催日）や
``tasks`` の中身に依らず **同じ** ``precheck`` 列を経由する。

``process_queue`` / 投入直後の ``run_storage_precheck`` で、ストレージ上の充足を判定し
``completed``（直完）か ``pending``（要取得）へ一括遷移する。判定ロジックは
``verify_horse_scrape_completeness`` / ``verify_scrape_completeness`` の
``is_*_task_satisfied`` 系に揃える。

GCS 利用時は、対象カテゴリのキーについて **まとめて** ``batch_check_keys``（年次
``batch_list_blobs`` の並列）を走らし、以降の満足判定の前提とする。ローカル専用
カテゴリ（例: ``race_lists``）はリスト対象外。調教（``horse_training``）は手元
JSON との二重照会を従来どおり行う。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.scraper.horse_training_local import horse_training_json_path
from src.scraper.run import _horse_training_local_dict_looks_stored, _read_horse_training_local_file

logger = logging.getLogger(__name__)


@dataclass
class PrecheckPassStats:
    from_precheck: int = 0
    to_pending: int = 0
    to_completed: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)


def _tasks_normalized_for_job(j: dict) -> list[str]:
    from src.scraper.queue_tasks import normalize_tasks

    raw = j.get("tasks") or j.get("types") or []
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        tlist = [str(x).strip() for x in raw if str(x).strip()]
    else:
        tlist = [str(raw).strip()] if str(raw).strip() else []
    if not tlist:
        return []
    return normalize_tasks(tlist)


def horse_training_storage_outcome(
    horse_id: str, storage: Any, *, batch: dict[str, float]
) -> str:
    """
    ``satisfied``: 再スクレイプ不要
    ``needs_work``: ワーカへ
    """
    hid = str(horse_id).strip()
    if not hid:
        return "needs_work"
    p = horse_training_json_path(storage._base_dir, hid)
    d = _read_horse_training_local_file(p)
    local_ok = bool(d and _horse_training_local_dict_looks_stored(d, hid))
    if not bool(getattr(storage, "gcs_enabled", None)):
        return "satisfied" if local_ok else "needs_work"
    if hid not in (batch or {}):
        return "needs_work"
    return "satisfied" if local_ok else "needs_work"


def _storage_local_only_category(storage: Any, category: str) -> bool:
    fn = getattr(storage, "_is_local_only", None)
    if callable(fn):
        try:
            return bool(fn(category))
        except Exception:
            return False
    return False


def _jra_race_ids_on_date_cached(
    storage: Any, dk: str, cache: dict[str, list[str]]
) -> list[str]:
    if dk in cache:
        return cache[dk]
    from src.scraper.verify_scrape_completeness import _jra_race_ids_on_date

    r = _jra_race_ids_on_date(storage, dk)
    cache[dk] = r
    return r


def collect_gcs_key_requirements(
    precheck_rows: list[dict], storage: Any
) -> dict[str, set[str]]:
    """
    precheck 行から、GCS 一括キー照会に乗せる (category, key) 集合を求める。

    開催日ジョブで日付配下のレースIDが要るタスクは ``race_lists`` から解決し、
    レース系カテゴリのキーに加える。``race_lists`` 自体は GCS 対象外（local_only）。
    """
    from src.scraper.verify_horse_scrape_completeness import _horse_task_storage_category
    from src.scraper.verify_scrape_completeness import (
        RACE_ALL_ATOMIC,
        _race_task_storage_category,
    )

    req: dict[str, set[str]] = {}
    rids_by_date: dict[str, list[str]] = {}

    def add(cat: str, key: str) -> None:
        k = str(key).strip()
        if not k or _storage_local_only_category(storage, cat):
            return
        req.setdefault(cat, set()).add(k)

    for j in precheck_rows:
        kind = str(j.get("job_kind") or "")
        tid = str(j.get("target_id", "")).strip()
        if not tid:
            continue
        tnorm = _tasks_normalized_for_job(j)
        if not tnorm:
            continue

        if kind == "horse":
            for t in tnorm:
                try:
                    c = _horse_task_storage_category(t)
                except ValueError:
                    continue
                add(c, tid)

        elif kind == "race":
            rid = tid
            for t in tnorm:
                if t == "race_all":
                    for a in RACE_ALL_ATOMIC:
                        try:
                            c = _race_task_storage_category(a)
                        except ValueError:
                            continue
                        add(c, rid)
                else:
                    try:
                        c = _race_task_storage_category(t)
                    except ValueError:
                        continue
                    add(c, rid)

        elif kind == "date":
            dk = tid[:8]
            for t in tnorm:
                if t == "race_list":
                    continue
                rids = _jra_race_ids_on_date_cached(storage, dk, rids_by_date)
                if t == "date_results":
                    for rid in rids:
                        add("race_result", rid)
                elif t == "date_cards":
                    for rid in rids:
                        add("race_shutuba", rid)
                elif t == "date_all":
                    for rid in rids:
                        for a in RACE_ALL_ATOMIC:
                            try:
                                c = _race_task_storage_category(a)
                            except ValueError:
                                continue
                            add(c, rid)
    return req


def run_gcs_key_batch_prefetch(
    storage: Any, requirements: dict[str, set[str]]
) -> dict[str, dict[str, float]]:
    """
    カテゴリ別に ``batch_check_keys`` を1回ずつ。GCS 未使用時は空の dict。
    """
    if not bool(getattr(storage, "gcs_enabled", None)):
        return {}
    if not requirements:
        return {}
    out: dict[str, dict[str, float]] = {}
    bck = getattr(storage, "batch_check_keys", None)
    if not callable(bck):
        return {}
    for cat, keyset in requirements.items():
        if not keyset:
            continue
        if _storage_local_only_category(storage, cat):
            continue
        klist = list(keyset)
        try:
            out[cat] = bck(cat, klist)  # type: ignore[operator]
        except Exception as e:
            logger.warning("batch_check_keys(%s) 失敗: %s", cat, e, exc_info=True)
            out[cat] = {}
    return out


def _gcs_key_present(
    storage: Any,
    gcs_blobs: dict[str, dict[str, float]] | None,
    category: str,
    key: str,
) -> bool:
    """GCS 利用時は blob 辞書にキーがあること。未使用・local_only カテは True（ゲート不要）。"""
    if not bool(getattr(storage, "gcs_enabled", None)) or not gcs_blobs:
        return True
    if _storage_local_only_category(storage, category):
        return True
    k = str(key).strip()
    return k in (gcs_blobs.get(category) or {})


def _one_job_storage_satisfied(
    j: dict, storage: Any, *, gcs_blobs: dict[str, dict[str, float]] | None
) -> bool:
    """1 行について、**すべてのタスク**が揃っていれば True（precheck 直完）。"""
    from src.scraper.verify_horse_scrape_completeness import (
        _horse_task_storage_category,
        is_horse_task_satisfied,
    )
    from src.scraper.verify_scrape_completeness import (
        RACE_ALL_ATOMIC,
        _race_task_storage_category,
        is_date_task_satisfied,
        is_race_task_satisfied,
    )

    kind = str(j.get("job_kind") or "")
    tid = str(j.get("target_id") or "").strip()
    if not tid:
        return False
    tnorm = _tasks_normalized_for_job(j)
    if not tnorm:
        return False
    bht = (gcs_blobs or {}).get("horse_training") or {}
    if kind == "horse":
        for t in tnorm:
            if t == "horse_training":
                o = horse_training_storage_outcome(tid, storage, batch=bht)
                if o != "satisfied":
                    return False
            else:
                try:
                    stc = _horse_task_storage_category(t)
                except ValueError:
                    return False
                if not _gcs_key_present(storage, gcs_blobs, stc, tid):
                    return False
                if not is_horse_task_satisfied(
                    storage, tid, t, satisfaction_mode="load_default", local_mirror_categories=None
                ):
                    return False
        return True
    if kind == "race":
        rid = tid
        for t in tnorm:
            if t == "race_all":
                for a in RACE_ALL_ATOMIC:
                    try:
                        c = _race_task_storage_category(a)
                    except ValueError:
                        return False
                    if not _gcs_key_present(storage, gcs_blobs, c, rid):
                        return False
                    if not is_race_task_satisfied(
                        storage,
                        rid,
                        a,
                        satisfaction_mode="load_default",
                        local_mirror_categories=None,
                    ):
                        return False
            else:
                try:
                    c = _race_task_storage_category(t)
                except ValueError:
                    return False
                if not _gcs_key_present(storage, gcs_blobs, c, rid):
                    return False
                if not is_race_task_satisfied(
                    storage,
                    rid,
                    t,
                    satisfaction_mode="load_default",
                    local_mirror_categories=None,
                ):
                    return False
        return True
    if kind == "date":
        dk = tid[:8]
        rids_cache: dict[str, list[str]] = {}
        for t in tnorm:
            if t == "race_list":
                try:
                    if not is_date_task_satisfied(storage, tid, t):
                        return False
                except (ValueError, Exception) as e:
                    logger.debug("is_date_task_satisfied: %s", e)
                    return False
                continue
            rids = _jra_race_ids_on_date_cached(storage, dk, rids_cache)
            if t in ("date_results", "date_cards", "date_all") and rids and gcs_blobs:
                if t == "date_results":
                    for rid in rids:
                        if not _gcs_key_present(storage, gcs_blobs, "race_result", rid):
                            return False
                elif t == "date_cards":
                    for rid in rids:
                        if not _gcs_key_present(storage, gcs_blobs, "race_shutuba", rid):
                            return False
                elif t == "date_all":
                    for rid in rids:
                        for a in RACE_ALL_ATOMIC:
                            try:
                                c = _race_task_storage_category(a)
                            except ValueError:
                                return False
                            if not _gcs_key_present(storage, gcs_blobs, c, rid):
                                return False
            try:
                if not is_date_task_satisfied(storage, tid, t):
                    return False
            except ValueError:
                return False
            except Exception as e:
                logger.debug("is_date_task_satisfied: %s", e)
                return False
        return True
    return False


def plan_unified_storage_precheck(
    jobs: list[dict], storage: Any
) -> list[tuple[str, bool]]:
    """
    precheck 行（上書きなし・スキップあり）の充足判定を一括行い、
    適用用の ``(job_id, satisfied)`` リストを返す。ジョブは改変しない。
    """
    pre: list[dict] = []
    for j in jobs:
        if str(j.get("status") or "") != "precheck":
            continue
        if j.get("overwrite") is True:
            continue
        if j.get("smart_skip") is False:
            continue
        pre.append(j)
    if not pre:
        return []
    req = collect_gcs_key_requirements(pre, storage)
    gcs_blobs = run_gcs_key_batch_prefetch(storage, req)
    out: list[tuple[str, bool]] = []
    for j in pre:
        jid = j.get("job_id")
        if not jid:
            continue
        try:
            sat = _one_job_storage_satisfied(j, storage, gcs_blobs=gcs_blobs)
        except Exception as e:
            logger.debug("precheck 判定失敗: job_id=%s: %s", jid, e)
            sat = False
        out.append((str(jid), sat))
    return out


def apply_unified_storage_precheck_patches(
    jobs: list[dict], patches: list[tuple[str, bool]]
) -> PrecheckPassStats:
    """I/O 済の patches をディスク上の行へ反映。呼び出し側は flock 内。"""
    st = PrecheckPassStats()
    st.from_precheck = len(patches)
    by_id = {str(j.get("job_id")): j for j in jobs if j.get("job_id")}

    for jid, sat in patches:
        j = by_id.get(jid)
        if not j or str(j.get("status") or "") != "precheck":
            st.skipped += 1
            continue
        if sat:
            j["status"] = "completed"
            j["completed_at"] = datetime.now().isoformat()
            j["error"] = None
            j["precheck_satisfied"] = True
            st.to_completed += 1
        else:
            j["status"] = "pending"
            j.pop("precheck_satisfied", None)
            st.to_pending += 1
    return st
