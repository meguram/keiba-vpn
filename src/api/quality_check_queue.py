"""品質チェック非同期ジョブキュー。"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any

from src.api.quality_health import CHECK_TYPES, run_check

logger = logging.getLogger(__name__)

_jobs: dict[str, dict] = {}
_lock = threading.Lock()
_queue_order: list[str] = []
_semaphore = threading.Semaphore(2)  # 同時 2 本まで（GCS 負荷抑制）


def _new_job_id(date: str, check_type: str) -> str:
    return f"qh:{date}:{check_type}:{uuid.uuid4().hex[:8]}"


def _run_job(job_id: str, date: str, check_type: str) -> None:
    with _lock:
        if job_id in _queue_order:
            _queue_order.remove(job_id)
        job = _jobs.get(job_id)
        if not job:
            return
        job["status"] = "running"
        job["started_at"] = time.time()
        job["queue_position"] = 0

    _semaphore.acquire()
    try:
        from src.scraper.storage import HybridStorage

        storage = HybridStorage()
        result = run_check(date, check_type, storage=storage)
        check_rec = (result.get("checks") or {}).get(check_type, {})
        with _lock:
            job = _jobs.get(job_id)
            if job:
                job["status"] = "done"
                job["finished_at"] = time.time()
                job["result"] = check_rec
                job["overall_status"] = result.get("overall_status")
    except Exception as e:
        logger.exception("quality job failed %s: %s", job_id, e)
        with _lock:
            job = _jobs.get(job_id)
            if job:
                job["status"] = "error"
                job["finished_at"] = time.time()
                job["error"] = str(e)
    finally:
        _semaphore.release()
        _update_positions()


def _update_positions() -> None:
    with _lock:
        for i, jid in enumerate(_queue_order):
            if jid in _jobs and _jobs[jid]["status"] == "queued":
                _jobs[jid]["queue_position"] = i + 1


def enqueue(date: str, check_type: str) -> dict[str, Any]:
    if check_type not in CHECK_TYPES:
        raise ValueError(f"check_type must be one of {CHECK_TYPES}")
    if not (date.isdigit() and len(date) == 8):
        raise ValueError("date must be YYYYMMDD")

    job_id = _new_job_id(date, check_type)
    with _lock:
        for j in _jobs.values():
            if (
                j.get("date") == date
                and j.get("check_type") == check_type
                and j.get("status") in ("queued", "running")
            ):
                return {"job_id": j["job_id"], "status": j["status"], "duplicate": True}

        _jobs[job_id] = {
            "job_id": job_id,
            "date": date,
            "check_type": check_type,
            "status": "queued",
            "created_at": time.time(),
            "queue_position": len(_queue_order) + 1,
        }
        _queue_order.append(job_id)

    threading.Thread(
        target=_run_job,
        args=(job_id, date, check_type),
        daemon=True,
        name=f"quality-{job_id}",
    ).start()
    return {"job_id": job_id, "status": "queued", "duplicate": False}


def list_jobs(*, limit: int = 30) -> list[dict]:
    with _lock:
        jobs = sorted(
            _jobs.values(),
            key=lambda j: j.get("created_at") or 0,
            reverse=True,
        )
        return [dict(j) for j in jobs[:limit]]


def get_active_jobs() -> list[dict]:
    with _lock:
        return [
            dict(j)
            for j in _jobs.values()
            if j.get("status") in ("queued", "running")
        ]
