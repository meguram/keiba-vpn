"""
キュー実行中ジョブの細かい進捗。

- プロセス内メモリ（同一プロセスの API と共有）
- 別プロセスの queue ワーカー用に `data/meta/_queue_job_progress.json` へスロットリング書き込み（API は読み取り）
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_by_job: dict[str, dict[str, Any]] = {}
_tl = threading.local()

# ワーカーと API が別プロセスのときに共有するスナップショット
PROGRESS_FILE = Path("data/local/meta/_queue_job_progress.json")
_FILE_WRITE_INTERVAL = 1.0
_last_file_write = 0.0

# スナップショットに載せる最大件数（完了ジョブは clear で消すが、並列時の上限）
MAX_PROGRESS_SNAPSHOT_JOBS = 64


def set_current_job_id(jid: str | None) -> None:
    _tl.job_id = jid

def clear_current_job_id() -> None:
    _tl.job_id = None

def get_current_job_id() -> str | None:
    return getattr(_tl, "job_id", None)


def update_job_progress(job_id: str, **kwargs: Any) -> None:
    if not str(job_id).strip():
        return
    now = time.time()
    with _lock:
        cur = dict(_by_job.get(job_id, {}))
        cur["job_id"] = job_id
        cur["updated_at"] = now
        for k, v in kwargs.items():
            cur[k] = v
        _by_job[job_id] = cur
    _write_snapshot_file(now, force=False)


def _prune_jobs_for_snapshot(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(jobs) <= MAX_PROGRESS_SNAPSHOT_JOBS:
        return jobs
    return sorted(
        jobs,
        key=lambda j: float(j.get("updated_at") or 0),
        reverse=True,
    )[:MAX_PROGRESS_SNAPSHOT_JOBS]


def _write_snapshot_file(now: float, *, force: bool) -> None:
    global _last_file_write
    if not force and now - _last_file_write < _FILE_WRITE_INTERVAL:
        return
    _last_file_write = now
    try:
        PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _lock:
            jobs = sorted(
                _by_job.values(),
                key=lambda j: float(j.get("updated_at") or 0),
                reverse=True,
            )
            jobs = _prune_jobs_for_snapshot(list(jobs))
            payload = {"jobs": jobs, "updated_at": now}
        PROGRESS_FILE.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
    except Exception as e:
        logger.debug("queue progress ファイル書込スキップ: %s", e)


def get_progress_snapshot() -> dict[str, Any]:
    """同一プロセス内のメモリのみ。"""
    now = time.time()
    with _lock:
        jobs = sorted(
            _by_job.values(),
            key=lambda j: float(j.get("updated_at") or 0),
            reverse=True,
        )
        jobs = _prune_jobs_for_snapshot(list(jobs))
        return {"jobs": jobs, "updated_at": now}


def get_progress_snapshot_for_api() -> dict[str, Any]:
    """API 用: ファイル（別プロセスのワーカー）を優先し、無ければメモリ。"""
    now = time.time()
    if PROGRESS_FILE.exists():
        try:
            raw = PROGRESS_FILE.read_text(encoding="utf-8")
            data = json.loads(raw)
            jobs = data.get("jobs")
            if isinstance(jobs, list):
                jobs = _prune_jobs_for_snapshot(
                    [j for j in jobs if isinstance(j, dict)]
                )
                return {"jobs": jobs, "updated_at": data.get("updated_at", now)}
        except Exception:
            pass
    return get_progress_snapshot()


def clear_job_progress(job_id: str) -> None:
    jid = str(job_id or "").strip()
    if not jid:
        return
    with _lock:
        _by_job.pop(jid, None)
    _write_snapshot_file(time.time(), force=True)
