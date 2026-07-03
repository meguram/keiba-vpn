"""
キュー投入・実行時の「既存をスキップするか / 上書き再取得するか」。

- ジョブ JSON の ``overwrite`` … 明示指定（未指定時は環境変数を参照）
- ``SCRAPE_DEFAULT_OVERWRITE`` … 投入時に ``overwrite`` が無いときの既定（運用で一括上書きしたい場合）
- ワーカは ``effective_smart_skip_for_queue_job(job, task=...)`` で **タスク単位**に解決する。
  **可変ページ**（結果・出馬表・オッズ・馬プロフィール等）… 既定は上書き再取得（smart_skip=False）。
  **不変ページ**（現状 ``horse_pedigree_5gen`` のみ）… 既定は既存スキップ（smart_skip=True）。
  ジョブに ``smart_skip`` が明示されていれば、そのタスクについてはそれを優先する。
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any


def coerce_bool(v: Any, *, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(int(v))
    s = str(v).strip().lower()
    if s in ("", "null", "none"):
        return default
    return s not in ("0", "false", "no", "off")


def env_default_overwrite() -> bool:
    return coerce_bool(os.environ.get("SCRAPE_DEFAULT_OVERWRITE"), default=False)


@lru_cache(maxsize=1)
def _catalog_task_ids() -> frozenset[str]:
    """queue_tasks.TASK_CATALOG の id 集合（遅延 import で循環回避）。"""
    from src.scraper.queue_tasks import TASK_CATALOG

    return frozenset(str(t["id"]) for t in TASK_CATALOG)


# 一度の取得で不変とみなし、既定は「既存があればスキップ」に寄せるタスク。
QUEUE_TASK_IMMUTABLE_DEFAULT_SKIP: frozenset[str] = frozenset({"horse_pedigree_5gen"})


def resolve_enqueue_overwrite_smart_skip(job: dict[str, Any]) -> tuple[bool, bool]:
    """
    add_job / bulk 正規化用。戻り値: (overwrite をキューに保存する bool, smart_skip を保存する bool)。
    ``smart_skip`` 未指定時は tasks から既定を推定（単一不変タスクのみ True、それ以外の原子タスクは False）。
    """
    if "overwrite" in job:
        overwrite = coerce_bool(job.get("overwrite"), default=False)
    else:
        overwrite = env_default_overwrite()
    raw_ss = job.get("smart_skip")

    try:
        from src.scraper.queue_tasks import normalize_tasks

        tasks = normalize_tasks(job.get("tasks"))
    except Exception:
        tasks = []

    mut_atomic = _catalog_task_ids() - QUEUE_TASK_IMMUTABLE_DEFAULT_SKIP - frozenset(
        {"race_all"},
    )

    if raw_ss is None and tasks:
        if all(t in QUEUE_TASK_IMMUTABLE_DEFAULT_SKIP for t in tasks):
            return overwrite, not overwrite
        if "race_all" not in tasks and all(t in mut_atomic for t in tasks):
            return overwrite, False

    base_skip = True if raw_ss is None else coerce_bool(raw_ss, default=True)
    smart_skip = base_skip and not overwrite
    return overwrite, smart_skip


def _job_level_smart_skip(job: dict[str, Any], *, overwrite: bool) -> bool:
    """ジョブ単位（race_all / タスク不明時）の従来解決。"""
    base = True if job.get("smart_skip") is None else coerce_bool(
        job.get("smart_skip"), default=True,
    )
    return base and not overwrite


def effective_smart_skip_for_queue_job(
    job: dict[str, Any],
    *,
    task: str | None = None,
) -> bool:
    """
    ワーカが各タスク実行時に使う最終 smart_skip（skip_existing に相当）。

    * ``task`` が None … ジョブ全体の既定（race_all や後方互換用）
    * ``task == "race_all"`` … 複合取得のためジョブ単位の smart_skip をそのまま使う
    * 上記以外のカタログタスク … 可変 / 不変の既定を適用し、ジョブに ``smart_skip`` が
      明示されていればそれを優先（上書き運用と明示スキップの両立）
    """
    if "overwrite" not in job:
        overwrite = env_default_overwrite()
    else:
        overwrite = coerce_bool(job.get("overwrite"), default=False)
    if overwrite:
        return False

    if task is None or task == "race_all":
        return _job_level_smart_skip(job, overwrite=overwrite)

    explicit = job.get("smart_skip")
    explicit_val = None if explicit is None else coerce_bool(explicit, default=True)

    if task in QUEUE_TASK_IMMUTABLE_DEFAULT_SKIP:
        if explicit_val is not None:
            return explicit_val
        return True

    if task in _catalog_task_ids():
        if explicit_val is not None:
            return explicit_val
        # 可変ページ: 既定は上書き再取得（smart_skip=False）
        return False

    # 未知タスク ID（将来追加）… ジョブ単位にフォールバック
    return _job_level_smart_skip(job, overwrite=overwrite)
