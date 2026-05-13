"""
キュー投入・実行時の「既存をスキップするか / 上書き再取得するか」。

- ジョブ JSON の ``overwrite`` … 明示指定（未指定時は環境変数を参照）
- ``SCRAPE_DEFAULT_OVERWRITE`` … 投入時に ``overwrite`` が無いときの既定（運用で一括上書きしたい場合）
- ワーカは ``effective_smart_skip_for_queue_job`` で最終的な smart_skip を解決する。
  キュー JSON に ``overwrite`` が無い旧ジョブは実行時に ``SCRAPE_DEFAULT_OVERWRITE`` を参照する。
  保存済みジョブは ``overwrite`` / ``smart_skip`` の保存値が優先される。
"""

from __future__ import annotations

import os
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


def resolve_enqueue_overwrite_smart_skip(job: dict[str, Any]) -> tuple[bool, bool]:
    """
    add_job / bulk 正規化用。戻り値: (overwrite をキューに保存する bool, smart_skip を保存する bool)。
    """
    if "overwrite" in job:
        overwrite = coerce_bool(job.get("overwrite"), default=False)
    else:
        overwrite = env_default_overwrite()
    raw_ss = job.get("smart_skip")
    base_skip = True if raw_ss is None else coerce_bool(raw_ss, default=True)
    smart_skip = base_skip and not overwrite
    return overwrite, smart_skip


def effective_smart_skip_for_queue_job(job: dict[str, Any]) -> bool:
    """ワーカが ``execute_job`` で使う最終 smart_skip。"""
    if "overwrite" not in job:
        overwrite = env_default_overwrite()
    else:
        overwrite = coerce_bool(job.get("overwrite"), default=False)
    base = True if job.get("smart_skip") is None else coerce_bool(
        job.get("smart_skip"), default=True,
    )
    return base and not overwrite
