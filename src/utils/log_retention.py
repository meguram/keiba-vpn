"""
リポジトリ ``logs/`` 直下の ``*.log`` について、最終更新時刻が保持日数を超えたものを削除する（世代管理）。

* 保持日数: 環境変数 ``LOGS_RETENTION_DAYS``（未設定時は 7）
* 対象: ``{base}/logs/*.log`` の通常ファイル（サブディレクトリは走査しない）
* 除外: 名前が ``server_latest.log`` のもの（常に直近ログへのシンボリックリンク用）
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 常に直近1本を指すシンボリックリンク（消さない）
_SKIP_NAMES = frozenset({"server_latest.log"})


def run_log_retention_once(
    base_dir: str | os.PathLike[str],
    *,
    max_age_days: float | None = None,
) -> dict[str, Any]:
    """
    ``logs/`` 直下の ``*.log`` で ``max_age_days`` 日を超えた mtime のファイルを削除。

    戻り値: removed（件数）, skipped, errors（メッセージリスト）, max_age_days, logs_dir
    """
    if max_age_days is None:
        raw = (os.environ.get("LOGS_RETENTION_DAYS") or "7").strip()
        try:
            max_age_days = float(raw)
        except ValueError:
            max_age_days = 7.0
    if max_age_days <= 0:
        return {
            "ok": True,
            "removed": 0,
            "skipped": 0,
            "bytes_freed": 0,
            "max_age_days": max_age_days,
            "note": "LOGS_RETENTION_DAYS<=0 のため削除なし",
        }

    root = Path(base_dir)
    d = (root / "logs").resolve()
    if not d.is_dir():
        return {
            "ok": True,
            "removed": 0,
            "skipped": 0,
            "bytes_freed": 0,
            "max_age_days": max_age_days,
            "note": f"no logs dir: {d}",
        }

    max_age_sec = max_age_days * 86400.0
    now = time.time()
    cutoff = now - max_age_sec
    removed = 0
    skipped = 0
    bytes_freed = 0
    errors: list[str] = []
    removed_names: list[str] = []

    for p in sorted(d.iterdir()):
        if p.name in _SKIP_NAMES:
            skipped += 1
            continue
        if p.is_dir():
            continue
        if p.suffix.lower() != ".log":
            continue
        try:
            st = p.stat()
            mtime = st.st_mtime
        except OSError as e:
            errors.append(f"{p.name}: stat {e}")
            continue
        if mtime >= cutoff:
            continue
        try:
            sz = st.st_size
            p.unlink()
            removed += 1
            bytes_freed += sz
            if len(removed_names) < 50:
                removed_names.append(p.name)
        except OSError as e:
            errors.append(f"{p.name}: unlink {e}")

    return {
        "ok": not errors,
        "removed": removed,
        "skipped": skipped,
        "bytes_freed": bytes_freed,
        "max_age_days": max_age_days,
        "cutoff_utc": datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat(),
        "sample_removed": removed_names,
        "errors": errors,
    }
