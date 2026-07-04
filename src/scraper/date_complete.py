"""
日付単位のスクレイピング完了フラグ管理。

全カテゴリのデータが揃った開催日を記録し、以降のキュー処理で即スキップ可能にする。
構造バージョンのフィンガープリントを記録し、構造変更時は自動で無効化される。
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REGISTRY_PATH = Path("data/local/meta/date_complete.json")


def _structure_fingerprint(versions: dict[str, dict]) -> str:
    """全カテゴリの構造バージョン情報からフィンガープリントを生成する。"""
    parts = sorted(
        f"{k}:{v.get('version', 0)}:{v.get('changed_at_unix', 0)}"
        for k, v in versions.items()
    )
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]


class DateCompleteRegistry:
    """
    開催日単位の完了フラグを管理する。

    フラグは data/meta/date_complete.json に永続化される。
    各エントリには構造フィンガープリントが含まれ、
    ページ構造が変わった場合は自動的に無効とみなされる。
    """

    def __init__(self, base_dir: str | Path = "."):
        self._path = Path(base_dir) / _REGISTRY_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: dict[str, dict[str, Any]] | None = None
        self._cache_mtime: float = 0.0

    # ── 読み書き ─────────────────────────────────

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            mtime = self._path.stat().st_mtime
        except OSError:
            return {}
        if self._cache is not None and mtime == self._cache_mtime:
            return self._cache
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._cache = data
            self._cache_mtime = mtime
            return data
        except Exception:
            return {}

    def _save(self, data: dict[str, dict[str, Any]]) -> None:
        tmp = self._path.with_suffix(".tmp")
        try:
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True),
                encoding="utf-8",
            )
            tmp.replace(self._path)
            self._cache = data
            try:
                self._cache_mtime = self._path.stat().st_mtime
            except OSError:
                self._cache_mtime = 0.0
        except Exception as e:
            logger.error("date_complete 保存失敗: %s", e)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    # ── 公開 API ─────────────────────────────────

    def is_complete(
        self,
        date: str,
        structure_versions: dict[str, dict] | None = None,
    ) -> bool:
        """
        指定日が完了済みか判定する。

        structure_versions を渡すと、構造フィンガープリントが一致するか検証する。
        渡さなければフィンガープリント検証をスキップ（フラグの有無のみ）。
        """
        with self._lock:
            registry = self._load()
        entry = registry.get(date)
        if not entry:
            return False
        if structure_versions is not None:
            current_fp = _structure_fingerprint(structure_versions)
            if entry.get("structure_fp") != current_fp:
                return False
        return True

    def mark_complete(
        self,
        date: str,
        *,
        structure_versions: dict[str, dict] | None = None,
        race_count: int = 0,
    ) -> None:
        """指定日を完了としてマークする。"""
        fp = _structure_fingerprint(structure_versions or {})
        entry = {
            "completed_at": time.time(),
            "structure_fp": fp,
            "race_count": race_count,
        }
        with self._lock:
            registry = self._load()
            registry[date] = entry
            self._save(registry)
        logger.info("日付完了フラグ設定: %s (races=%d, fp=%s)", date, race_count, fp)

    def invalidate(self, date: str) -> bool:
        """指定日の完了フラグを取り消す。フラグが存在した場合 True を返す。"""
        with self._lock:
            registry = self._load()
            if date not in registry:
                return False
            del registry[date]
            self._save(registry)
        logger.info("日付完了フラグ無効化: %s", date)
        return True

    def invalidate_stale(self, structure_versions: dict[str, dict]) -> int:
        """構造変更により無効になったフラグを一括削除する。削除件数を返す。"""
        current_fp = _structure_fingerprint(structure_versions)
        with self._lock:
            registry = self._load()
            stale_dates = [
                d for d, e in registry.items()
                if e.get("structure_fp") != current_fp
            ]
            if not stale_dates:
                return 0
            for d in stale_dates:
                del registry[d]
            self._save(registry)
        logger.info("構造変更により %d 件の完了フラグを無効化 (new fp=%s)", len(stale_dates), current_fp)
        return len(stale_dates)

    def summary(self) -> dict[str, Any]:
        """統計情報を返す。"""
        with self._lock:
            registry = self._load()
        return {
            "total_complete": len(registry),
            "dates": sorted(registry.keys()),
        }

    def get_all(self) -> dict[str, dict[str, Any]]:
        """全エントリを返す（API 用）。"""
        with self._lock:
            return dict(self._load())
