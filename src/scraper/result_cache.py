"""
分析結果の永続ディスクキャッシュ。

重い計算 (追走難度予測、レース質分析、血統分析、3D適性) の結果を
ローカルに JSON として保存し、同じリクエストには GCS を経由せず即応答する。

ディレクトリ: data/cache/results/{namespace}/{key}.json

TTL は namespace ごとに設定可能。デフォルトは 24 時間。
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

_CACHE_DIR = Path("data/cache/results")
_mem_lock = threading.Lock()
_mem_cache: dict[str, tuple[float, Any]] = {}
_MEM_MAX = 500

DEFAULT_TTL = 86400  # 24h

NAMESPACE_TTLS: dict[str, float] = {
    "tracking_difficulty": 86400,
    "race_quality": 86400,
    "race_quality_day": 86400,
    "race_quality_aptitude": 86400,
    "race_note_3d": 86400,
    "race_note_3d_compare": 86400,
    "sire_factor_stats": 604800,     # 7日 — 再構築は手動
    "bloodline_analyze": 604800,
    "course_bloodline": 604800,
    "pedigree_map": 604800,
    "track_speed": 86400,
}


def _cache_dir(namespace: str, base_dir: str | Path = ".") -> Path:
    return Path(base_dir) / "data" / "cache" / "results" / namespace


def _cache_path(namespace: str, key: str, base_dir: str | Path = ".") -> Path:
    safe_key = key.replace("/", "_").replace("\\", "_")
    if len(safe_key) > 200:
        safe_key = hashlib.md5(key.encode()).hexdigest()
    return _cache_dir(namespace, base_dir) / f"{safe_key}.json"


def _ttl(namespace: str) -> float:
    return NAMESPACE_TTLS.get(namespace, DEFAULT_TTL)


def get(namespace: str, key: str, base_dir: str | Path = ".") -> Any | None:
    """キャッシュから取得。TTL 内のデータがあれば返す。"""
    mem_key = f"{namespace}/{key}"

    with _mem_lock:
        cached = _mem_cache.get(mem_key)
    if cached and (time.time() - cached[0]) < _ttl(namespace):
        return cached[1]

    path = _cache_path(namespace, key, base_dir)
    if not path.exists():
        return None

    try:
        mtime = path.stat().st_mtime
        if (time.time() - mtime) > _ttl(namespace):
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        with _mem_lock:
            if len(_mem_cache) >= _MEM_MAX:
                oldest = min(_mem_cache, key=lambda k: _mem_cache[k][0])
                del _mem_cache[oldest]
            _mem_cache[mem_key] = (time.time(), data)
        return data
    except Exception:
        return None


def put(namespace: str, key: str, data: Any, base_dir: str | Path = ".") -> None:
    """キャッシュに保存。"""
    mem_key = f"{namespace}/{key}"
    with _mem_lock:
        if len(_mem_cache) >= _MEM_MAX:
            oldest = min(_mem_cache, key=lambda k: _mem_cache[k][0])
            del _mem_cache[oldest]
        _mem_cache[mem_key] = (time.time(), data)

    path = _cache_path(namespace, key, base_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    except Exception as e:
        logger.debug("result_cache 書込失敗 %s/%s: %s", namespace, key, e)


def invalidate(namespace: str, key: str = "", base_dir: str | Path = ".") -> int:
    """キャッシュを無効化。key 省略で namespace 全体。"""
    removed = 0
    with _mem_lock:
        if key:
            if _mem_cache.pop(f"{namespace}/{key}", None):
                removed += 1
        else:
            keys_to_del = [k for k in _mem_cache if k.startswith(f"{namespace}/")]
            for k in keys_to_del:
                del _mem_cache[k]
                removed += 1

    if key:
        path = _cache_path(namespace, key, base_dir)
        if path.exists():
            path.unlink(missing_ok=True)
            removed += 1
    else:
        d = _cache_dir(namespace, base_dir)
        if d.exists():
            import shutil
            count = sum(1 for _ in d.glob("*.json"))
            shutil.rmtree(d, ignore_errors=True)
            removed += count

    return removed
