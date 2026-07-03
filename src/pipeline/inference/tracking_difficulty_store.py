"""
追走難度の事前計算結果を ``data/calculated_data/tracking_difficulty/`` に永続保存する。

ページ API はここを読むだけとし、オンデマンド再計算は行わない（refresh 時のみ再計算）。
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config.data_paths import TRACKING_DIFFICULTY_DIR

logger = logging.getLogger("TrackingDifficultyStore")

CACHE_VERSION = 6
MODEL_KEY = "tracking_difficulty"
_META_FILENAME = "_index_meta.json"


def store_dir() -> Path:
    d = TRACKING_DIFFICULTY_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path_for(race_id: str) -> Path:
    safe = str(race_id).strip()
    if not safe:
        raise ValueError("race_id が空です")
    return store_dir() / f"{safe}.json"


def _is_valid_meta(meta: dict | None) -> bool:
    if not meta:
        return False
    if meta.get("version") != CACHE_VERSION:
        return False
    if meta.get("model_key") != MODEL_KEY:
        return False
    return True


def load_local(race_id: str) -> dict | None:
    """ローカル事前計算 JSON を読む（TTL なし、version のみ検証）。"""
    path = _path_for(race_id)
    if not path.is_file():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("追走難度ローカル読込失敗 %s: %s", race_id, exc)
        return None
    if not isinstance(blob, dict):
        return None
    meta = blob.get("_cache_meta")
    if not _is_valid_meta(meta):
        return None
    out = {k: v for k, v in blob.items() if k != "_cache_meta"}
    out["_from_cache"] = True
    out["_cache_meta"] = meta
    out["_data_path"] = str(path)
    return out


def exists_local(race_id: str) -> bool:
    return load_local(race_id) is not None


def save_local(race_id: str, payload: dict, *, source: str = "batch") -> Path:
    """事前計算結果をローカルに保存。"""
    path = _path_for(race_id)
    wrapped = {k: v for k, v in payload.items() if k != "_cache_meta"}
    wrapped["_cache_meta"] = {
        "version": CACHE_VERSION,
        "model_key": MODEL_KEY,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "computed_at_epoch": time.time(),
        "source": source,
        "entity_id": race_id,
        "storage": "calculated_data",
    }
    path.write_text(
        json.dumps(wrapped, ensure_ascii=False, indent=None),
        encoding="utf-8",
    )
    logger.debug("追走難度ローカル保存: %s source=%s", race_id, source)
    return path


def count_local() -> int:
    d = store_dir()
    return sum(1 for p in d.glob("*.json") if p.name != _META_FILENAME)


def index_meta() -> dict[str, Any]:
    p = store_dir() / _META_FILENAME
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def update_index_meta(*, batch_source: str | None = None) -> dict[str, Any]:
    meta = {
        "version": CACHE_VERSION,
        "model_key": MODEL_KEY,
        "race_count": count_local(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "dir": str(store_dir()),
    }
    if batch_source:
        meta["last_batch_source"] = batch_source
    (store_dir() / _META_FILENAME).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return meta
