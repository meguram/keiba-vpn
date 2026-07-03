"""
成長曲線の事前計算結果を ``data/calculated_data/growth_curve/`` に永続保存する。

- ページ表示・API はローカル JSON を優先（オンデマンド計算時も随時保存）
- 金曜 18:00 バッチで馬名インデックス対象馬を一括更新
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config.data_paths import CALC_GROWTH_CURVE_DIR

logger = logging.getLogger("GrowthCurveStore")

CACHE_VERSION = 1
ARTIFACT_KEY = "growth_curve"
_META_FILENAME = "_index_meta.json"


def store_dir() -> Path:
    d = CALC_GROWTH_CURVE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path_for(horse_id: str) -> Path:
    safe = str(horse_id).strip()
    if not safe:
        raise ValueError("horse_id が空です")
    return store_dir() / f"{safe}.json"


def _is_valid_meta(meta: dict | None) -> bool:
    if not meta:
        return False
    if meta.get("version") != CACHE_VERSION:
        return False
    if meta.get("artifact_key") != ARTIFACT_KEY:
        return False
    return True


def load_local(horse_id: str) -> dict | None:
    path = _path_for(horse_id)
    if not path.is_file():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("成長曲線ローカル読込失敗 %s: %s", horse_id, exc)
        return None
    if not isinstance(blob, dict):
        return None
    meta = blob.get("_cache_meta")
    if not _is_valid_meta(meta):
        return None
    out = {k: v for k, v in blob.items() if k != "_cache_meta"}
    if not out.get("races"):
        return None
    out["_from_cache"] = True
    out["_cache_meta"] = meta
    out["_data_path"] = str(path)
    return out


def exists_local(horse_id: str) -> bool:
    return load_local(horse_id) is not None


def is_local_fresh(horse_id: str, *, max_age_days: float = 7.0) -> bool:
    """ローカル成長曲線が有効かつ max_age_days 以内に計算済みなら True。"""
    path = _path_for(horse_id)
    if not path.is_file():
        return False
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(blob, dict) or not blob.get("races"):
        return False
    meta = blob.get("_cache_meta")
    if not _is_valid_meta(meta):
        return False
    epoch = float(meta.get("computed_at_epoch") or 0)
    if epoch <= 0:
        return False
    return (time.time() - epoch) < max_age_days * 86400


def save_local(horse_id: str, payload: dict, *, source: str = "api") -> Path:
    path = _path_for(horse_id)
    wrapped = {k: v for k, v in payload.items() if not str(k).startswith("_")}
    wrapped["_cache_meta"] = {
        "version": CACHE_VERSION,
        "artifact_key": ARTIFACT_KEY,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "computed_at_epoch": time.time(),
        "source": source,
        "entity_id": horse_id,
        "storage": "calculated_data",
    }
    path.write_text(
        json.dumps(wrapped, ensure_ascii=False, indent=None),
        encoding="utf-8",
    )
    logger.debug("成長曲線ローカル保存: %s source=%s", horse_id, source)
    return path


def apply_limit(payload: dict, limit: int | None) -> dict:
    """表示用に直近 N 走へ絞る（保存データは全出走のまま）。"""
    if not limit or limit <= 0:
        return payload
    out = dict(payload)
    races = list(out.get("races") or [])
    if len(races) <= limit:
        return out
    out["races"] = races[:limit]
    out["total_races"] = len(out["races"])
    weights = [r["weight"] for r in out["races"] if r.get("weight")]
    ranks = [r["rank"] for r in out["races"] if r.get("rank")]
    out["avg_weight"] = sum(weights) / len(weights) if weights else 0
    out["weight_range"] = [min(weights), max(weights)] if weights else [0, 0]
    out["best_rank"] = min(ranks) if ranks else None
    out["avg_rank"] = sum(ranks) / len(ranks) if ranks else None
    return out


def count_local() -> int:
    return sum(1 for p in store_dir().glob("*.json") if p.name != _META_FILENAME)


def index_meta() -> dict[str, Any]:
    p = store_dir() / _META_FILENAME
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def update_index_meta(*, batch_source: str | None = None, **extra: Any) -> dict[str, Any]:
    meta = {
        "version": CACHE_VERSION,
        "artifact_key": ARTIFACT_KEY,
        "horse_count": count_local(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "dir": str(store_dir()),
        **extra,
    }
    if batch_source:
        meta["last_batch_source"] = batch_source
    (store_dir() / _META_FILENAME).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return meta
