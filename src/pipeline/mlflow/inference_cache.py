"""
推論レスポンスの storage キャッシュ共通処理（モデルカタログ連携）。
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any

from src.pipeline.mlflow.catalog import ModelSpec, get_model_spec
from src.utils.logger import get_logger

logger = get_logger("InferenceCache")

CACHE_VERSION = 1


class InferenceCacheMixin:
    """
    レース単位などの推論 JSON を HybridStorage に保存するミックスイン。

    サブクラスで ``model_key`` と ``cache_version`` を定義する。
    """

    model_key: str = ""
    cache_version: int = CACHE_VERSION

    @classmethod
    def spec(cls) -> ModelSpec:
        return get_model_spec(cls.model_key)

    @classmethod
    def cache_category(cls) -> str:
        cat = cls.spec().cache_category
        if not cat:
            raise ValueError(f"{cls.model_key}: cache_category 未設定")
        return cat

    @classmethod
    def cache_enabled(cls) -> bool:
        spec = cls.spec()
        env_name = spec.cache_enabled_env or f"KEIBA_{spec.key.upper()}_CACHE"
        v = os.environ.get(env_name, "1").strip().lower()
        return v not in ("0", "false", "no", "off")

    @classmethod
    def _cache_ttl_sec(cls) -> float:
        spec = cls.spec()
        env_name = spec.cache_ttl_env or f"KEIBA_{spec.key.upper()}_CACHE_TTL_SEC"
        try:
            return float(os.environ.get(env_name, "86400"))
        except ValueError:
            return 86400.0

    @classmethod
    def _is_fresh(cls, meta: dict | None) -> bool:
        if not meta:
            return False
        if meta.get("version") != cls.cache_version:
            return False
        if meta.get("model_key") != cls.model_key:
            return False
        ts = meta.get("computed_at_epoch")
        if ts is None:
            return False
        return (time.time() - float(ts)) < cls._cache_ttl_sec()

    @classmethod
    def load_cached(cls, storage, entity_id: str) -> dict | None:
        if not cls.cache_enabled():
            return None
        blob = storage.load(cls.cache_category(), entity_id)
        if not blob or not isinstance(blob, dict):
            return None
        meta = blob.get("_cache_meta") or {}
        if not cls._is_fresh(meta):
            return None
        out = {k: v for k, v in blob.items() if k != "_cache_meta"}
        out["_from_cache"] = True
        out["_cache_meta"] = meta
        return out

    @classmethod
    def save_cached(
        cls,
        storage,
        entity_id: str,
        payload: dict,
        *,
        source: str = "api",
    ) -> None:
        if not cls.cache_enabled():
            return
        wrapped = dict(payload)
        wrapped["_cache_meta"] = {
            "version": cls.cache_version,
            "model_key": cls.model_key,
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "computed_at_epoch": time.time(),
            "source": source,
            "entity_id": entity_id,
        }
        storage.save(cls.cache_category(), entity_id, wrapped)
        logger.info(
            "推論キャッシュ保存 [%s]: %s source=%s",
            cls.model_key,
            entity_id,
            source,
        )
