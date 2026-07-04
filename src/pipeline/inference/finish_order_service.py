"""
着順予測 inference サービス（次期実装用スタブ）。

実装時は ``tracking_difficulty_service.py`` と同様に
``InferenceCacheMixin`` + ``build_*_response`` + ``get_or_compute`` を提供する。
"""

from __future__ import annotations

from src.pipeline.mlflow.inference_cache import InferenceCacheMixin


class FinishOrderCache(InferenceCacheMixin):
    model_key = "finish_order"
    cache_version = 1


def cache_enabled() -> bool:
    return FinishOrderCache.cache_enabled()
