"""
MLflow モデリング・推論の共通基盤。

各予測タスク（追走難度・着順・最終オッズ等）は ``catalog.ModelSpec`` で登録し、
学習・Registry ロード・Model Serving・storage キャッシュを同じパターンで扱う。
"""

from src.pipeline.mlflow.catalog import (
    MODEL_CATALOG,
    ModelFlavor,
    ModelLifecycle,
    ModelSpec,
    get_model_spec,
    iter_model_specs,
    list_model_keys,
)
from src.pipeline.mlflow.inference_cache import InferenceCacheMixin
from src.pipeline.mlflow.runtime import (
    ensure_mlruns_symlink,
    get_serve_client,
    load_lightgbm_booster,
    platform_health,
    resolve_serve_uri,
)

__all__ = [
    "MODEL_CATALOG",
    "ModelFlavor",
    "ModelLifecycle",
    "ModelSpec",
    "InferenceCacheMixin",
    "ensure_mlruns_symlink",
    "get_model_spec",
    "get_serve_client",
    "iter_model_specs",
    "list_model_keys",
    "load_lightgbm_booster",
    "platform_health",
    "resolve_serve_uri",
]
