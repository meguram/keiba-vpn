"""
MLflow モデルカタログ — 学習・Registry・Serving・キャッシュの正規定義。

新しい予測タスクを追加する手順:
  1. ``MODEL_CATALOG`` に ``ModelSpec`` を追加（lifecycle=planned でも可）
  2. ``config/settings.yaml`` の ``mlflow.models.<key>`` を追記
  3. 必要なら ``mlflow/server/docker-compose.yml`` に serve サービスを追加
  4. ``src/pipeline/inference/<task>_service.py`` で ``InferenceCacheMixin`` を継承
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterator


class ModelFlavor(str, Enum):
    LIGHTGBM = "lightgbm"
    SKLEARN = "sklearn"


class ModelLifecycle(str, Enum):
    """active=本番利用可, planned=設計のみ・未実装推論可"""

    ACTIVE = "active"
    PLANNED = "planned"


@dataclass(frozen=True)
class ModelSpec:
    """1 つの Registry モデルとその運用メタデータ。"""

    key: str
    title: str
    description: str
    experiment_name: str
    registered_name: str
    flavor: ModelFlavor = ModelFlavor.LIGHTGBM
    lifecycle: ModelLifecycle = ModelLifecycle.ACTIVE
    # Model Serving（未設定時はローカル Booster / ヒューリスティックへフォールバック）
    default_serve_port: int | None = None
    # settings.yaml: mlflow.models.<key>.serve_uri
    serve_uri_setting_key: str | None = None
    # 後方互換用の環境変数（先頭が優先）
    legacy_serve_env_vars: tuple[str, ...] = ()
    # HybridStorage キャッシュ（API 事前計算）
    cache_category: str | None = None
    cache_enabled_env: str | None = None
    cache_ttl_env: str | None = None
    # ローカル Booster フォールバック（mlflow/runs 探索より優先）
    local_booster_relpaths: tuple[str, ...] = ()
    local_artifact_glob: str = "**/artifacts/model.lgb"


# ── カタログ（キー = settings.yaml の mlflow.models.<key> と一致）──

MODEL_CATALOG: dict[str, ModelSpec] = {
    "keiba_lgbm": ModelSpec(
        key="keiba_lgbm",
        title="着順予測（既存 LGBM）",
        description="メインの着順・複勝圏予測（trainer.py）。今後 finish_order に統合予定。",
        experiment_name="keiba-prediction",
        registered_name="keiba-lgbm",
        flavor=ModelFlavor.LIGHTGBM,
        lifecycle=ModelLifecycle.ACTIVE,
        default_serve_port=5010,
        serve_uri_setting_key="serve_uri",
        legacy_serve_env_vars=("KEIBA_MLFLOW_SERVE_KEIBA_LGBM_URI",),
        cache_category="race_predictions",
        cache_enabled_env="KEIBA_RACE_PREDICTIONS_CACHE",
        cache_ttl_env="KEIBA_RACE_PREDICTIONS_CACHE_TTL_SEC",
        local_booster_relpaths=("models/keiba_lgbm.txt", "models/keiba_lgbm.lgb"),
    ),
    "tracking_difficulty": ModelSpec(
        key="tracking_difficulty",
        title="追走難度・位置取り",
        description="枠順・隣接・前走から追走しやすさと位置取りフローを推定。",
        experiment_name="tracking-difficulty",
        registered_name="tracking-difficulty-lgbm",
        flavor=ModelFlavor.LIGHTGBM,
        lifecycle=ModelLifecycle.ACTIVE,
        default_serve_port=5001,
        serve_uri_setting_key="serve_uri",
        legacy_serve_env_vars=(
            "KEIBA_MLFLOW_SERVE_TRACKING_DIFFICULTY_URI",
            "KEIBA_MLFLOW_SERVE_TRACKING_URI",
        ),
        cache_category="tracking_difficulty",
        cache_enabled_env="KEIBA_TRACKING_DIFFICULTY_CACHE",
        cache_ttl_env="KEIBA_TRACKING_DIFFICULTY_CACHE_TTL_SEC",
        local_booster_relpaths=(
            "models/tracking_difficulty.txt",
            "models/tracking_difficulty.lgb",
        ),
    ),
    "finish_order": ModelSpec(
        key="finish_order",
        title="着順予測（次期）",
        description="レース単位の着順・複勝圏確率。特徴量ストア連携の本番モデル。",
        experiment_name="finish-order",
        registered_name="keiba-finish-order-lgbm",
        flavor=ModelFlavor.LIGHTGBM,
        lifecycle=ModelLifecycle.PLANNED,
        default_serve_port=5002,
        serve_uri_setting_key="serve_uri",
        legacy_serve_env_vars=("KEIBA_MLFLOW_SERVE_FINISH_ORDER_URI",),
        cache_category="finish_order_prediction",
        cache_enabled_env="KEIBA_FINISH_ORDER_CACHE",
        cache_ttl_env="KEIBA_FINISH_ORDER_CACHE_TTL_SEC",
        local_booster_relpaths=("models/finish_order.lgb",),
    ),
    "final_odds": ModelSpec(
        key="final_odds",
        title="最終オッズ予測",
        description="発走前の単勝・複勝オッズ推定（2020-24学習/2025評価・豊富な非市場特徴）。",
        experiment_name="final-odds",
        registered_name="keiba-final-odds-lgbm",
        flavor=ModelFlavor.LIGHTGBM,
        lifecycle=ModelLifecycle.ACTIVE,
        default_serve_port=5003,
        serve_uri_setting_key="serve_uri",
        legacy_serve_env_vars=("KEIBA_MLFLOW_SERVE_FINAL_ODDS_URI",),
        cache_category="final_odds_prediction",
        cache_enabled_env="KEIBA_FINAL_ODDS_CACHE",
        cache_ttl_env="KEIBA_FINAL_ODDS_CACHE_TTL_SEC",
        local_booster_relpaths=(
            "models/final_odds.lgb",
            "models/odds_predictor.json",
        ),
    ),
    "pace_predictor": ModelSpec(
        key="pace_predictor",
        title="ペース予測（1F/3F）",
        description="レースペースの 1 角・ 3 角タイム予測（pace_predictor.py）。",
        experiment_name="pace-prediction",
        registered_name="pace-predictor-lgbm",
        flavor=ModelFlavor.LIGHTGBM,
        lifecycle=ModelLifecycle.ACTIVE,
        default_serve_port=5004,
        serve_uri_setting_key="serve_uri",
        legacy_serve_env_vars=("KEIBA_MLFLOW_SERVE_PACE_URI",),
        local_booster_relpaths=(),
    ),
}


def get_model_spec(key: str) -> ModelSpec:
    if key not in MODEL_CATALOG:
        raise KeyError(
            f"未知のモデルキー: {key}. 登録済み: {', '.join(sorted(MODEL_CATALOG))}"
        )
    return MODEL_CATALOG[key]


def list_model_keys(*, lifecycle: ModelLifecycle | None = None) -> list[str]:
    keys = []
    for k, spec in MODEL_CATALOG.items():
        if lifecycle is None or spec.lifecycle == lifecycle:
            keys.append(k)
    return sorted(keys)


def iter_model_specs() -> Iterator[ModelSpec]:
    for key in list_model_keys():
        yield MODEL_CATALOG[key]
