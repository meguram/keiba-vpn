"""最終オッズ予測 inference サービス。"""

from __future__ import annotations

from src.pipeline.mlflow.inference_cache import InferenceCacheMixin
from src.pipeline.models.final_odds_predictor import FinalOddsPredictor, predict_final_odds


class FinalOddsCache(InferenceCacheMixin):
    model_key = "final_odds"
    cache_version = 1


def cache_enabled() -> bool:
    return FinalOddsCache.cache_enabled()


def predict_odds_for_features(features_df):
    """特徴量 DataFrame から想定オッズを返す。"""
    return predict_final_odds(features_df)
