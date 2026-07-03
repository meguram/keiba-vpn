"""MLflow / ローカル bundle による最終オッズ推論。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.pipeline.models.final_odds_features import enrich_final_odds_features, select_model_feature_columns
from src.pipeline.models.odds_predictor import PredictedOdds

logger = logging.getLogger(__name__)

MLFLOW_MODEL_KEY = "final_odds"
LOCAL_BUNDLE_PATH = Path("models/final_odds_bundle.json")
HEADS = ("win", "place_min", "place_max")


class FinalOddsPredictor:
    """3ヘッド（単勝・複勝min・複勝max）で想定オッズを推定。"""

    def __init__(self):
        self._bundle: dict | None = None
        self._models: dict[str, Any] = {}

    def load(self) -> bool:
        if self._models:
            return True
        if LOCAL_BUNDLE_PATH.exists():
            if self._load_local(LOCAL_BUNDLE_PATH):
                return True
        return self._load_mlflow_artifacts()

    def _load_local(self, path: Path) -> bool:
        try:
            import lightgbm as lgb

            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self._bundle = data
            for name, head in data.get("heads", {}).items():
                self._models[name] = lgb.Booster(model_str=head["model_txt"])
            logger.info("final_odds bundle ロード: %s (%d heads)", path, len(self._models))
            return bool(self._models)
        except Exception as exc:
            logger.warning("final_odds ローカルロード失敗: %s", exc)
            return False

    def _load_mlflow_artifacts(self) -> bool:
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            from src.pipeline.mlflow.catalog import get_model_spec

            spec = get_model_spec(MLFLOW_MODEL_KEY)
            client = MlflowClient()
            versions = client.get_latest_versions(spec.registered_name, stages=["None", "Staging", "Production"])
            if not versions:
                versions = client.get_latest_versions(spec.registered_name)
            if not versions:
                return False
            run_id = versions[0].run_id
            import lightgbm as lgb

            for head in HEADS:
                local = mlflow.artifacts.download_artifacts(
                    run_id=run_id,
                    artifact_path=f"model_{head}",
                )
                model_dir = Path(local)
                model_files = list(model_dir.rglob("model.lgb")) + list(model_dir.rglob("*.lgb"))
                if model_files:
                    self._models[head] = lgb.Booster(model_file=str(model_files[0]))
            bundle_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path="final_odds_bundle",
            )
            for p in Path(bundle_path).glob("*.json"):
                with open(p, encoding="utf-8") as f:
                    self._bundle = json.load(f)
                break
            return bool(self._models.get("win"))
        except Exception as exc:
            logger.debug("final_odds MLflow ロード失敗: %s", exc)
            return False

    def predict(self, features_df: pd.DataFrame) -> list[PredictedOdds]:
        if features_df.empty:
            return []
        if not self.load():
            from src.pipeline.models.odds_predictor import OddsPredictor

            return OddsPredictor().predict(features_df)

        enriched = enrich_final_odds_features(features_df)
        feature_cols = self._bundle.get("feature_cols") if self._bundle else select_model_feature_columns(enriched)
        available = [c for c in feature_cols if c in enriched.columns]
        missing = [c for c in feature_cols if c not in enriched.columns]
        X = enriched[available].fillna(0).astype(float)
        if missing:
            for c in missing:
                X[c] = 0.0
            X = X[feature_cols]

        preds: dict[str, np.ndarray] = {}
        for head in HEADS:
            m = self._models.get(head)
            if m is None:
                continue
            preds[head] = m.predict(X)

        if "win" not in preds:
            from src.pipeline.models.odds_predictor import OddsPredictor

            return OddsPredictor().predict(features_df)

        results: list[PredictedOdds] = []
        for i, (_, row) in enumerate(enriched.iterrows()):
            win_odds = max(float(np.exp(preds["win"][i])), 1.0)
            if "place_min" in preds:
                place_min = max(float(np.exp(preds["place_min"][i])), 1.0)
            else:
                place_min = max(win_odds * 0.25, 1.0)
            if "place_max" in preds:
                place_max = max(float(np.exp(preds["place_max"][i])), place_min + 0.1)
            else:
                place_max = max(win_odds * 0.55, place_min + 0.1)

            top2_est = round(win_odds * 1.65, 1)
            results.append(
                PredictedOdds(
                    horse_number=int(row.get("horse_number", i + 1)),
                    predicted_win_odds=round(win_odds, 1),
                    predicted_place_odds_min=round(place_min, 1),
                    predicted_place_odds_max=round(place_max, 1),
                    confidence=0.82,
                    source="final_odds_ml",
                )
            )
            # attach top2 on dict path via get_predicted_odds extension
        return results


def predict_final_odds(features_df: pd.DataFrame) -> list[PredictedOdds]:
    return FinalOddsPredictor().predict(features_df)
