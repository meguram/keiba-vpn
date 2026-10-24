"""学習済みアンサンブルの推論専用ローダー（学習クラスに依存しない）。

``EnsembleTrainer`` が保存する形式（``ensemble_meta.json`` と ``<learner>_model.pkl``、
``meta_model.pkl``）を読み込んで予測する。推論環境（VPS/GCP）は学習用の状態を持たず、
``model_registry.fetch_latest`` で取得したディレクトリをここに渡すだけでよい。

特徴量は**列名と順序を学習時と一致**させてから渡す（``ensemble_meta.json`` の
``feature_names``）。足りない列があれば例外にして、静かな誤予測を防ぐ。
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class FeatureMismatchError(ValueError):
    """推論に渡した特徴量が、学習時の列と一致しない。"""


class EnsemblePredictor:
    def __init__(
        self,
        learners: list[str],
        base_models: dict[str, Any],
        meta_model: Any,
        feature_names: list[str],
        *,
        version: str | None = None,
        manifest: dict | None = None,
    ):
        self.learners = learners
        self.base_models = base_models
        self.meta_model = meta_model
        self.feature_names = feature_names
        self.version = version
        self.manifest = manifest or {}

    @classmethod
    def load(cls, model_dir: str | Path) -> "EnsemblePredictor":
        d = Path(model_dir)
        meta_path = d / "ensemble_meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"アンサンブルモデルが見つかりません: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        base_models: dict[str, Any] = {}
        for name in meta["base_learners"]:
            with open(d / f"{name}_model.pkl", "rb") as f:
                base_models[name] = pickle.load(f)
        with open(d / "meta_model.pkl", "rb") as f:
            meta_model = pickle.load(f)

        manifest_path = d / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        return cls(
            list(meta["base_learners"]),
            base_models,
            meta_model,
            list(meta["feature_names"]),
            version=manifest.get("version"),
            manifest=manifest,
        )

    @staticmethod
    def _predict_single(name: str, model: Any, X: pd.DataFrame) -> np.ndarray:
        if name == "lightgbm":
            return model.predict(X)
        if name == "xgboost":
            import xgboost as xgb

            return model.predict(xgb.DMatrix(X))
        if name == "catboost":
            return model.predict_proba(X)[:, 1]
        if name == "mlp":
            X_in = model._scaler.transform(X) if hasattr(model, "_scaler") else X
            return model.predict_proba(X_in)[:, 1]
        raise ValueError(f"Unknown learner: {name}")

    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """学習時の列名・順序に揃える。足りない列があれば ``FeatureMismatchError``。"""
        missing = [c for c in self.feature_names if c not in df.columns]
        if missing:
            head = ", ".join(missing[:5])
            raise FeatureMismatchError(
                f"学習時の特徴量が {len(missing)} 列足りません（例: {head}）"
            )
        return df[self.feature_names]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self.select_features(df)
        base = np.zeros((len(X), len(self.learners)))
        for i, name in enumerate(self.learners):
            base[:, i] = self._predict_single(name, self.base_models[name], X)
        return self.meta_model.predict_proba(base)[:, 1]
