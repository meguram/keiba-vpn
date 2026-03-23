"""
アンサンブル学習パイプライン

LightGBM / XGBoost / CatBoost / 軽量NN の4モデルをスタッキングで統合し、
単一モデルより高精度な予測を実現する。

アーキテクチャ:
  Layer 1 (Base Learners):
    - LightGBM LambdaRank (既存モデルの発展)
    - XGBoost binary:logistic
    - CatBoost with native categorical support
    - 2-layer MLP (PyTorch or sklearn MLPClassifier)

  Layer 2 (Meta Learner):
    - Logistic Regression (out-of-fold predictions → 最終スコア)

フロー:
  1. 特徴量・ターゲットを準備 (ModelTrainer と共通)
  2. K-Fold (GroupKFold by race_id) で各 base learner の OOF 予測を生成
  3. OOF 予測をスタック → meta learner を学習
  4. テストデータで base learner 予測 → meta learner で統合
  5. MLflow に全モデル + メトリクスを記録
"""

from __future__ import annotations

import json
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("pipeline.ensemble")
warnings.filterwarnings("ignore", category=UserWarning)


class EnsembleTrainer:
    """LightGBM + XGBoost + CatBoost + NN のスタッキングアンサンブル"""

    EXPERIMENT_NAME = "keiba-ensemble"
    MODEL_NAME = "keiba-ensemble-v1"

    def __init__(
        self,
        mlflow_tracking_uri: str = "http://localhost:5000",
        encoder_path: str = "models/encoder.json",
        config: dict[str, Any] | None = None,
        n_folds: int = 5,
    ):
        self.mlflow_uri = mlflow_tracking_uri
        self.encoder_path = encoder_path
        self.config = config or {}
        self.n_folds = n_folds

        self._base_models: dict[str, Any] = {}
        self._meta_model: Any = None
        self._feature_names: list[str] = []
        self._available_learners: list[str] = []

    def train(
        self,
        features_dir: str = "data/features",
        test_ratio: float = 0.2,
        years: list[str] | None = None,
    ) -> dict[str, Any]:
        t0 = time.time()

        X, y, groups, numeric_cols, encoder = self._prepare_data(
            features_dir, years
        )
        if X is None:
            return {"error": "学習データなし"}

        self._feature_names = numeric_cols

        split_idx = int(len(X) * (1 - test_ratio))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        race_ids_train = groups.iloc[:split_idx] if groups is not None else None
        race_ids_test = groups.iloc[split_idx:] if groups is not None else None

        groups_test = (
            race_ids_test.groupby(race_ids_test).size().values
            if race_ids_test is not None else None
        )

        self._available_learners = self._detect_available_learners()
        logger.info("利用可能なベースモデル: %s", self._available_learners)

        if len(self._available_learners) < 2:
            logger.warning("2つ以上のモデルが必要です。LightGBM 単体に fallback")
            from pipeline.trainer import ModelTrainer
            fallback = ModelTrainer(
                mlflow_tracking_uri=self.mlflow_uri,
                encoder_path=self.encoder_path,
                config=self.config,
            )
            return fallback.train(features_dir, test_ratio, years=years)

        # ── Layer 1: OOF predictions ──
        logger.info("=" * 60)
        logger.info("Layer 1: Base Learner OOF 予測生成 (K=%d)", self.n_folds)

        oof_train = np.zeros((len(X_train), len(self._available_learners)))
        test_preds = np.zeros((len(X_test), len(self._available_learners)))

        from sklearn.model_selection import GroupKFold, KFold

        if race_ids_train is not None:
            kf = GroupKFold(n_splits=self.n_folds)
            split_iter = kf.split(X_train, y_train, groups=race_ids_train)
        else:
            kf = KFold(n_splits=self.n_folds, shuffle=False)
            split_iter = kf.split(X_train, y_train)

        folds = list(split_iter)

        individual_metrics: dict[str, dict] = {}

        for i, learner_name in enumerate(self._available_learners):
            logger.info("--- %s (%d/%d) ---",
                        learner_name, i + 1, len(self._available_learners))
            oof_col, test_col, model = self._train_base_learner(
                learner_name, X_train, y_train, X_test, folds,
                race_ids_train,
            )
            oof_train[:, i] = oof_col
            test_preds[:, i] = test_col
            self._base_models[learner_name] = model

            from sklearn.metrics import roc_auc_score
            if len(y_test.unique()) > 1:
                auc = round(roc_auc_score(y_test, test_col), 4)
            else:
                auc = 0.0
            individual_metrics[learner_name] = {"auc": auc}
            logger.info("  %s AUC (test): %.4f", learner_name, auc)

        # ── Layer 2: Meta Learner ──
        logger.info("=" * 60)
        logger.info("Layer 2: Meta Learner (LogisticRegression)")

        from sklearn.linear_model import LogisticRegression
        meta = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        valid_mask = ~np.isnan(oof_train).any(axis=1)
        meta.fit(oof_train[valid_mask], y_train.values[valid_mask])
        self._meta_model = meta

        final_preds = meta.predict_proba(test_preds)[:, 1]

        # ── Evaluation ──
        metrics = self._evaluate(final_preds, y_test, groups_test)
        metrics["n_features"] = len(numeric_cols)
        metrics["n_train"] = len(X_train)
        metrics["n_test"] = len(X_test)
        metrics["n_folds"] = self.n_folds
        metrics["base_learners"] = self._available_learners
        metrics["individual_metrics"] = individual_metrics
        metrics["model_type"] = "ensemble"
        metrics["training_time_sec"] = round(time.time() - t0, 1)

        meta_weights = dict(zip(
            self._available_learners,
            meta.coef_[0].tolist(),
        ))
        metrics["meta_weights"] = meta_weights
        logger.info("Meta weights: %s", meta_weights)

        self._save_models(numeric_cols)
        self._log_mlflow(metrics, numeric_cols)

        return metrics

    # ── データ準備 ──

    def _prepare_data(
        self, features_dir: str, years: list[str] | None,
    ) -> tuple:
        from pipeline.trainer import ModelTrainer
        from pipeline.encoder import FeatureEncoder
        from pipeline.feature_builder import PUBLIC_INDICATOR_SET

        base_trainer = ModelTrainer(
            mlflow_tracking_uri=self.mlflow_uri,
            encoder_path=self.encoder_path,
            config=self.config,
        )
        df = base_trainer._load_or_build(features_dir, years)
        if df.empty:
            return None, None, None, None, None

        df = base_trainer._apply_data_freshness_policy(df)
        if df.empty:
            return None, None, None, None, None

        logger.info("学習データ: %d行 x %d列", len(df), len(df.columns))

        encoder = FeatureEncoder()
        encoder.fit(df, target_col="finish_position")
        Path(self.encoder_path).parent.mkdir(parents=True, exist_ok=True)
        encoder.save(self.encoder_path)

        target = df["finish_position"].copy()
        groups = df["race_id"].copy() if "race_id" in df.columns else None

        encoded = encoder.transform(df)
        if "finish_position" in encoded.columns:
            encoded.drop(columns=["finish_position"], inplace=True)

        excluded = set(PUBLIC_INDICATOR_SET)
        fp = self.config.get("feature_policy", {})
        excluded.update(fp.get("excluded_features", []))
        whitelist = set(fp.get("smartrc_whitelist_features", []))
        excluded -= whitelist

        numeric_cols = [
            c for c in encoded.select_dtypes(include=[np.number]).columns
            if c not in excluded
        ]
        non_feature = {"horse_number", "race_round", "field_size_y"}
        numeric_cols = [c for c in numeric_cols if c not in non_feature]

        X = encoded[numeric_cols].fillna(0)
        y = ((target >= 1) & (target <= 3)).astype(int)

        return X, y, groups, numeric_cols, encoder

    # ── ベースモデル検出 ──

    @staticmethod
    def _detect_available_learners() -> list[str]:
        available = []

        try:
            import lightgbm
            available.append("lightgbm")
        except ImportError:
            pass

        try:
            import xgboost
            available.append("xgboost")
        except ImportError:
            pass

        try:
            import catboost
            available.append("catboost")
        except ImportError:
            pass

        try:
            from sklearn.neural_network import MLPClassifier
            available.append("mlp")
        except ImportError:
            pass

        return available

    # ── ベースモデル学習 (OOF) ──

    def _train_base_learner(
        self,
        name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        folds: list[tuple],
        race_ids_train: pd.Series | None,
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        """K-Fold で OOF 予測を生成し、全 fold のテスト予測を平均する。"""

        oof = np.full(len(X_train), np.nan)
        test_preds = np.zeros(len(X_test))
        final_model = None

        for fold_idx, (tr_idx, va_idx) in enumerate(folds):
            X_tr = X_train.iloc[tr_idx]
            y_tr = y_train.iloc[tr_idx]
            X_va = X_train.iloc[va_idx]
            y_va = y_train.iloc[va_idx]

            groups_tr = None
            if race_ids_train is not None and name == "lightgbm":
                g = race_ids_train.iloc[tr_idx]
                groups_tr = g.groupby(g).size().values

            model = self._fit_single(name, X_tr, y_tr, X_va, y_va, groups_tr)
            oof[va_idx] = self._predict_single(name, model, X_va)
            test_preds += self._predict_single(name, model, X_test) / len(folds)
            final_model = model

        return oof, test_preds, final_model

    def _fit_single(
        self, name: str,
        X_tr: pd.DataFrame, y_tr: pd.Series,
        X_va: pd.DataFrame, y_va: pd.Series,
        groups_tr: np.ndarray | None = None,
    ) -> Any:
        if name == "lightgbm":
            return self._fit_lgb(X_tr, y_tr, X_va, y_va, groups_tr)
        elif name == "xgboost":
            return self._fit_xgb(X_tr, y_tr, X_va, y_va)
        elif name == "catboost":
            return self._fit_catboost(X_tr, y_tr, X_va, y_va)
        elif name == "mlp":
            return self._fit_mlp(X_tr, y_tr, X_va, y_va)
        raise ValueError(f"Unknown learner: {name}")

    def _predict_single(self, name: str, model: Any, X: pd.DataFrame) -> np.ndarray:
        if name == "lightgbm":
            return model.predict(X)
        elif name == "xgboost":
            import xgboost as xgb
            return model.predict(xgb.DMatrix(X))
        elif name == "catboost":
            return model.predict_proba(X)[:, 1]
        elif name == "mlp":
            X_scaled = model._scaler.transform(X) if hasattr(model, "_scaler") else X
            return model.predict_proba(X_scaled)[:, 1]
        raise ValueError(f"Unknown learner: {name}")

    # ── 個別モデル fit ──

    def _fit_lgb(self, X_tr, y_tr, X_va, y_va, groups_tr):
        import lightgbm as lgb

        use_rank = groups_tr is not None and len(groups_tr) > 0

        params = {
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 10,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
        }

        if use_rank:
            params["objective"] = "lambdarank"
            params["metric"] = "ndcg"
            params["ndcg_eval_at"] = [3, 5]
            train_data = lgb.Dataset(X_tr, label=y_tr, group=groups_tr)
            va_groups = y_va.groupby(
                pd.Series(range(len(y_va)))).size().values  # dummy
            valid_data = lgb.Dataset(X_va, label=y_va, reference=train_data)
        else:
            params["objective"] = "binary"
            params["metric"] = "auc"
            train_data = lgb.Dataset(X_tr, label=y_tr)
            valid_data = lgb.Dataset(X_va, label=y_va, reference=train_data)

        callbacks = [lgb.early_stopping(30), lgb.log_evaluation(0)]
        model = lgb.train(
            params, train_data, num_boost_round=300,
            valid_sets=[valid_data], callbacks=callbacks,
        )
        return model

    @staticmethod
    def _fit_xgb(X_tr, y_tr, X_va, y_va):
        import xgboost as xgb

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_va, label=y_va)

        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "verbosity": 0,
        }

        model = xgb.train(
            params, dtrain, num_boost_round=300,
            evals=[(dval, "eval")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )
        return model

    @staticmethod
    def _fit_catboost(X_tr, y_tr, X_va, y_va):
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            eval_metric="AUC",
            verbose=0,
            early_stopping_rounds=30,
            random_seed=42,
        )
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
        return model

    @staticmethod
    def _fit_mlp(X_tr, y_tr, X_va, y_va):
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=256,
            learning_rate="adaptive",
            learning_rate_init=0.001,
            max_iter=100,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
            random_state=42,
            verbose=False,
        )
        model.fit(X_tr_s, y_tr)
        model._scaler = scaler
        return model

    # ── 評価 ──

    @staticmethod
    def _evaluate(preds, y_test, groups_test) -> dict:
        from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

        metrics: dict[str, Any] = {}

        if len(y_test.unique()) > 1:
            metrics["auc"] = round(roc_auc_score(y_test, preds), 4)
            pred_prob = np.clip(preds, 1e-7, 1 - 1e-7)
            metrics["log_loss"] = round(log_loss(y_test, pred_prob), 4)
        else:
            metrics["auc"] = 0.0
            metrics["log_loss"] = 0.0

        pred_binary = (preds > 0.5).astype(int)
        metrics["accuracy"] = round(accuracy_score(y_test, pred_binary), 4)

        if groups_test is not None and len(groups_test) > 0:
            from pipeline.trainer import ModelTrainer
            metrics["ndcg_3"] = ModelTrainer._calc_ndcg(preds, y_test, groups_test, 3)
            metrics["ndcg_5"] = ModelTrainer._calc_ndcg(preds, y_test, groups_test, 5)
            metrics["top1_hit_rate"] = ModelTrainer._top_k_hit_rate(
                preds, y_test, groups_test, 1,
            )
            metrics["top3_hit_rate"] = ModelTrainer._top_k_hit_rate(
                preds, y_test, groups_test, 3,
            )
        return metrics

    # ── 保存 ──

    def _save_models(self, feature_names: list[str]):
        models_dir = Path("models/ensemble")
        models_dir.mkdir(parents=True, exist_ok=True)

        import pickle

        for name, model in self._base_models.items():
            path = models_dir / f"{name}_model.pkl"
            with open(path, "wb") as f:
                pickle.dump(model, f)
            logger.info("保存: %s", path)

        if self._meta_model is not None:
            with open(models_dir / "meta_model.pkl", "wb") as f:
                pickle.dump(self._meta_model, f)

        meta_info = {
            "base_learners": self._available_learners,
            "feature_names": feature_names,
            "n_folds": self.n_folds,
        }
        with open(models_dir / "ensemble_meta.json", "w") as f:
            json.dump(meta_info, f, indent=2, ensure_ascii=False)

    def _log_mlflow(self, metrics: dict, feature_names: list[str]):
        try:
            import mlflow
            import mlflow.sklearn

            mlflow.set_tracking_uri(self.mlflow_uri)
            mlflow.set_experiment(self.EXPERIMENT_NAME)

            with mlflow.start_run(run_name=f"ensemble-{len(self._available_learners)}models"):
                mlflow.log_params({
                    "model_type": "ensemble_stacking",
                    "base_learners": ",".join(self._available_learners),
                    "n_folds": self.n_folds,
                    "n_features": len(feature_names),
                })

                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v)

                for k, v in metrics.get("individual_metrics", {}).items():
                    for mk, mv in v.items():
                        mlflow.log_metric(f"{k}_{mk}", mv)

                if self._meta_model is not None:
                    mlflow.sklearn.log_model(
                        self._meta_model, "meta_model",
                        registered_model_name=self.MODEL_NAME,
                    )

                models_dir = Path("models/ensemble")
                for path in models_dir.glob("*.pkl"):
                    mlflow.log_artifact(str(path), "base_models")
                for path in models_dir.glob("*.json"):
                    mlflow.log_artifact(str(path))

                metrics["mlflow_run_id"] = mlflow.active_run().info.run_id

        except Exception as e:
            logger.warning("MLflow 記録失敗: %s", e)
            metrics["mlflow_error"] = str(e)

    # ── 推論 ──

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """学習済みアンサンブルで予測する。"""
        if not self._base_models or self._meta_model is None:
            self._load_models()

        base_preds = np.zeros((len(X), len(self._available_learners)))
        for i, name in enumerate(self._available_learners):
            model = self._base_models[name]
            base_preds[:, i] = self._predict_single(name, model, X)

        return self._meta_model.predict_proba(base_preds)[:, 1]

    def _load_models(self):
        import pickle

        models_dir = Path("models/ensemble")
        meta_path = models_dir / "ensemble_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError("アンサンブルモデルが見つかりません")

        with open(meta_path) as f:
            meta = json.load(f)
        self._available_learners = meta["base_learners"]
        self._feature_names = meta["feature_names"]

        for name in self._available_learners:
            path = models_dir / f"{name}_model.pkl"
            with open(path, "rb") as f:
                self._base_models[name] = pickle.load(f)

        with open(models_dir / "meta_model.pkl", "rb") as f:
            self._meta_model = pickle.load(f)

        logger.info("アンサンブルモデルをロード: %s", self._available_learners)
