"""
モデル学習パイプライン

大衆指標 (オッズ・人気) を一切排除したうえで LightGBM ランキングモデルを学習し、
MLflow で実験管理・モデル登録を行う。

フロー:
  1. GCS から過去 race_result / race_shutuba 等を取得
  2. build_race_features() で特徴量テーブルを構築
  3. FeatureEncoder で fit → transform
  4. LightGBM LambdaRank で学習 (group=race_id)
  5. MLflow に params / metrics / artifacts / model を記録
  6. Model Registry に登録 (自動的に latest version)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("pipeline.trainer")


class ModelTrainer:
    """大衆指標排除モデルの学習と MLflow 登録。"""

    EXPERIMENT_NAME = "keiba-no-public-indicators"
    MODEL_NAME = "keiba-lgbm-nopi"

    def __init__(
        self,
        mlflow_tracking_uri: str = "http://localhost:5000",
        encoder_path: str = "models/encoder.json",
        config: dict[str, Any] | None = None,
    ):
        self.mlflow_uri = mlflow_tracking_uri
        self.encoder_path = encoder_path
        self.config: dict[str, Any] = config if config is not None else {}

    def _get_optuna_n_trials(self) -> int:
        """
        設定ファイルの lightgbm.n_optuna_trials から Optuna の試行回数を取得する。
        未設定の場合はデフォルト値 50 を返す。
        """
        return self.config.get("lightgbm", {}).get("n_optuna_trials", 50)

    def _apply_data_freshness_policy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        data_policy に基づいて学習データをフィルタリングする。
        use_post_restructure_only=True かつ restructure_cutoff_date が設定されている場合、
        カットオフ日以降のデータのみを使用する。
        """
        data_policy = self.config.get("data_policy", {})
        if not data_policy.get("use_post_restructure_only", False):
            return df

        cutoff_date = data_policy.get("restructure_cutoff_date")
        if cutoff_date is None:
            logger.warning(
                "use_post_restructure_only=True ですが restructure_cutoff_date が未設定です。"
                "フィルタリングをスキップします。"
            )
            return df

        date_col = None
        for col in ["race_date", "date", "kaisai_date"]:
            if col in df.columns:
                date_col = col
                break

        if date_col is None:
            logger.warning("日付カラムが見つかりません。フィルタリングをスキップします。")
            return df

        original_len = len(df)
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        cutoff = pd.to_datetime(cutoff_date)
        df = df[df[date_col] >= cutoff]

        min_samples = data_policy.get("min_required_samples", 1000)
        filtered_len = len(df)
        logger.info(
            "データ鮮度フィルタ適用: %d → %d レコード (カットオフ: %s)",
            original_len,
            filtered_len,
            cutoff_date,
        )
        if filtered_len < min_samples:
            logger.warning(
                "フィルタ後サンプル数 %d が最低必要数 %d を下回っています。"
                "過学習リスクに注意してください。",
                filtered_len,
                min_samples,
            )
        return df

    def train(
        self,
        features_dir: str = "data/features",
        test_ratio: float = 0.2,
        params: dict[str, Any] | None = None,
        years: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        学習を実行して MLflow に記録する。

        Args:
            features_dir: 特徴量キャッシュ CSV ディレクトリ
            test_ratio: テストデータの割合 (時系列分割)
            params: LightGBM パラメータ上書き
            years: 対象年度リスト (None=全年度)
        """
        t0 = time.time()

        df = self._load_or_build(features_dir, years)
        if df.empty:
            logger.error("学習データが空です")
            return {"error": "学習データなし"}

        # data_policy に基づくデータフィルタリング
        df = self._apply_data_freshness_policy(df)
        if df.empty:
            logger.error("データ鮮度フィルタ適用後、学習データが空です")
            return {"error": "フィルタ後の学習データなし"}

        logger.info("学習データ: %d行 x %d列", len(df), len(df.columns))

        from pipeline.encoder import FeatureEncoder
        encoder = FeatureEncoder()
        encoder.fit(df, target_col="finish_position")
        Path(self.encoder_path).parent.mkdir(parents=True, exist_ok=True)
        encoder.save(self.encoder_path)

        target = df["finish_position"].copy()
        group_col = df["race_id"].copy() if "race_id" in df.columns else None

        encoded = encoder.transform(df)
        if "finish_position" in encoded.columns:
            encoded.drop(columns=["finish_position"], inplace=True)

        from pipeline.feature_builder import PUBLIC_INDICATOR_SET
        excluded = set(PUBLIC_INDICATOR_SET)
        fp = self.config.get("feature_policy", {})
        excluded.update(fp.get("excluded_features", []))
        whitelist = set(fp.get("smartrc_whitelist_features", []))
        excluded -= whitelist

        numeric_cols = [
            c for c in encoded.select_dtypes(include=[np.number]).columns
            if c not in excluded
        ]

        non_feature_cols = {"horse_number", "race_round", "field_size_y"}
        numeric_cols = [c for c in numeric_cols if c not in non_feature_cols]

        X = encoded[numeric_cols].fillna(0)
        y = ((target >= 1) & (target <= 3)).astype(int)

        split_idx = int(len(X) * (1 - test_ratio))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        if group_col is not None:
            groups_train = (
                group_col.iloc[:split_idx]
                .groupby(group_col.iloc[:split_idx])
                .size()
                .values
            )
            groups_test = (
                group_col.iloc[split_idx:]
                .groupby(group_col.iloc[split_idx:])
                .size()
                .values
            )
        else:
            groups_train = groups_test = None

        lgb_params = self._default_params()
        if params:
            lgb_params.update(params)

        try:
            import lightgbm as lgb
        except ImportError:
            logger.warning("LightGBM 未インストール — sklearn GBC で代替")
            return self._train_sklearn_fallback(
                X_train, y_train, X_test, y_test, numeric_cols
            )

        use_lambdarank = (
            groups_train is not None
            and len(groups_train) > 0
            and all(g > 0 for g in groups_train)
        )

        if use_lambdarank:
            train_data = lgb.Dataset(X_train, label=y_train, group=groups_train)
            valid_data = lgb.Dataset(
                X_test, label=y_test, group=groups_test, reference=train_data
            )
        else:
            lgb_params["objective"] = "binary"
            lgb_params["metric"] = "auc"
            lgb_params.pop("ndcg_eval_at", None)
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        n_est = lgb_params.pop("n_estimators", 500)
        early = lgb_params.pop("early_stopping_rounds", 50)

        callbacks = [lgb.early_stopping(early), lgb.log_evaluation(50)]
        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=n_est,
            valid_sets=[valid_data],
            callbacks=callbacks,
        )

        metrics = self._evaluate(model, X_test, y_test, groups_test)
        metrics["n_features"] = len(numeric_cols)
        metrics["n_train"] = len(X_train)
        metrics["n_test"] = len(X_test)
        metrics["n_races_train"] = len(groups_train) if groups_train is not None else 0
        metrics["n_races_test"] = len(groups_test) if groups_test is not None else 0
        metrics["objective"] = lgb_params.get("objective", "lambdarank")
        metrics["training_time_sec"] = round(time.time() - t0, 1)

        importance = dict(
            zip(
                numeric_cols,
                model.feature_importance(importance_type="gain").tolist(),
            )
        )
        metrics["top_features"] = dict(
            sorted(importance.items(), key=lambda x: -x[1])[:20]
        )
        metrics["feature_names"] = numeric_cols
        metrics["model_type"] = "lightgbm"
        metrics["public_indicators_excluded"] = True

        self._log_mlflow(model, metrics, numeric_cols, lgb_params)

        return metrics

    def _default_params(self) -> dict[str, Any]:
        return {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [3, 5],
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 10,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_estimators": 500,
            "early_stopping_rounds": 50,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
        }

    def _evaluate(self, model, X_test, y_test, groups_test) -> dict:
        preds = model.predict(X_test)

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
            metrics["ndcg_3"] = self._calc_ndcg(preds, y_test, groups_test, k=3)
            metrics["ndcg_5"] = self._calc_ndcg(preds, y_test, groups_test, k=5)
            metrics["top1_hit_rate"] = self._top_k_hit_rate(
                preds, y_test, groups_test, k=1
            )
            metrics["top3_hit_rate"] = self._top_k_hit_rate(
                preds, y_test, groups_test, k=3
            )

        return metrics

    @staticmethod
    def _calc_ndcg(preds, labels, groups, k: int) -> float:
        from sklearn.metrics import ndcg_score

        scores = []
        idx = 0
        for g in groups:
            g = int(g)
            if g < 2:
                idx += g
                continue
            y_true = labels.iloc[idx : idx + g].values.reshape(1, -1)
            y_score = preds[idx : idx + g].reshape(1, -1)
            try:
                scores.append(ndcg_score(y_true, y_score, k=k))
            except ValueError:
                pass
            idx += g
        return round(np.mean(scores), 4) if scores else 0.0

    @staticmethod
    def _top_k_hit_rate(preds, labels, groups, k: int) -> float:
        hits = 0
        total = 0
        idx = 0
        for g in groups:
            g = int(g)
            if g < 2:
                idx += g
                continue
            pred_slice = preds[idx : idx + g]
            label_slice = labels.iloc[idx : idx + g].values
            top_k_idx = np.argsort(-pred_slice)[:k]
            if any(label_slice[i] == 1 for i in top_k_idx):
                hits += 1
            total += 1
            idx += g
        return round(hits / total, 4) if total > 0 else 0.0

    def _train_sklearn_fallback(
        self, X_train, y_train, X_test, y_test, feature_names
    ) -> dict:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import roc_auc_score, accuracy_score

        clf = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42
        )
        clf.fit(X_train, y_train)

        preds = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds) if len(y_test.unique()) > 1 else 0
        acc = accuracy_score(y_test, (preds > 0.5).astype(int))

        importance = dict(zip(feature_names, clf.feature_importances_.tolist()))

        metrics = {
            "auc": round(auc, 4),
            "accuracy": round(acc, 4),
            "model_type": "sklearn_gbc",
            "n_features": len(feature_names),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "public_indicators_excluded": True,
            "top_features": dict(sorted(importance.items(), key=lambda x: -x[1])[:20]),
            "feature_names": feature_names,
        }

        self._log_mlflow_sklearn(clf, metrics, feature_names)
        return metrics

    def _log_mlflow(self, model, metrics: dict, feature_names: list[str], lgb_params: dict):
        try:
            import mlflow
            import mlflow.lightgbm

            mlflow.set_tracking_uri(self.mlflow_uri)
            exp = mlflow.set_experiment(self.EXPERIMENT_NAME)

            run_name = f"lgbm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            with mlflow.start_run(run_name=run_name) as run:
                mlflow.set_tag("model_type", "lightgbm")
                mlflow.set_tag("public_indicators", "excluded")

                mlflow.log_params({
                    k: str(v) for k, v in lgb_params.items()
                })
                mlflow.log_params({
                    "n_features": metrics["n_features"],
                    "n_train": metrics["n_train"],
                    "n_test": metrics["n_test"],
                    "n_races_train": metrics.get("n_races_train", 0),
                    "n_races_test": metrics.get("n_races_test", 0),
                    "n_optuna_trials": self._get_optuna_n_trials(),
                })

                log_metrics = {}
                for k in ("auc", "accuracy", "log_loss", "ndcg_3", "ndcg_5",
                          "top1_hit_rate", "top3_hit_rate", "training_time_sec"):
                    if k in metrics and isinstance(metrics[k], (int, float)):
                        log_metrics[k] = metrics[k]
                mlflow.log_metrics(log_metrics)

                mlflow.lightgbm.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=self.MODEL_NAME,
                )

                if os.path.exists(self.encoder_path):
                    mlflow.log_artifact(self.encoder_path, artifact_path="encoder")

                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False, encoding="utf-8"
                ) as f:
                    json.dump(
                        {
                            "feature_names": feature_names,
                            "public_indicators_excluded": True,
                            "top_features": metrics.get("top_features", {}),
                            "created_at": datetime.now().isoformat(),
                            "n_optuna_trials": self._get_optuna_n_trials(),
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                    tmp_path = f.name
                try:
                    mlflow.log_artifact(tmp_path, artifact_path="feature_info")
                finally:
                    os.unlink(tmp_path)

                logger.info(
                    "MLflow 記録完了: experiment=%s run=%s",
                    self.EXPERIMENT_NAME,
                    run.info.run_id,
                )
                metrics["mlflow_run_id"] = run.info.run_id
                metrics["mlflow_experiment_id"] = exp.experiment_id

        except Exception as e:
            logger.warning("MLflow 記録失敗: %s", e)
            metrics["mlflow_error"] = str(e)

    def _log_mlflow_sklearn(self, clf, metrics: dict, feature_names: list[str]):
        try:
            import mlflow
            import mlflow.sklearn

            mlflow.set_tracking_uri(self.mlflow_uri)
            mlflow.set_experiment(self.EXPERIMENT_NAME)

            run_name = f"sklearn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with mlflow.start_run(run_name=run_name) as run:
                mlflow.set_tag("model_type", "sklearn_gbc")
                mlflow.set_tag("public_indicators", "excluded")
                mlflow.log_params({
                    "n_features": len(feature_names),
                    "n_optuna_trials": self._get_optuna_n_trials(),
                })
                mlflow.log_metrics({"auc": metrics["auc"], "accuracy": metrics["accuracy"]})
                mlflow.sklearn.log_model(
                    clf,
                    artifact_path="model",
                    registered_model_name=self.MODEL_NAME + "-sklearn",
                )
                logger.info("MLflow 記録完了 (sklearn): run=%s", run.info.run_id)
                metrics["mlflow_run_id"] = run.info.run_id
        except Exception as e:
            logger.warning("MLflow 記録失敗 (sklearn): %s", e)

    def _load_or_build(self, features_dir: str, years: list[str] | None = None) -> pd.DataFrame:
        """特徴量CSVまたはGCSデータからDataFrameを構築する。"""
        p = Path(features_dir)
        if p.exists():
            csvs = list(p.glob("*.csv"))
            if csvs:
                frames = [pd.read_csv(c) for c in csvs]
                logger.info("CSV から %d ファイル読込", len(csvs))
                return pd.concat(frames, ignore_index=True)

        from scraper.storage import HybridStorage
        from pipeline.feature_builder import build_race_features

        storage = HybridStorage(".")
        if not storage.gcs_enabled:
            logger.warning("GCS 未接続")
            return pd.DataFrame()

        if years is None:
            years = storage.list_years("race_result")
        if not years:
            return pd.DataFrame()

        race_ids = []
        for y in years:
            race_ids.extend(storage.list_keys("race_result", y))

        logger.info("GCS から %d レースの特徴量を構築中...", len(race_ids))

        frames = []
        for i, race_id in enumerate(race_ids, 1):
            if i % 10 == 0:
                logger.info("  特徴量構築: %d/%d", i, len(race_ids))

            race_data = {"race_id": race_id}
            subdir_map = {
                "race_shutuba": "race_card",
                "race_index": "speed_index",
                "race_shutuba_past": "shutuba_past",
                "race_paddock": "paddock",
                "race_barometer": "barometer",
                "race_oikiri": "oikiri",
            }
            for cat, key in subdir_map.items():
                d = storage.load(cat, race_id)
                if d:
                    race_data[key] = d

            race_data["horses"] = {}
            card = race_data.get("race_card", {})
            for entry in card.get("entries", []):
                hid = entry.get("horse_id", "")
                if hid:
                    hdata = storage.load("horse_result", hid)
                    if hdata:
                        race_data["horses"][hid] = hdata

            try:
                df = build_race_features(race_data)
                if not df.empty:
                    result_data = storage.load("race_result", race_id) or {}
                    result_entries = {
                        e["horse_number"]: e
                        for e in result_data.get("entries", [])
                    }
                    df["finish_position"] = df["horse_number"].map(
                        lambda hn, re=result_entries: re.get(hn, {}).get(
                            "finish_position", 0
                        )
                    )
                    frames.append(df)
            except Exception as e:
                logger.warning("特徴量構築失敗 [%s]: %s", race_id, e)

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            p.mkdir(parents=True, exist_ok=True)
            csv_path = p / "all_features.csv"
            combined.to_csv(csv_path, index=False, encoding="utf-8-sig")
            logger.info("特徴量 CSV 保存: %s (%d行)", csv_path, len(combined))
            return combined

        return pd.DataFrame()
