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

logger = logging.getLogger("pipeline.models.trainer")


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
        *,
        use_split_manifest: bool | None = None,
        split_manifest_path: str | Path | None = None,
        split_protocol: str = "model_selection_primary",
    ) -> dict[str, Any]:
        """
        学習を実行して MLflow に記録する。

        Args:
            features_dir: 特徴量キャッシュ CSV ディレクトリ
            test_ratio: レガシー分割時のテスト割合（マニフェスト未使用時のみ）
            params: LightGBM パラメータ上書き
            years: 対象年度リスト (None=全年度)
            use_split_manifest: True で `dataset_split_manifest.json` に従う。
                None のとき、ファイルが存在すれば True。
            split_manifest_path: マニフェストパス（省略時は data/meta/modeling/dataset_split_manifest.json）
            split_protocol: protocols のキー（既定: model_selection_primary = 2020-23 train / 2024 valid / 2025 test）
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

        manifest_path = Path(
            split_manifest_path or "data/meta/modeling/dataset_split_manifest.json"
        )
        use_manifest = (
            use_split_manifest if use_split_manifest is not None else manifest_path.is_file()
        )
        split_mode = "legacy"
        manifest: dict[str, Any] | None = None

        if use_manifest and manifest_path.is_file():
            from src.pipeline.models.dataset_split import assign_split_column, load_manifest

            manifest = load_manifest(manifest_path)
            df["_split"] = assign_split_column(df, manifest, split_protocol)
            n_unk = int((df["_split"] == "unknown").sum())
            if n_unk:
                logger.warning(
                    "split=unknown の行を除外: %d (race_id 年がマニフェスト外)", n_unk
                )
            df = df[df["_split"] != "unknown"].copy()
            if df.empty:
                logger.error("マニフェスト分割後にデータが空です")
                return {"error": "split 後データなし"}
            split_mode = "manifest"
            logger.info(
                "マニフェスト分割 (%s): train=%d valid=%d test=%d 行",
                split_protocol,
                int((df["_split"] == "train").sum()),
                int((df["_split"] == "valid").sum()),
                int((df["_split"] == "test").sum()),
            )

        from src.pipeline.models.encoder import FeatureEncoder
        encoder = FeatureEncoder()
        if split_mode == "manifest":
            df_fit = df[df["_split"] == "train"]
            encoder.fit(df_fit, target_col="finish_position")
        else:
            encoder.fit(df, target_col="finish_position")
        Path(self.encoder_path).parent.mkdir(parents=True, exist_ok=True)
        encoder.save(self.encoder_path)

        target_full = df["finish_position"].copy()
        group_col_full = df["race_id"].copy() if "race_id" in df.columns else None
        split_col = df["_split"].copy() if split_mode == "manifest" else None

        encoded = encoder.transform(df)
        if "finish_position" in encoded.columns:
            encoded.drop(columns=["finish_position"], inplace=True)
        if split_col is not None:
            encoded["_split"] = split_col.values

        from src.pipeline.features.feature_builder import PUBLIC_INDICATOR_SET
        excluded = set(PUBLIC_INDICATOR_SET)
        fp = self.config.get("feature_policy", {})
        excluded.update(fp.get("excluded_features", []))
        whitelist = set(fp.get("smartrc_whitelist_features", []))
        excluded -= whitelist

        numeric_cols = [
            c for c in encoded.select_dtypes(include=[np.number]).columns
            if c not in excluded
        ]

        non_feature_cols = {"horse_number", "race_round", "field_size_y", "_split"}
        numeric_cols = [c for c in numeric_cols if c not in non_feature_cols]

        # race 単位で連続するようソート（LambdaRank の group 整合）
        if "race_id" in encoded.columns:
            encoded = encoded.sort_values("race_id", kind="mergesort")

        target = target_full.loc[encoded.index]
        X = encoded[numeric_cols].fillna(0)
        y = ((target >= 1) & (target <= 3)).astype(int)
        group_col = (
            group_col_full.loc[encoded.index] if group_col_full is not None else None
        )

        if split_mode == "manifest" and split_col is not None:
            sc = encoded["_split"]
            tr = sc == "train"
            va = sc == "valid"
            te = sc == "test"
            if tr.sum() == 0 or va.sum() == 0 or te.sum() == 0:
                logger.error(
                    "マニフェスト分割で train/valid/test のいずれかが空です (train=%d valid=%d test=%d)",
                    int(tr.sum()),
                    int(va.sum()),
                    int(te.sum()),
                )
                return {"error": "empty split branch"}
            X_train, y_train = X.loc[tr], y.loc[tr]
            X_valid, y_valid = X.loc[va], y.loc[va]
            X_test, y_test = X.loc[te], y.loc[te]
            gc_tr = group_col.loc[tr] if group_col is not None else None
            gc_va = group_col.loc[va] if group_col is not None else None
            gc_te = group_col.loc[te] if group_col is not None else None
            groups_train = self._group_sizes(gc_tr) if gc_tr is not None else None
            groups_valid = self._group_sizes(gc_va) if gc_va is not None else None
            groups_test = self._group_sizes(gc_te) if gc_te is not None else None
        else:
            split_idx = int(len(X) * (1 - test_ratio))
            X_train, X_valid = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_valid = y.iloc[:split_idx], y.iloc[split_idx:]
            X_test, y_test = X_valid, y_valid
            if group_col is not None:
                gc_tr = group_col.iloc[:split_idx]
                gc_va = group_col.iloc[split_idx:]
                groups_train = self._group_sizes(gc_tr)
                groups_valid = self._group_sizes(gc_va)
                groups_test = groups_valid
            else:
                groups_train = groups_valid = groups_test = None

        lgb_params = self._default_params()
        if params:
            lgb_params.update(params)

        try:
            import lightgbm as lgb
        except ImportError:
            logger.warning("LightGBM 未インストール — sklearn GBC で代替")
            if split_mode == "manifest":
                return self._train_sklearn_fallback(
                    X_train, y_train, X_test, y_test, numeric_cols
                )
            return self._train_sklearn_fallback(
                X_train, y_train, X_valid, y_valid, numeric_cols
            )

        use_lambdarank = (
            groups_train is not None
            and len(groups_train) > 0
            and all(g > 0 for g in groups_train)
            and groups_valid is not None
            and len(groups_valid) > 0
        )

        if use_lambdarank:
            train_data = lgb.Dataset(X_train, label=y_train, group=groups_train)
            valid_data = lgb.Dataset(
                X_valid, label=y_valid, group=groups_valid, reference=train_data
            )
        else:
            lgb_params["objective"] = "binary"
            lgb_params["metric"] = "auc"
            lgb_params.pop("ndcg_eval_at", None)
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

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

        # 最終報告はホールドアウト test（マニフェスト時は 2025）
        metrics = self._evaluate(model, X_test, y_test, groups_test)
        metrics["n_features"] = len(numeric_cols)
        metrics["n_train"] = len(X_train)
        metrics["n_valid"] = len(X_valid)
        metrics["n_test"] = len(X_test)
        metrics["n_races_train"] = len(groups_train) if groups_train is not None else 0
        metrics["n_races_valid"] = len(groups_valid) if groups_valid is not None else 0
        metrics["n_races_test"] = len(groups_test) if groups_test is not None else 0
        metrics["objective"] = lgb_params.get("objective", "lambdarank")
        metrics["training_time_sec"] = round(time.time() - t0, 1)
        metrics["split_mode"] = split_mode
        metrics["split_protocol"] = split_protocol if split_mode == "manifest" else "legacy_ratio"
        if manifest:
            metrics["split_manifest_version"] = manifest.get("version")

        if split_mode == "manifest":
            metrics_valid = self._evaluate(model, X_valid, y_valid, groups_valid)
            for k, v in metrics_valid.items():
                if isinstance(v, (int, float)):
                    metrics[f"valid_{k}"] = v

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

        self._log_mlflow(
            model, metrics, numeric_cols, lgb_params, manifest_path=str(manifest_path)
        )

        return metrics

    @staticmethod
    def _group_sizes(race_id_series: pd.Series) -> np.ndarray | None:
        """race_id 昇順ソート済みの系列から、各レースの頭数配列を返す。"""
        if race_id_series is None or len(race_id_series) == 0:
            return None
        return race_id_series.groupby(race_id_series, sort=False).size().values

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

    def _log_mlflow(
        self,
        model,
        metrics: dict,
        feature_names: list[str],
        lgb_params: dict,
        manifest_path: str | None = None,
    ):
        try:
            import mlflow
            import mlflow.lightgbm

            mlflow.set_tracking_uri(self.mlflow_uri)
            exp = mlflow.set_experiment(self.EXPERIMENT_NAME)

            run_name = f"lgbm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            with mlflow.start_run(run_name=run_name) as run:
                mlflow.set_tag("model_type", "lightgbm")
                mlflow.set_tag("public_indicators", "excluded")
                if metrics.get("split_mode") == "manifest":
                    mlflow.set_tag("split_mode", "manifest")
                    mlflow.set_tag("split_protocol", str(metrics.get("split_protocol", "")))
                if manifest_path:
                    mlflow.set_tag("split_manifest_path", manifest_path)

                extra_params: dict[str, Any] = {
                    "n_features": metrics["n_features"],
                    "n_train": metrics["n_train"],
                    "n_test": metrics["n_test"],
                    "n_races_train": metrics.get("n_races_train", 0),
                    "n_races_test": metrics.get("n_races_test", 0),
                    "n_optuna_trials": self._get_optuna_n_trials(),
                }
                if "n_valid" in metrics:
                    extra_params["n_valid"] = metrics["n_valid"]
                if "n_races_valid" in metrics:
                    extra_params["n_races_valid"] = metrics["n_races_valid"]
                if metrics.get("split_manifest_version") is not None:
                    extra_params["split_manifest_version"] = metrics["split_manifest_version"]

                mlflow.log_params({
                    k: str(v) for k, v in lgb_params.items()
                })
                mlflow.log_params(extra_params)

                log_metrics = {}
                for k in (
                    "auc",
                    "accuracy",
                    "log_loss",
                    "ndcg_3",
                    "ndcg_5",
                    "top1_hit_rate",
                    "top3_hit_rate",
                    "training_time_sec",
                ):
                    if k in metrics and isinstance(metrics[k], (int, float)):
                        log_metrics[k] = metrics[k]
                for k, v in metrics.items():
                    if k.startswith("valid_") and isinstance(v, (int, float)):
                        log_metrics[k] = v
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
        """特徴量を読み込む。優先順位: LayerA 学習 Parquet → FeatureStore スナップショット → 列ストア → CSV → GCS。"""

        # 0. LayerA 学習母表（pipeline.build_layer_a_dataset で生成・当日直前パイプライン整合）
        layer_a = Path("data/modeling/layer_a_train.parquet")
        if layer_a.is_file():
            logger.info("LayerA 学習 Parquet から読込: %s", layer_a)
            return pd.read_parquet(layer_a)

        # 1. FeatureStore スナップショット
        try:
            from src.pipeline.features.feature_store import FeatureStore
            store = FeatureStore()
            snapshots = store.list_snapshots()
            if snapshots:
                latest = snapshots[-1]
                logger.info("FeatureStore スナップショットから読込: %s", latest)
                return store.load_snapshot(latest)
        except Exception:
            pass

        # 2. 登録済み特徴量列
        try:
            from src.pipeline.features.feature_store import FeatureStore
            store = FeatureStore()
            cols = store.list_columns()
            if cols:
                logger.info("FeatureStore 列ストアから %d 特徴量を読込", len(cols))
                return store.load_columns(cols, years=years)
        except Exception:
            pass

        # 3. CSV キャッシュ
        p = Path(features_dir)
        if p.exists():
            csvs = list(p.glob("*.csv"))
            if csvs:
                frames = [pd.read_csv(c) for c in csvs]
                logger.info("CSV から %d ファイル読込", len(csvs))
                return pd.concat(frames, ignore_index=True)

        # 4. GCS フォールバック
        return self._build_from_gcs(features_dir, years)

    def _build_from_gcs(self, features_dir: str, years: list[str] | None = None) -> pd.DataFrame:
        """GCS から1レースずつ特徴量を構築する (レガシーフォールバック)。"""
        from src.scraper.storage import HybridStorage
        from src.pipeline.features.feature_builder import build_race_features

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
            p = Path(features_dir)
            p.mkdir(parents=True, exist_ok=True)
            csv_path = p / "all_features.csv"
            combined.to_csv(csv_path, index=False, encoding="utf-8-sig")
            logger.info("特徴量 CSV 保存: %s (%d行)", csv_path, len(combined))
            return combined

        return pd.DataFrame()
