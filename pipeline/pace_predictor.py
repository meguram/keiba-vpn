"""
ペース予測モデル（前半1F, 3Fラップタイム予測）

実際のレースデータから学習し、レース条件（距離、馬場状態、脚質分布など）から
前半ラップタイムを高精度に予測する。
"""

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

logger = logging.getLogger("pipeline.pace_predictor")

EXPERIMENT_NAME = "pace-predictor"
MODEL_NAME = "pace-lap-time-predictor"

FEATURE_COLUMNS = [
    "distance",
    "distance_category",  # 0:短距離 1:マイル 2:中距離 3:長距離
    "surface_code",  # 0:芝 1:ダート
    "track_condition_code",  # 0:良 1:稍 2:重 3:不良
    "field_size",
    "front_runner_count",  # 逃げ馬の数
    "stalker_count",  # 先行馬の数
    "early_pressure",  # 前に行きたい馬の比率
    "venue_code",  # 競馬場コード
    "grade_code",  # グレード (0:一般 1:OP 2:G3 3:G2 4:G1)
]


def build_training_dataset(storage, years: list[str] | None = None) -> pd.DataFrame:
    """
    過去レースデータから学習データセットを構築する。

    Target:
        - lap_time_1f: 最初の1F (200m) のラップタイム（秒）
        - lap_time_3f: 最初の3F (600m) のラップタイム（秒）
    """
    if years is None:
        years = ["2022", "2023", "2024", "2025"]

    result_keys = storage.list_keys("race_result")
    logger.info("race_result キー数: %d", len(result_keys))

    rows = []
    processed = 0
    skipped = 0

    for key in sorted(result_keys):
        year = key[:4]
        if year not in years:
            continue

        result_data = storage.load("race_result", key)
        if not result_data:
            skipped += 1
            continue

        # ラップタイムがあるかチェック
        lap_time_str = result_data.get("lap_time", "")
        if not lap_time_str:
            skipped += 1
            continue

        # ラップタイムをパース (例: "12.1-10.9-11.2-11.5-..." → [12.1, 10.9, ...])
        lap_times = _parse_lap_times(lap_time_str)
        if len(lap_times) < 3:
            skipped += 1
            continue

        # 前半1F, 3Fを取得
        lap_1f = lap_times[0]
        lap_3f = sum(lap_times[:3])

        # レース条件
        distance = result_data.get("distance", 0)
        surface = result_data.get("surface", "")
        track_cond = result_data.get("track_condition", "")
        field_size = result_data.get("field_size", len(result_data.get("entries", [])))
        venue = result_data.get("venue", "")
        race_name = result_data.get("race_name", "")

        if distance < 1000 or field_size < 4:
            skipped += 1
            continue

        # 脚質分布を推定（出馬表がないので簡易版）
        entries = result_data.get("entries", [])
        # 通過順から逃げ・先行を推定
        front_count = 0
        stalker_count = 0
        for e in entries:
            passing = e.get("passing_order", "")
            if passing:
                try:
                    positions = [int(x) for x in passing.split("-") if x.strip().isdigit()]
                    if positions:
                        first_pos_norm = (positions[0] - 1) / max(field_size - 1, 1)
                        if first_pos_norm <= 0.15:
                            front_count += 1
                        elif first_pos_norm <= 0.35:
                            stalker_count += 1
                except:
                    pass

        early_pressure = (front_count + stalker_count) / field_size if field_size > 0 else 0.3

        # グレード判定
        grade_code = _detect_grade(race_name)

        row = {
            "race_id": result_data.get("race_id", key),
            "distance": distance,
            "distance_category": _distance_category(distance),
            "surface_code": 0 if surface in ("芝", "芝ダ") else 1,
            "track_condition_code": _track_condition_code(track_cond),
            "field_size": field_size,
            "front_runner_count": front_count,
            "stalker_count": stalker_count,
            "early_pressure": round(early_pressure, 3),
            "venue_code": _venue_code(venue),
            "grade_code": grade_code,
            "lap_time_1f": lap_1f,
            "lap_time_3f": lap_3f,
        }

        rows.append(row)
        processed += 1

        if processed % 200 == 0:
            logger.info("処理済み: %d レース, 行数: %d", processed, len(rows))

    logger.info(
        "データセット構築完了: %d レース処理, %d スキップ, %d 行",
        processed, skipped, len(rows)
    )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def _parse_lap_times(lap_time_str: str) -> list[float]:
    """ラップタイム文字列をパース"""
    parts = lap_time_str.replace(" ", "").split("-")
    times = []
    for p in parts:
        try:
            times.append(float(p))
        except:
            pass
    return times


def _distance_category(distance: int) -> int:
    """距離カテゴリ"""
    if distance <= 1400:
        return 0  # 短距離
    elif distance <= 1800:
        return 1  # マイル
    elif distance <= 2200:
        return 2  # 中距離
    else:
        return 3  # 長距離


def _track_condition_code(cond: str) -> int:
    """馬場状態コード"""
    if cond == "良":
        return 0
    elif cond == "稍":
        return 1
    elif cond == "重":
        return 2
    elif cond == "不良":
        return 3
    else:
        return 0


def _venue_code(venue: str) -> int:
    """競馬場コード"""
    venues = {
        "札幌": 1, "函館": 2, "福島": 3, "新潟": 4, "東京": 5,
        "中山": 6, "中京": 7, "京都": 8, "阪神": 9, "小倉": 10,
    }
    return venues.get(venue, 0)


def _detect_grade(race_name: str) -> int:
    """グレード判定"""
    if "G1" in race_name or "GⅠ" in race_name:
        return 4
    elif "G2" in race_name or "GⅡ" in race_name:
        return 3
    elif "G3" in race_name or "GⅢ" in race_name:
        return 2
    elif "OP" in race_name or "オープン" in race_name or "リステッド" in race_name:
        return 1
    else:
        return 0


class PacePredictorTrainer:
    """ペース予測モデルの学習とMLflow登録"""

    def __init__(self, mlflow_tracking_uri: str = "http://localhost:5000"):
        self.mlflow_uri = mlflow_tracking_uri

    def train(
        self,
        df: pd.DataFrame | None = None,
        storage=None,
        years: list[str] | None = None,
        test_ratio: float = 0.2,
    ) -> dict[str, Any]:
        """モデルを学習してMLflowに登録"""
        t0 = time.time()

        if df is None:
            if storage is None:
                from scraper.storage import HybridStorage
                storage = HybridStorage()
            df = build_training_dataset(storage, years=years)

        if df.empty:
            return {"error": "学習データなし"}

        logger.info("学習データ: %d行", len(df))

        # 特徴量とターゲット
        X = df[FEATURE_COLUMNS].fillna(0).astype(float)
        y_1f = df["lap_time_1f"].values
        y_3f = df["lap_time_3f"].values

        # Train/Test分割
        split_idx = int(len(X) * (1 - test_ratio))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_1f_train, y_1f_test = y_1f[:split_idx], y_1f[split_idx:]
        y_3f_train, y_3f_test = y_3f[:split_idx], y_3f[split_idx:]

        try:
            import lightgbm as lgb
        except ImportError:
            logger.error("LightGBM 未インストール")
            return {"error": "LightGBM required"}

        # 1Fモデル
        lgb_params = {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }

        logger.info("1Fモデル学習中...")
        train_1f = lgb.Dataset(X_train, label=y_1f_train)
        valid_1f = lgb.Dataset(X_test, label=y_1f_test, reference=train_1f)
        model_1f = lgb.train(
            lgb_params,
            train_1f,
            num_boost_round=500,
            valid_sets=[valid_1f],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
        )

        # 3Fモデル
        logger.info("3Fモデル学習中...")
        train_3f = lgb.Dataset(X_train, label=y_3f_train)
        valid_3f = lgb.Dataset(X_test, label=y_3f_test, reference=train_3f)
        model_3f = lgb.train(
            lgb_params,
            train_3f,
            num_boost_round=500,
            valid_sets=[valid_3f],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
        )

        # 評価
        pred_1f = model_1f.predict(X_test)
        pred_3f = model_3f.predict(X_test)

        mae_1f = float(np.mean(np.abs(pred_1f - y_1f_test)))
        mae_3f = float(np.mean(np.abs(pred_3f - y_3f_test)))
        rmse_1f = float(np.sqrt(np.mean((pred_1f - y_1f_test) ** 2)))
        rmse_3f = float(np.sqrt(np.mean((pred_3f - y_3f_test) ** 2)))

        metrics = {
            "mae_1f": round(mae_1f, 3),
            "mae_3f": round(mae_3f, 3),
            "rmse_1f": round(rmse_1f, 3),
            "rmse_3f": round(rmse_3f, 3),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "training_time_sec": round(time.time() - t0, 1),
        }

        logger.info("1F: MAE=%.3f秒, RMSE=%.3f秒", mae_1f, rmse_1f)
        logger.info("3F: MAE=%.3f秒, RMSE=%.3f秒", mae_3f, rmse_3f)

        # MLflowに記録
        self._log_mlflow(model_1f, model_3f, metrics, lgb_params)

        return metrics

    def _log_mlflow(self, model_1f, model_3f, metrics: dict, params: dict):
        """MLflowに記録"""
        try:
            import mlflow
            import mlflow.lightgbm

            mlflow.set_tracking_uri(self.mlflow_uri)
            mlflow.set_experiment(EXPERIMENT_NAME)

            run_name = f"pace_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            with mlflow.start_run(run_name=run_name) as run:
                mlflow.set_tag("model_type", "pace_predictor")
                mlflow.set_tag("algorithm", "lightgbm_regression")

                mlflow.log_params({k: str(v) for k, v in params.items()})
                mlflow.log_params({
                    "n_features": len(FEATURE_COLUMNS),
                    "n_train": metrics["n_train"],
                    "n_test": metrics["n_test"],
                })

                mlflow.log_metrics({
                    "mae_1f": metrics["mae_1f"],
                    "mae_3f": metrics["mae_3f"],
                    "rmse_1f": metrics["rmse_1f"],
                    "rmse_3f": metrics["rmse_3f"],
                    "training_time_sec": metrics["training_time_sec"],
                })

                # モデルを保存
                mlflow.lightgbm.log_model(
                    model_1f,
                    artifact_path="model_1f",
                    registered_model_name=f"{MODEL_NAME}-1f",
                )
                mlflow.lightgbm.log_model(
                    model_3f,
                    artifact_path="model_3f",
                    registered_model_name=f"{MODEL_NAME}-3f",
                )

                # 特徴量情報
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False, encoding="utf-8"
                ) as f:
                    json.dump({
                        "feature_names": FEATURE_COLUMNS,
                        "metrics": metrics,
                    }, f, ensure_ascii=False, indent=2)
                    tmp_path = f.name
                mlflow.log_artifact(tmp_path, "feature_info")
                os.unlink(tmp_path)

                logger.info("MLflow 記録完了: run_id=%s", run.info.run_id)

        except Exception as e:
            logger.warning("MLflow 記録失敗: %s", e)


# モデルキャッシュ
_cached_models: dict[str, Any] = {}
_cached_models_time = 0


def load_models() -> tuple[Any, Any] | None:
    """MLflowから1F, 3Fモデルをロード（キャッシュ付き）"""
    global _cached_models, _cached_models_time

    if _cached_models and (time.time() - _cached_models_time) < 3600:
        return _cached_models.get("1f"), _cached_models.get("3f")

    try:
        import mlflow
        import mlflow.lightgbm
        from mlflow.tracking import MlflowClient

        from utils.mlflow_client import init_mlflow
        init_mlflow()

        client = MlflowClient()

        # 1Fモデル
        versions_1f = client.search_model_versions(
            f"name='{MODEL_NAME}-1f'",
            order_by=["version_number DESC"],
            max_results=1,
        )
        model_1f = None
        if versions_1f:
            v = versions_1f[0]
            model_uri = f"models:/{MODEL_NAME}-1f/{v.version}"
            model_1f = mlflow.lightgbm.load_model(model_uri)
            logger.info("1Fモデルロード: version=%s", v.version)

        # 3Fモデル
        versions_3f = client.search_model_versions(
            f"name='{MODEL_NAME}-3f'",
            order_by=["version_number DESC"],
            max_results=1,
        )
        model_3f = None
        if versions_3f:
            v = versions_3f[0]
            model_uri = f"models:/{MODEL_NAME}-3f/{v.version}"
            model_3f = mlflow.lightgbm.load_model(model_uri)
            logger.info("3Fモデルロード: version=%s", v.version)

        if model_1f and model_3f:
            _cached_models = {"1f": model_1f, "3f": model_3f}
            _cached_models_time = time.time()
            return model_1f, model_3f

    except Exception as e:
        logger.warning("MLflow モデルロード失敗: %s", e)

    # ローカルフォールバック
    local_dir = Path(__file__).parent.parent / "models" / "pace_predictor"
    if (local_dir / "model_1f.txt").exists() and (local_dir / "model_3f.txt").exists():
        import lightgbm as lgb
        model_1f = lgb.Booster(model_file=str(local_dir / "model_1f.txt"))
        model_3f = lgb.Booster(model_file=str(local_dir / "model_3f.txt"))
        _cached_models = {"1f": model_1f, "3f": model_3f}
        _cached_models_time = time.time()
        return model_1f, model_3f

    return None


def predict_lap_times(
    distance: int,
    surface: str,
    track_condition: str,
    field_size: int,
    front_runner_count: int,
    stalker_count: int,
    venue: str = "",
    race_name: str = "",
) -> dict:
    """
    レース条件から前半ラップタイムを予測

    Returns:
        {"first_1f": float, "first_3f": float}
    """
    models = load_models()

    if models is None:
        # モデルがない場合はヒューリスティック
        logger.warning("ペース予測モデルがロードできません。ヒューリスティック値を使用。")
        return _heuristic_lap_times(distance, surface, track_condition, front_runner_count, stalker_count, field_size)

    model_1f, model_3f = models

    early_pressure = (front_runner_count + stalker_count) / max(field_size, 1)

    features = {
        "distance": distance,
        "distance_category": _distance_category(distance),
        "surface_code": 0 if surface in ("芝", "芝ダ") else 1,
        "track_condition_code": _track_condition_code(track_condition),
        "field_size": field_size,
        "front_runner_count": front_runner_count,
        "stalker_count": stalker_count,
        "early_pressure": early_pressure,
        "venue_code": _venue_code(venue),
        "grade_code": _detect_grade(race_name),
    }

    X = pd.DataFrame([features])[FEATURE_COLUMNS]

    try:
        pred_1f = float(model_1f.predict(X)[0])
        pred_3f = float(model_3f.predict(X)[0])

        return {
            "first_1f": round(pred_1f, 1),
            "first_3f": round(pred_3f, 1),
        }
    except Exception as e:
        logger.error("予測エラー: %s", e)
        return _heuristic_lap_times(distance, surface, track_condition, front_runner_count, stalker_count, field_size)


def _heuristic_lap_times(distance, surface, track_cond, front_count, stalker_count, field_size) -> dict:
    """ヒューリスティックな推定値（フォールバック用）"""
    base_1f = 12.0
    base_3f = 35.0

    # 距離補正
    if distance <= 1400:
        adj = -0.3
    elif distance >= 2200:
        adj = 0.4
    else:
        adj = 0.0

    # ペース補正
    early_pressure = (front_count + stalker_count) / max(field_size, 1)
    if early_pressure > 0.5:
        adj -= 0.5
    elif early_pressure < 0.25:
        adj += 0.5

    # 馬場補正
    if track_cond in ("重", "不良"):
        adj += 0.8
    elif track_cond == "稍":
        adj += 0.3

    # 芝/ダート補正
    if surface in ("ダ", "ダート"):
        adj += 0.5

    return {
        "first_1f": round(base_1f + adj, 1),
        "first_3f": round(base_3f + adj * 3, 1),
    }
