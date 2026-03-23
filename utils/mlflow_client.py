"""
MLflow 接続・実験管理ユーティリティ

学習時のメトリクス記録、モデル登録、モデルロードを一元管理する。
MLflow サーバーが起動していなくてもローカルファイルベースで動作する。
"""

from __future__ import annotations

import os
from typing import Any

import yaml
import mlflow
from mlflow.tracking import MlflowClient

from utils.logger import get_logger

logger = get_logger("MLflow")

_DEFAULT_EXPERIMENT = "keiba-prediction"
_DEFAULT_MODEL_NAME = "keiba-lgbm"


def _load_mlflow_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("mlflow", {})
    return {}


def init_mlflow(tracking_uri: str | None = None) -> str:
    """
    MLflow を初期化する。

    tracking_uri が指定されなければ settings.yaml から読み込む。
    サーバーに接続できなければローカルファイルストアにフォールバック。

    Returns:
        使用中の tracking URI
    """
    cfg = _load_mlflow_config()
    uri = tracking_uri or cfg.get("tracking_uri", "mlruns")

    try:
        mlflow.set_tracking_uri(uri)
        if uri.startswith("http"):
            import requests as _req
            resp = _req.get(f"{uri.rstrip('/')}/health", timeout=3)
            resp.raise_for_status()
        logger.info("MLflow 接続: %s", uri)
    except Exception:
        fallback = os.path.join(os.path.dirname(__file__), "..", "mlruns")
        os.makedirs(fallback, exist_ok=True)
        mlflow.set_tracking_uri(fallback)
        uri = fallback
        logger.warning("MLflow サーバーに接続不可。ローカルストアを使用: %s", uri)

    return uri


def get_or_create_experiment(name: str | None = None) -> str:
    """実験を取得または作成し、experiment_id を返す。"""
    cfg = _load_mlflow_config()
    exp_name = name or cfg.get("experiment_name", _DEFAULT_EXPERIMENT)

    experiment = mlflow.get_experiment_by_name(exp_name)
    if experiment is None:
        exp_id = mlflow.create_experiment(exp_name)
        logger.info("実験作成: %s (id=%s)", exp_name, exp_id)
    else:
        exp_id = experiment.experiment_id
        logger.info("実験使用: %s (id=%s)", exp_name, exp_id)

    mlflow.set_experiment(exp_name)
    return exp_id


def log_training_run(
    params: dict[str, Any],
    metrics: dict[str, float],
    model: Any,
    feature_names: list[str],
    feature_importance: dict[str, Any],
    spec_summary: str = "",
    tags: dict[str, str] | None = None,
) -> str:
    """
    学習結果を MLflow に記録し、モデルを登録する。

    Returns:
        run_id
    """
    cfg = _load_mlflow_config()
    model_name = cfg.get("registered_model_name", _DEFAULT_MODEL_NAME)

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        if tags:
            mlflow.set_tags(tags)
        mlflow.set_tag("spec_summary", spec_summary[:250] if spec_summary else "")

        # 特徴量の重要度を artifact として保存
        import json
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "feature_names": feature_names,
                "importance": feature_importance,
            }, f, ensure_ascii=False, indent=2)
            tmp_path = f.name
        mlflow.log_artifact(tmp_path, "feature_info")
        os.unlink(tmp_path)

        mlflow.lightgbm.log_model(
            model,
            name="model",
            registered_model_name=model_name,
            input_example=None,
        )

        logger.info(
            "MLflow run 記録完了: run_id=%s  accuracy=%.4f  f1=%.4f  auc=%.4f",
            run_id, metrics.get("accuracy", 0), metrics.get("f1_score", 0), metrics.get("roc_auc", 0),
        )

    return run_id


def load_latest_model(model_name: str | None = None) -> tuple[Any, dict]:
    """
    MLflow Model Registry から最新の Production/最新バージョンのモデルをロードする。

    Returns:
        (model, run_info_dict)
    """
    cfg = _load_mlflow_config()
    name = model_name or cfg.get("registered_model_name", _DEFAULT_MODEL_NAME)

    client = MlflowClient()

    # まず Production ステージを探す、なければ最新バージョン
    model_uri = None
    run_id = None
    try:
        # search_model_versions (非推奨の get_latest_versions の代替)
        versions = client.search_model_versions(f"name='{name}'", order_by=["version_number DESC"], max_results=5)
        if not versions:
            raise ValueError(f"モデル '{name}' のバージョンが見つかりません")

        target = versions[0]
        model_uri = f"models:/{name}/{target.version}"
        run_id = target.run_id
        logger.info("モデルロード: %s version=%s", name, target.version)
    except Exception as e:
        logger.warning("Model Registry からの読み込みに失敗: %s", e)
        # フォールバック: 直近の run からロード
        experiment = mlflow.get_experiment_by_name(
            cfg.get("experiment_name", _DEFAULT_EXPERIMENT)
        )
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
            )
            if runs:
                run_id = runs[0].info.run_id
                model_uri = f"runs:/{run_id}/model"
                logger.info("フォールバック: 直近 run からロード run_id=%s", run_id)

    if model_uri is None:
        raise RuntimeError(f"ロード可能なモデルが見つかりません (model_name={name})")

    model = mlflow.lightgbm.load_model(model_uri)

    run_info = {}
    if run_id:
        run_data = client.get_run(run_id)
        run_info = {
            "run_id": run_id,
            "model_uri": model_uri,
            "params": dict(run_data.data.params),
            "metrics": dict(run_data.data.metrics),
            "tags": dict(run_data.data.tags),
            "start_time": str(run_data.info.start_time),
        }

    return model, run_info


def get_experiment_history(limit: int = 10) -> list[dict]:
    """直近の学習履歴を返す。"""
    cfg = _load_mlflow_config()
    exp_name = cfg.get("experiment_name", _DEFAULT_EXPERIMENT)

    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(exp_name)
    if not experiment:
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=limit,
    )

    history = []
    for run in runs:
        history.append({
            "run_id": run.info.run_id,
            "start_time": str(run.info.start_time),
            "status": run.info.status,
            "params": dict(run.data.params),
            "metrics": dict(run.data.metrics),
        })
    return history
