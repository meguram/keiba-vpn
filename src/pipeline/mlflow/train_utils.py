"""
学習スクリプト向け MLflow 登録ヘルパー（カタログ連携）。
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import mlflow
import mlflow.lightgbm

from src.pipeline.mlflow.catalog import ModelFlavor, ModelSpec, get_model_spec
from src.utils.logger import get_logger
from src.utils.mlflow_client import init_mlflow

logger = get_logger("MLflowTrain")


def start_training_run(
    spec: ModelSpec | str,
    *,
    params: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
    tracking_uri: str | None = None,
) -> Any:
    """
    カタログの experiment に紐づく MLflow run を開始する。

    Usage::

        with start_training_run("finish_order", params={...}) as run:
            ...
    """
    if isinstance(spec, str):
        spec = get_model_spec(spec)
    init_mlflow(tracking_uri)
    mlflow.set_experiment(spec.experiment_name)
    run_tags = {"model_key": spec.key, "registered_name": spec.registered_name}
    if tags:
        run_tags.update(tags)
    return mlflow.start_run(tags=run_tags)


def log_lightgbm_and_register(
    spec: ModelSpec | str,
    model: Any,
    *,
    metrics: dict[str, float],
    params: dict[str, Any] | None = None,
    feature_names: list[str] | None = None,
    feature_importance: dict[str, Any] | None = None,
    artifact_name: str = "model",
) -> str:
    """
    実行中 run に LightGBM を記録し Model Registry へ登録する。

    Returns:
        run_id
    """
    if isinstance(spec, str):
        spec = get_model_spec(spec)
    if spec.flavor != ModelFlavor.LIGHTGBM:
        raise ValueError(f"LightGBM 以外は未対応: {spec.key}")

    if params:
        mlflow.log_params({k: str(v) for k, v in params.items()})
    if metrics:
        mlflow.log_metrics(metrics)

    if feature_names or feature_importance:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "feature_names": feature_names or [],
                    "importance": feature_importance or {},
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
            tmp_path = f.name
        mlflow.log_artifact(tmp_path, "feature_info")
        os.unlink(tmp_path)

    mlflow.lightgbm.log_model(
        model,
        name=artifact_name,
        registered_model_name=spec.registered_name,
        input_example=None,
    )
    run_id = mlflow.active_run().info.run_id if mlflow.active_run() else ""
    logger.info(
        "Registry 登録 [%s]: %s run_id=%s",
        spec.key,
        spec.registered_name,
        run_id,
    )
    return run_id
