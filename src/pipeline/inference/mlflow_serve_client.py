"""
MLflow Model Serving クライアント（後方互換エイリアス）。

新規コードは ``src.pipeline.mlflow.runtime.get_serve_client(model_key)`` を使用する。
"""

from __future__ import annotations

from src.pipeline.mlflow.runtime import MlflowServeClient, get_serve_client


def get_tracking_serve_client() -> MlflowServeClient | None:
    """追走難度モデル用 Serving クライアント（``tracking_difficulty``）。"""
    return get_serve_client("tracking_difficulty")


__all__ = ["MlflowServeClient", "get_serve_client", "get_tracking_serve_client"]
