import pytest

from src.pipeline.mlflow import train_utils


def test_raises_when_falling_back_to_local_store(monkeypatch):
    monkeypatch.setattr(train_utils, "init_mlflow", lambda uri=None: "/tmp/mlflow/runs")
    with pytest.raises(RuntimeError, match="MLflow サーバーに接続できません"):
        train_utils.require_tracking_server()


def test_returns_uri_when_server_is_reachable(monkeypatch):
    class _Client:
        def search_experiments(self, max_results=1):
            return []

    monkeypatch.setattr(train_utils, "init_mlflow", lambda uri=None: "http://127.0.0.1:23230")
    monkeypatch.setattr(train_utils.mlflow, "MlflowClient", lambda: _Client())
    assert train_utils.require_tracking_server() == "http://127.0.0.1:23230"
