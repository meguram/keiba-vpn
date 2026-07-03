"""
MLflow ランタイム — Tracking 初期化・Registry ロード・Model Serving クライアント。
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import yaml

from src.pipeline.mlflow.catalog import MODEL_CATALOG, ModelFlavor, ModelSpec, get_model_spec
from src.utils.logger import get_logger
from src.utils.mlflow_client import init_mlflow

logger = get_logger("MLflowRuntime")

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SETTINGS_PATH = _PROJECT_ROOT / "config" / "settings.yaml"
_SERVE_TIMEOUT = float(os.environ.get("KEIBA_MLFLOW_SERVE_TIMEOUT_SEC", "30"))
_HEALTH_CACHE_SEC = 60.0

_booster_cache: dict[str, Any] = {}
_booster_cache_time: dict[str, float] = {}
_serve_clients: dict[str, Any] = {}
_serve_client_uris: dict[str, str] = {}
_CACHE_TTL_SEC = 3600.0


def _load_settings() -> dict:
    if not _SETTINGS_PATH.is_file():
        return {}
    with open(_SETTINGS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def model_settings(key: str) -> dict:
    """settings.yaml の mlflow.models.<key> ブロック。"""
    cfg = _load_settings().get("mlflow") or {}
    models = cfg.get("models") or {}
    block = models.get(key)
    return dict(block) if isinstance(block, dict) else {}


def resolve_serve_uri(key: str) -> str:
    """Serving のベース URI（未設定なら空文字）。"""
    spec = get_model_spec(key)
    for env_name in spec.legacy_serve_env_vars:
        v = os.environ.get(env_name, "").strip()
        if v:
            return v.rstrip("/")

    ms = model_settings(key)
    uri = str(ms.get(spec.serve_uri_setting_key or "serve_uri") or "").strip()
    if uri:
        return uri.rstrip("/")

    port = ms.get("serve_port") or spec.default_serve_port
    if port:
        return f"http://localhost:{int(port)}"
    return ""


def ensure_mlruns_symlink() -> Path | None:
    """
    Registry artifact が ``mlruns/`` を指す場合に ``mlflow/runs`` へリンクする。

    Returns:
        作成したリンクの Path、既存または不要なら None
    """
    link = _PROJECT_ROOT / "mlruns"
    target = _PROJECT_ROOT / "mlflow" / "runs"
    if not target.is_dir():
        return None
    if link.is_symlink():
        return link
    if link.exists():
        return None
    try:
        link.symlink_to("mlflow/runs")
        logger.info("mlruns → mlflow/runs シンボリックリンクを作成")
        return link
    except OSError as e:
        logger.debug("mlruns リンク作成スキップ: %s", e)
        return None


class MlflowServeClient:
    """MLflow ``models serve`` の /invocations ラッパー（モデル非依存）。"""

    def __init__(self, base_uri: str, *, model_key: str = "", timeout: float = _SERVE_TIMEOUT):
        self.base_uri = base_uri.rstrip("/")
        self.model_key = model_key
        self.timeout = timeout
        self._last_health_ok = 0.0
        self._health_ok = False

    def is_available(self, *, force_check: bool = False) -> bool:
        import requests

        now = time.time()
        if not force_check and (now - self._last_health_ok) < _HEALTH_CACHE_SEC:
            return self._health_ok
        try:
            r = requests.get(
                f"{self.base_uri}/health",
                timeout=min(5.0, self.timeout),
            )
            self._health_ok = r.status_code == 200
        except Exception:
            self._health_ok = False
        self._last_health_ok = now
        return self._health_ok

    def predict_dataframe(
        self,
        columns: list[str],
        rows: list[list[Any]],
    ) -> list[float]:
        import requests

        if not rows:
            return []
        payload = {
            "dataframe_split": {
                "columns": columns,
                "data": rows,
            }
        }
        t0 = time.perf_counter()
        resp = requests.post(
            f"{self.base_uri}/invocations",
            json=payload,
            timeout=self.timeout,
            headers={"Content-Type": "application/json"},
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()
        data = resp.json()
        preds = _parse_predictions(data, len(rows))
        logger.info(
            "MLflow serve [%s]: n=%d %.0fms %s",
            self.model_key or "?",
            len(rows),
            elapsed_ms,
            self.base_uri,
        )
        return preds


def _parse_predictions(data: Any, n: int) -> list[float]:
    if isinstance(data, dict):
        if "predictions" in data:
            data = data["predictions"]
        elif "outputs" in data:
            data = data["outputs"]
    if not isinstance(data, list):
        raise ValueError(f"unexpected invocations response: {type(data)}")
    out: list[float] = []
    for item in data:
        if isinstance(item, (list, tuple)):
            out.append(float(item[0]))
        else:
            out.append(float(item))
    if len(out) != n:
        raise ValueError(f"prediction length mismatch: got {len(out)} expected {n}")
    return out


def get_serve_client(key: str) -> MlflowServeClient | None:
    uri = resolve_serve_uri(key)
    if not uri:
        return None
    if key in _serve_clients and _serve_client_uris.get(key) == uri:
        return _serve_clients[key]
    client = MlflowServeClient(uri, model_key=key)
    _serve_clients[key] = client
    _serve_client_uris[key] = uri
    return client


def _resolve_local_booster_path(spec: ModelSpec) -> Path | None:
    for rel in spec.local_booster_relpaths:
        p = _PROJECT_ROOT / rel
        if p.is_file():
            return p

    try:
        from mlflow.tracking import MlflowClient

        init_mlflow()
        client = MlflowClient()
        versions = client.search_model_versions(
            f"name='{spec.registered_name}'",
            order_by=["version_number DESC"],
            max_results=1,
        )
        if versions:
            run = client.get_run(versions[0].run_id)
            uri = run.info.artifact_uri or ""
            if uri.startswith("file:"):
                base = Path(uri.replace("file://", "", 1))
                for name in ("model.lgb", "model/model.lgb"):
                    candidate = base / name
                    if candidate.is_file():
                        return candidate
                for sub in ("model", ""):
                    folder = base / sub if sub else base
                    if folder.is_dir():
                        for ext in ("*.lgb", "*.txt"):
                            hits = sorted(folder.glob(ext), key=lambda p: p.stat().st_mtime, reverse=True)
                            if hits:
                                return hits[0]
    except Exception as e:
        logger.debug("Registry artifact パス解決失敗 [%s]: %s", spec.key, e)

    return None


def load_lightgbm_booster(
    key: str,
    *,
    force_reload: bool = False,
    cache_ttl_sec: float = _CACHE_TTL_SEC,
) -> Any | None:
    """
    Registry またはローカル artifact から LightGBM Booster をロードする。

    Returns:
        lightgbm.Booster または None
    """
    spec = get_model_spec(key)
    if spec.flavor != ModelFlavor.LIGHTGBM:
        raise ValueError(f"{key} は LightGBM ではありません: {spec.flavor}")

    now = time.time()
    if (
        not force_reload
        and key in _booster_cache
        and (now - _booster_cache_time.get(key, 0)) < cache_ttl_sec
    ):
        return _booster_cache[key]

    ensure_mlruns_symlink()
    model = None

    try:
        import mlflow.lightgbm
        from mlflow.tracking import MlflowClient

        init_mlflow()
        client = MlflowClient()
        versions = client.search_model_versions(
            f"name='{spec.registered_name}'",
            order_by=["version_number DESC"],
            max_results=1,
        )
        if versions:
            v = versions[0]
            model_uri = f"models:/{spec.registered_name}/{v.version}"
            model = mlflow.lightgbm.load_model(model_uri)
            logger.info(
                "Booster ロード [%s]: Registry v%s",
                key,
                v.version,
            )
    except Exception as e:
        logger.warning("Registry ロード失敗 [%s]: %s", key, e)

    if model is None:
        lgb_path = _resolve_local_booster_path(spec)
        if lgb_path is not None:
            import lightgbm as lgb

            model = lgb.Booster(model_file=str(lgb_path))
            logger.info("Booster ロード [%s]: ローカル %s", key, lgb_path)

    if model is not None:
        _booster_cache[key] = model
        _booster_cache_time[key] = now
    return model


def booster_feature_names(key: str) -> list[str]:
    """ロード済み Booster の feature_name（未ロード時は空）。"""
    model = load_lightgbm_booster(key)
    if model is None:
        return []
    names = getattr(model, "feature_name", lambda: [])()
    return list(names) if names else []


def infer_backend_label(key: str) -> str:
    """推論バックエンドのラベル（health / compute_meta 用）。"""
    client = get_serve_client(key)
    if client is not None and client.is_available():
        return "mlflow_serve"
    if load_lightgbm_booster(key) is not None:
        return "local_mlflow"
    return "heuristic"


def _tracking_health(tracking_uri: str) -> dict:
    import requests

    t0 = time.perf_counter()
    out: dict[str, Any] = {"uri": tracking_uri}
    if str(tracking_uri).startswith("http"):
        try:
            r = requests.get(f"{tracking_uri.rstrip('/')}/health", timeout=5)
            out["ok"] = r.status_code == 200
            out["error"] = None
        except Exception as e:
            out["ok"] = False
            out["error"] = str(e)
    else:
        out["ok"] = True
        out["error"] = "file_store"
    out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


def model_health(key: str, *, force_serve_check: bool = False) -> dict:
    """単一モデルの Serving / ローカル Booster 状態。"""
    spec = get_model_spec(key)
    serve_uri = resolve_serve_uri(key)
    client = get_serve_client(key)
    serve_ok = False
    serve_ms = None
    if client:
        t0 = time.perf_counter()
        serve_ok = client.is_available(force_check=force_serve_check)
        serve_ms = round((time.perf_counter() - t0) * 1000, 1)
    booster = load_lightgbm_booster(key)
    return {
        "key": key,
        "title": spec.title,
        "lifecycle": spec.lifecycle.value,
        "registered_name": spec.registered_name,
        "experiment_name": spec.experiment_name,
        "serve_uri": serve_uri or None,
        "serve_ok": serve_ok,
        "serve_ms": serve_ms,
        "local_booster_loaded": booster is not None,
        "n_features": len(booster_feature_names(key)) if booster else 0,
        "infer_backend": infer_backend_label(key),
        "cache_category": spec.cache_category,
    }


def platform_health(*, force_serve_check: bool = False) -> dict:
    """全モデル + Tracking のヘルススナップショット。"""
    tracking_uri = init_mlflow()
    models = [
        model_health(k, force_serve_check=force_serve_check)
        for k in sorted(MODEL_CATALOG)
    ]
    return {
        "mlflow_tracking": _tracking_health(tracking_uri),
        "models": models,
    }
