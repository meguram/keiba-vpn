"""ローカルサーバのポート定義（正は ``config/ports.env``。環境変数 ``KEIBA_PORT_*`` が優先）。"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_PORTS_FILE = Path(__file__).resolve().parent.parent.parent / "config" / "ports.env"

SERVICES = ("frontend", "flask", "fastapi", "mlflow")


@lru_cache(maxsize=1)
def _file_ports() -> dict[str, int]:
    out: dict[str, int] = {}
    for raw in _PORTS_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if "=" not in line:
            continue
        key, val = (s.strip() for s in line.split("=", 1))
        if key.startswith("KEIBA_PORT_") and val.isdigit():
            out[key] = int(val)
    return out


def _resolve(key: str) -> int:
    env_val = os.environ.get(key, "").strip()
    if env_val.isdigit():
        return int(env_val)
    return _file_ports()[key]


def port(service: str, env: str | None = None) -> int:
    """``service`` は frontend|flask|fastapi|mlflow。``env`` 省略時は現在の KEIBA_ENV（dev|stg|prod）。"""
    if service not in SERVICES:
        raise ValueError(f"unknown service: {service}")
    if env is None:
        from src.config.deployment import keiba_env

        env = keiba_env()
    if env not in ("dev", "stg", "prod"):
        raise ValueError(f"unknown env: {env}")
    return _resolve(f"KEIBA_PORT_{env.upper()}_{service.upper()}")


def monitor_port() -> int:
    return _resolve("KEIBA_PORT_MONITOR")


def postgres_port() -> int:
    return _resolve("KEIBA_PORT_POSTGRES")


def redis_port() -> int:
    return _resolve("KEIBA_PORT_REDIS")


def local_url(service: str, env: str | None = None) -> str:
    return f"http://127.0.0.1:{port(service, env)}"


def mlflow_tracking_uri() -> str:
    """``MLFLOW_TRACKING_URI`` があればそれ、なければ現在環境のローカル MLflow。"""
    return os.environ.get("MLFLOW_TRACKING_URI", "").strip() or f"http://localhost:{port('mlflow')}"
