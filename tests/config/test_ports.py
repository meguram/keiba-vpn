"""config/ports.env と src/config/ports.py / scripts/server/ports.sh の整合性。"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from src.config import ports

ROOT = Path(__file__).resolve().parents[2]
PORTS_ENV = ROOT / "config" / "ports.env"
AVOID = {80, 443, 3000, 3001, 5000, 5001, 5432, 6379, 8000, 8080, 8888, 9000, 9090}
ENVS = ("dev", "stg", "prod")


def _all_ports() -> dict[str, int]:
    out = {}
    for raw in PORTS_ENV.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = int(v)
    return out


def test_ports_are_unique_and_uncommon():
    allp = _all_ports()
    assert len(set(allp.values())) == len(allp), "ポート番号が重複している"
    for key, value in allp.items():
        assert value not in AVOID, f"{key}={value} は衝突しやすい既定ポート"
        assert 10000 <= value < 32768, f"{key}={value} は ephemeral 範囲外の 1xxxx-2xxxx にすること"


def test_every_env_has_every_service():
    for env in ENVS:
        for svc in ports.SERVICES:
            assert ports.port(svc, env) == _all_ports()[f"KEIBA_PORT_{env.upper()}_{svc.upper()}"]


def test_env_var_overrides_file(monkeypatch):
    monkeypatch.setenv("KEIBA_PORT_DEV_FLASK", "23999")
    assert ports.port("flask", "dev") == 23999


def test_current_env_resolution(monkeypatch):
    monkeypatch.setenv("KEIBA_ENV", "stg")
    assert ports.port("fastapi") == _all_ports()["KEIBA_PORT_STG_FASTAPI"]


def test_unknown_service_or_env_rejected():
    with pytest.raises(ValueError):
        ports.port("nope", "dev")
    with pytest.raises(ValueError):
        ports.port("flask", "qa")


def test_mlflow_uri_prefers_env(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    assert ports.mlflow_tracking_uri() == "http://mlflow:5000"
    monkeypatch.delenv("MLFLOW_TRACKING_URI")
    monkeypatch.setenv("KEIBA_ENV", "dev")
    assert ports.mlflow_tracking_uri() == f"http://localhost:{_all_ports()['KEIBA_PORT_DEV_MLFLOW']}"


def test_shell_loader_matches_python():
    out = subprocess.run(
        ["bash", "-c", f'source "{ROOT}/scripts/server/ports.sh"; load_keiba_ports "{ROOT}"; keiba_port stg flask; echo; echo "$KEIBA_PORT_MONITOR"'],
        capture_output=True, text=True, check=True, env={"PATH": "/usr/bin:/bin"},
    ).stdout.split()
    assert int(out[0]) == ports.port("flask", "stg")
    assert int(out[1]) == ports.monitor_port()


def test_no_hardcoded_legacy_ports_in_server_scripts():
    legacy = re.compile(r"(?<![0-9])(3000|3001|5100|8000|9090)(?![0-9])")
    for name in ("service_start.profiles.sh", "start_monitor.sh", "start_flask_api.sh", "tunnel_tcpexposer.profiles.sh"):
        text = (ROOT / "scripts" / "server" / name).read_text(encoding="utf-8")
        code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
        assert not legacy.search(code), f"{name} に旧ポートの直書きが残っている"
