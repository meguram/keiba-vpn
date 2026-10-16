"""サービス状態 API が環境（dev/stg/prod）ごとに FastAPI・MLflow まで返し、旧キーはこのマシンの環境の値になること。
stg のホストで、dev のポートを見て『Flask と Next.js が停止』と出し続けた不具合（2026-10-16）の再発防止。"""

from __future__ import annotations

import pytest

from src.monitor import app as M


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("KEIBA_ENV", "stg")
    up = {M.ENV_CONFIGS["stg"]["next_url"], f"{M.ENV_CONFIGS['stg']['flask_url']}/api/v1/health", f"{M.ENV_CONFIGS['stg']['fastapi_url']}/api/health", M.ENV_CONFIGS["stg"]["mlflow_url"]}

    def fake_get(url, timeout=5.0):
        return (200, {"ok": True}, 3.0) if url in up or url.startswith(M.FASTAPI_URL) else (0, None, -1.0)

    monkeypatch.setattr(M, "_http_get", fake_get)
    monkeypatch.setattr(M, "_db_ping", lambda url: {"ok": True, "elapsed_ms": 1})
    monkeypatch.setattr(M, "_check_model_data_mode", lambda url: None)
    monkeypatch.setattr(M, "_git_head_short", lambda: "abc")
    c = M.app.test_client()
    with c.session_transaction() as s:
        s["monitor_auth"] = True
    return c


def test_each_env_reports_all_four_services_with_its_own_ports(client):
    d = client.get("/api/internal/status").get_json()
    assert set(d["envs"]) == {"dev", "stg", "prod"} and d["local_env"] == "stg"
    ports = {e: {k: d["envs"][e][k]["url"].rsplit(":", 1)[-1] for k in ("next", "flask", "fastapi", "mlflow")} for e in d["envs"]}
    assert ports["stg"] == {"next": "23200", "flask": "23210", "fastapi": "23220", "mlflow": "23230"}
    assert ports["dev"]["next"] == "23100" and ports["prod"]["flask"] == "23310"


def test_local_env_is_flagged_and_up_while_other_envs_are_down(client):
    envs = client.get("/api/internal/status").get_json()["envs"]
    assert envs["stg"]["is_local"] and envs["stg"]["all_ok"]
    assert not envs["dev"]["is_local"] and not envs["dev"]["next"]["ok"]
    assert not envs["prod"]["is_local"]


def test_legacy_service_keys_follow_the_local_env_not_dev(client):
    d = client.get("/api/internal/status").get_json()
    assert d["services"]["flask"]["ok"] and d["services"]["nextjs"]["ok"], "stg のホストで Flask/Next.js が停止と出ていた不具合"
    assert d["services"]["flask"]["url"].endswith(":23210") and d["services"]["nextjs"]["url"].endswith(":23200")
