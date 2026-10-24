"""監視ポータルの cron 状態 API(/api/internal/cron-status)の認証と応答のテスト。"""

from __future__ import annotations

import pytest

from src.monitor import app as monitor_app


@pytest.fixture
def client(monkeypatch, tmp_path):
    crontab = tmp_path / "crontab.txt"
    crontab.write_text("0 9 * * 5 bash /p/scripts/cron/cron_run.sh weekly-update -- bash /p/scripts/cron/run_auto_scrape_logged.sh /p weekly-update logs/weekly_update.log\n")
    monkeypatch.setenv("KEIBA_CRONTAB_FILE", str(crontab))
    monkeypatch.setenv("KEIBA_CRON_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr("src.monitor.cron_status.RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr("src.monitor.cron_status.LOG_DIR", tmp_path / "logs")
    monitor_app.app.config["TESTING"] = True
    return monitor_app.app.test_client()


def test_cron_status_requires_login(client):
    r = client.get("/api/internal/cron-status")
    assert r.status_code == 302 and "/login" in r.headers["Location"]
    assert client.get("/cron").status_code == 302


def test_cron_status_returns_the_agreed_shape(client):
    with client.session_transaction() as s:
        s["monitor_auth"] = True
    d = client.get("/api/internal/cron-status").get_json()
    assert set(d) >= {"generated_at", "server_tz", "summary", "groups"}
    jobs = [j for g in d["groups"] for j in g["jobs"]]
    assert [j["id"] for j in jobs] == ["weekly-update"] and jobs[0]["state"] == "waiting" and jobs[0]["schedule"] == "金 18:00 JST"
