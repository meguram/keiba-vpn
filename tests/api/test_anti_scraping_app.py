"""実際の FastAPI アプリに載せたときの結合確認（ミドルウェアの順序・内部ジョブ・ハニーポット・未ログイン）。"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from starlette.testclient import TestClient  # noqa: E402

from src.api.app import app  # noqa: E402
from src.utils import anti_scraping as AS  # noqa: E402

BROWSER = {
    "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
    "accept-language": "ja",
    "sec-fetch-mode": "cors",
    "referer": "https://example.test/",
}


@pytest.fixture()
def client():
    eng = AS.get_engine()
    saved = (eng._cfg, eng._cfg_loaded)
    eng._cfg = AS._merge(AS.DEFAULTS, {"mode": {"auth": "enforce", "rate_limit": "enforce", "bot": "off", "honeypot": "enforce", "login": "enforce", "session": "off"},
                                       "rate_limit": {"data_per_min": 8, "data_burst_per_10s": 50}})
    eng._cfg_loaded = 1e18  # ファイルを読み直さない
    eng._blocked.clear(); eng._windows.clear(); eng._viol.clear(); eng._ids.clear()
    yield TestClient(app, client=("127.0.0.1", 50000))
    eng._cfg, eng._cfg_loaded = saved


def h(ip: str) -> dict:
    return {**BROWSER, "x-forwarded-for": ip}


def test_anonymous_data_api_is_401_when_enforced(client):
    r = client.get("/api/race/202604040201/predictions", headers=h("203.0.113.10"))
    assert r.status_code == 401 and r.json()["reason"] == "login_required"
    assert r.headers["x-robots-tag"].startswith("noindex")


def test_health_and_auth_status_stay_public(client):
    assert client.get("/api/health", headers=h("203.0.113.11")).status_code == 200
    assert client.get("/api/auth/status", headers=h("203.0.113.11")).status_code == 200


def test_internal_loopback_job_without_forward_headers_is_untouched(client):
    for _ in range(20):
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/scrape-jobs").status_code == 200


def test_honeypot_blocks_the_ip_everywhere(client):
    assert client.get("/api/_hp/probe", headers=h("203.0.113.12")).status_code == 403
    assert client.get("/api/health", headers=h("203.0.113.12")).status_code in (200, 403)  # 内部パスは対象外
    assert client.get("/api/race/202604040201/predictions", headers=h("203.0.113.12")).status_code == 403
    assert client.get("/api/race/202604040201/predictions", headers=h("203.0.113.99")).status_code == 401  # 別の接続元は別扱い


def test_rate_limit_applies_per_client(client):
    codes = [client.get(f"/api/x{i}", headers=h("203.0.113.13")).status_code for i in range(14)]
    assert 429 in codes or 403 in codes
    assert client.get("/api/x0", headers=h("203.0.113.14")).status_code != 429  # 別のクライアントは巻き添えにならない


def test_login_page_is_reachable_and_login_lockout(client, monkeypatch):
    monkeypatch.setenv("DEV_PASSWORD", "pw-for-test")
    monkeypatch.setenv("DEV_SECRET_KEY", "k" * 20)
    hd = h("203.0.113.15")
    codes = [client.post("/login", data={"password": "wrong", "next": "/"}, headers=hd).status_code for _ in range(7)]
    assert codes[:5] == [200] * 5 and codes[5] == 429
    ok = client.post("/login", data={"password": "pw-for-test", "next": "/"}, headers=h("203.0.113.16"), follow_redirects=False)
    assert ok.status_code in (302, 303) and "keiba_dev_session" in ok.headers.get("set-cookie", "")
