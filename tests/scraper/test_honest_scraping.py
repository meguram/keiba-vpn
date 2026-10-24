"""取得は誠実な形式: 固定の識別可能な UA、403 でも UA を変えない、制限を検知したら別経路へ移らず保留する。"""
from __future__ import annotations

import pytest

from src.scraper import client as client_mod
from src.scraper import identity
from src.scraper.access_guard import AccessGuard, AccessRestrictionDetected

U = "https://db.netkeiba.com/horse/2020100001/"


class R:
    def __init__(self, code, body=""):
        self.status_code, self.text, self.content, self.headers = code, body, body.encode(), {}

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code}")


def test_user_agent_is_fixed_identifiable_and_not_a_browser():
    ua = identity.user_agent({})
    assert ua == "Mozilla/5.0 (compatible; keiba-research-bot/1.0)"
    ua2 = identity.user_agent({"KEIBA_SCRAPER_CONTACT": "ops@example.com"})
    assert ua2 == "Mozilla/5.0 (compatible; keiba-research-bot/1.0; +ops@example.com)"
    assert "Chrome" not in ua2 and "Firefox" not in ua2 and "Safari" not in ua2


def test_headers_have_no_browser_impersonation():
    h = client_mod._build_browser_headers({"ua": identity.user_agent({})})
    assert h["User-Agent"].startswith("Mozilla/5.0 (compatible; keiba-research-bot")
    assert not [k for k in h if k.lower().startswith(("sec-", "upgrade-insecure"))]


def test_client_never_rotates_the_user_agent(monkeypatch):
    monkeypatch.setenv("KEIBA_SCRAPER_CONTACT", "ops@example.com")
    c = client_mod.NetkeibaClient(auto_login=False)
    ua0 = dict(c._current_ua)
    monkeypatch.setattr(c, "_throttle", lambda *a, **k: None)
    monkeypatch.setattr(client_mod.time, "sleep", lambda *_: None)
    seq = iter([R(403, "x"), R(403, "x"), R(403, "x"), R(403, "x")])

    class S:
        headers = {}

        def get(self, url, timeout=None):
            return next(seq)

    real = c._session
    c._session = S()
    with pytest.raises(Exception):
        c._get_with_backoff(U)                                               # 403 を待機つきで上限回数だけ再試行して失敗
    assert c._current_ua == ua0
    c._session = real
    c._refresh_session()
    assert c._current_ua == ua0


def test_access_guard_trip_stops_without_switching_route(monkeypatch):
    pauses = []
    import src.scraper.scrape_access_pause as sp

    monkeypatch.setattr(sp, "write_access_pause", lambda *, reason: pauses.append(reason))
    g = AccessGuard(persist=True)
    with pytest.raises(AccessRestrictionDetected):
        g(U, R(403, "x"))
    assert g.tripped is not None and pauses and "403" in pauses[0]           # キュー全体の保留フラグを立てる
    with pytest.raises(AccessRestrictionDetected):
        g(U, R(200, "ok"))                                                   # 一度 trip したら以降も同じ例外で失敗（続行しない）


def test_access_hold_persists_and_clears(tmp_path):
    from src.ops import access_hold

    p = tmp_path / "h.json"
    assert access_hold.read_hold(p) is None
    access_hold.set_hold("x", task_id="t", path=p)
    assert access_hold.read_hold(p)["task_id"] == "t"
    assert access_hold.clear_hold(p) and access_hold.read_hold(p) is None


def test_no_ua_pool_or_ua_rotation_left_in_scraper_code():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for f in ("src/scraper/client.py", "src/scraper/http400_strategy.py"):
        text = (root / f).read_text(encoding="utf-8")
        assert "_UA_POOL" not in text and "random.choice(_UA" not in text
