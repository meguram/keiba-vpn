"""SmartRC クライアントのアクセス制限への退避のテスト。実際のサイトには接続しない。"""

from __future__ import annotations

import json
import threading

import pytest

from src.scraper import smartrc_client as S


class _Resp:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code = status
        self._body = body if body is not None else {"success": True, "data": []}
        self.headers = headers or {}

    def json(self):
        return self._body


@pytest.fixture
def client(tmp_path, monkeypatch):
    sleeps = []
    me = threading.get_ident()   # time.sleep はプロセス全体で差し替わるので、他のスレッド（前のテストの残り等）の sleep は数えない
    monkeypatch.setattr(S.time, "sleep", lambda s: sleeps.append(s) if threading.get_ident() == me else None)
    c = S.SmartRCClient(interval=0.0, jitter=0.0, backoff=(1, 2), cooldown=600, state_path=tmp_path / "state.json")
    c.sleeps = sleeps
    return c


def _script(monkeypatch, responses):
    calls = []

    def post(url, **kw):
        calls.append(url)
        r = responses[min(len(calls) - 1, len(responses) - 1)]
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(S.requests, "post", post)
    return calls


def test_transient_429_is_retried_after_waiting_then_succeeds(client, monkeypatch):
    calls = _script(monkeypatch, [_Resp(429), _Resp(200, {"success": True, "data": [1]})])
    assert client._request("runners", {"rcode": "x"})["data"] == [1]
    assert len(calls) == 2 and client.sleeps == [1]


def test_retry_after_header_is_honoured(client, monkeypatch):
    _script(monkeypatch, [_Resp(429, headers={"Retry-After": "40"}), _Resp(200)])
    client._request("days")
    assert client.sleeps == [40.0]


def test_persistent_restriction_raises_and_blocks_further_requests(client, monkeypatch, tmp_path):
    calls = _script(monkeypatch, [_Resp(403)])
    with pytest.raises(S.SmartRCRestricted):
        client._request("runners", {"rcode": "x"})
    assert len(calls) == 3 and client.sleeps == [1, 2]                    # 初回 + 再試行 2 回
    assert json.loads((tmp_path / "state.json").read_text())["status"] == 403
    n = len(calls)
    with pytest.raises(S.SmartRCRestricted):                              # 退避中は一切リクエストしない
        client._request("runners", {"rcode": "y"})
    assert len(calls) == n
    other = S.SmartRCClient(interval=0.0, jitter=0.0, state_path=tmp_path / "state.json")       # 別プロセス(次の cron)も記録を見て止まる
    with pytest.raises(S.SmartRCRestricted):
        other._request("days")
    assert len(calls) == n


def test_network_error_is_treated_like_a_restriction_after_retries(client, monkeypatch):
    calls = _script(monkeypatch, [S.requests.ConnectionError("boom")])
    with pytest.raises(S.SmartRCRestricted):
        client._request("days")
    assert len(calls) == 3


def test_404_returns_empty_without_retry(client, monkeypatch):
    calls = _script(monkeypatch, [_Resp(404)])
    assert client._request("horses", {"hcode": "z"}) == {"success": False, "data": []}
    assert len(calls) == 1 and not client.sleeps


def test_scrape_race_does_not_swallow_restriction_and_stops_hammering(client, monkeypatch):
    runners = [{"hcode": f"h{i}"} for i in range(5)]
    calls = []

    def post(url, **kw):
        calls.append(url)
        return _Resp(200, {"success": True, "data": runners}) if url.endswith("runners/view") else _Resp(429)

    monkeypatch.setattr(S.requests, "post", post)
    with pytest.raises(S.SmartRCRestricted):
        client.scrape_race("2026101012030101")
    assert len(calls) <= 1 + 3                                              # runners 1 回 + 最初の馬の再試行まで。残りの馬には進まない


def test_same_horse_is_reused_across_races_within_ttl(client, monkeypatch):
    calls = []

    def post(url, **kw):
        calls.append(url.rsplit("/", 2)[-2])
        body = {"runners": [{"hcode": "h1"}], "horses": [{"hcode": "h1"}], "fullresults": [{"rdate": "20260101"}]}
        key = url.rsplit("/", 2)[-2]
        return _Resp(200, {"success": True, "data": body.get(key, [])})

    monkeypatch.setattr(S.requests, "post", post)
    client.scrape_race("2026101012030101")
    n1 = len(calls)
    client.scrape_race("2026101012030102")
    assert len(calls) - n1 == 1, "2 回目は runners だけ（馬の horses/fullresults は使い回す）"
