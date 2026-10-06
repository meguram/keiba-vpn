"""アクセス制限の即時検知（AccessGuard）のテスト。方針: 「ページが存在しない」以外のエラーはアクセス制限とみなす。"""

from __future__ import annotations

import threading

import pytest
import requests

from src.scraper import client as client_mod
from src.scraper.access_guard import AccessGuard, AccessRestrictionDetected

NOT_FOUND_BODY = "<html><body>お探しのページが見つかりません</body></html>"


class R:
    def __init__(self, code, url="https://db.netkeiba.com/race/1/", body="<html>" + "x" * 300):
        self.status_code, self.url, self.text = code, url, body
        self.content = body.encode("utf-8")
        self.headers = {"Content-Type": "text/html"}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)


U = "https://db.netkeiba.com/a"


@pytest.mark.parametrize("code", [400, 401, 403, 404, 408, 429, 500, 502, 503])
def test_any_http_error_is_treated_as_access_restriction(code):
    g = AccessGuard()
    g(U, R(200))                                                             # 成功は素通り
    with pytest.raises(AccessRestrictionDetected) as ei:
        g(U, R(code))
    assert ei.value.status == code and ei.value.kind == "http_status" and g.tripped_at
    assert not isinstance(ei.value, Exception)                               # BaseException: 既存の `except Exception` を突き抜ける


def test_page_not_found_is_the_only_exception():
    g = AccessGuard()
    g(U, R(404, body=NOT_FOUND_BODY))                                        # 本文が「存在しない」→ 制限ではない
    g(U, R(400, body="<html>該当データはありません</html>"))
    g(U, R(404, body="<html>No Data</html>"))
    assert g.tripped is None and g.not_found == 3
    with pytest.raises(AccessRestrictionDetected):
        g(U, R(404, body="<html><body>Not Found</body></html>"))             # 本文が存在しないを示さない 404 は制限の疑い
    strict = AccessGuard(exempt_not_found=False)
    with pytest.raises(AccessRestrictionDetected):
        strict(U, R(404, body=NOT_FOUND_BODY))                               # --strict-not-found


def test_once_tripped_everything_fails_including_other_threads():
    g = AccessGuard()
    with pytest.raises(AccessRestrictionDetected):
        g(U, R(404, body="x"))
    for call in (lambda: g("https://race.netkeiba.com/y", R(200)), lambda: g.on_error(U, RuntimeError("x"))):
        with pytest.raises(AccessRestrictionDetected):
            call()
    errs = []

    def other():
        try:
            g(U, R(200))
        except AccessRestrictionDetected as e:
            errs.append(e)

    t = threading.Thread(target=other)
    t.start()
    t.join()
    assert len(errs) == 1


def test_transport_errors_count_as_restriction():
    g = AccessGuard()
    with pytest.raises(AccessRestrictionDetected) as ei:
        g.on_error(U, requests.exceptions.ConnectTimeout("timed out"))
    assert ei.value.kind == "transport_error" and "ConnectTimeout" in ei.value.detail
    quiet = AccessGuard(watch_transport_errors=False)
    quiet.on_error(U, requests.exceptions.ConnectionError("x"))
    assert quiet.tripped is None


def test_block_page_with_2xx_is_detected():
    g = AccessGuard()
    g(U, R(200, body="<html>通常のページ</html>"))
    with pytest.raises(AccessRestrictionDetected) as ei:
        g(U, R(200, body="<html>ただいまアクセスが集中しています</html>"))
    assert ei.value.kind == "block_page"


def test_ignores_other_hosts_and_resets_on_success_and_threshold():
    g = AccessGuard(threshold=2)
    g("https://example.com/", R(500, body="x"))                              # netkeiba 以外は対象外
    g(U, R(500, body="x"))                                                   # 1 回目(しきい値 2)
    g(U, R(200))                                                             # 正常な応答で連続カウントがリセット
    g(U, R(500, body="x"))
    g(U, R(404, body=NOT_FOUND_BODY))                                        # 存在しないページは中立（カウントもリセットもしない）
    assert g.tripped is None
    with pytest.raises(AccessRestrictionDetected) as ei:
        g(U, R(503, body="x"))                                               # 連続 2 回
    assert ei.value.count == 2


def test_explicit_status_set_limits_what_counts():
    g = AccessGuard({403})
    g(U, R(500, body="x"))
    g(U, R(404, body="x"))
    assert g.tripped is None
    with pytest.raises(AccessRestrictionDetected):
        g(U, R(403, body="x"))


def _client(monkeypatch, responses):
    c = client_mod.NetkeibaClient(auto_login=False)
    monkeypatch.setattr(c, "_throttle", lambda *a, **k: None)
    it = iter(responses)

    class S:
        headers = {}

        def get(self, url, timeout=None):
            r = next(it)
            if isinstance(r, Exception):
                raise r
            return r

    c._session = S()
    return c


def test_client_notifies_observers_and_registration_is_scoped(monkeypatch):
    c = _client(monkeypatch, [R(200), R(404, body="x")])
    assert client_mod._RESPONSE_OBSERVERS == []
    seen = []
    client_mod.add_response_observer(lambda url, resp: seen.append(resp.status_code))
    try:
        c._get_with_backoff(U)
        with pytest.raises(Exception):
            c._get_with_backoff(U)                                           # 観測だけ。従来どおり HTTPError
    finally:
        client_mod._RESPONSE_OBSERVERS.clear()
    assert seen == [200, 404]


def test_guard_context_stops_the_client_on_errors(monkeypatch):
    c = _client(monkeypatch, [R(200), R(404, body="x"), R(200), requests.exceptions.ReadTimeout("t")])
    with AccessGuard() as g:
        assert g in client_mod._RESPONSE_OBSERVERS
        c._get_with_backoff(U)
        with pytest.raises(AccessRestrictionDetected):
            c._get_with_backoff(U)                                           # 制限の疑い → 即中断
        with pytest.raises(AccessRestrictionDetected):
            c._get_with_backoff(U)                                           # 以降は（200 を返されるはずの要求でも）失敗し続ける
    assert client_mod._RESPONSE_OBSERVERS == []
    c2 = _client(monkeypatch, [R(404, body="x")])                            # 登録が無い通常時は従来どおり
    with pytest.raises(Exception) as ei:
        c2._get_with_backoff(U)
    assert not isinstance(ei.value, AccessRestrictionDetected)


def test_guard_sees_transport_errors_through_the_client(monkeypatch):
    c = _client(monkeypatch, [requests.exceptions.ConnectionError("refused")])
    with AccessGuard():
        with pytest.raises(AccessRestrictionDetected) as ei:
            c._get_with_backoff(U)
    assert ei.value.kind == "transport_error"
    assert client_mod._RESPONSE_OBSERVERS == []
