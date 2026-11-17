"""GCS の障害の間の Cloud Run の入口（503 + no-store・取り込みを待たない）と、応答の使い回し・ミラーの時間切れ。外部接続なし。
2026-11-15 の 504（docs/operations/api_memory_20261117.md）の再発防止。"""

import asyncio
import time

import pytest


@pytest.fixture
def entry(monkeypatch):
    monkeypatch.setenv("KEIBA_ENV", "dev")
    monkeypatch.setenv("KEIBA_CLOUDRUN_LAZY", "1")
    monkeypatch.delenv("KEIBA_SERVING_STATE_PULL", raising=False)
    from src.api import cloudrun_entry

    return cloudrun_entry


def _run_send(entry, monkeypatch, *, failfast: bool, unhealthy: bool, method="GET", path="/api/race/1", status=200):
    from src.scraper import storage as S

    monkeypatch.setattr(S, "gcs_unhealthy_now", lambda: unhealthy)
    sent = []

    async def send(msg):
        sent.append(msg)

    flag = {"failfast": True} if failfast else {}
    wrapped = entry._degraded_send(send, method, path, flag)
    start = {"type": "http.response.start", "status": status,
             "headers": [(b"content-type", b"application/json"), (b"cache-control", b"public, max-age=2592000")]}
    asyncio.run(wrapped(start))
    asyncio.run(wrapped({"type": "http.response.body", "body": b"{}"}))
    return sent[0], sent[1]


def test_healthy_responses_are_untouched(entry, monkeypatch):
    start, body = _run_send(entry, monkeypatch, failfast=False, unhealthy=False)
    assert start["status"] == 200 and (b"cache-control", b"public, max-age=2592000") in start["headers"]
    assert body["body"] == b"{}"


def test_failfast_page_get_becomes_503_no_store(entry, monkeypatch):
    start, _ = _run_send(entry, monkeypatch, failfast=True, unhealthy=True)
    h = dict(start["headers"])
    assert start["status"] == 503 and h[b"cache-control"] == b"no-store" and h[b"retry-after"] == b"30"
    assert [k for k, _ in start["headers"]].count(b"cache-control") == 1      # 公開の保存の指定は消す
    start, _ = _run_send(entry, monkeypatch, failfast=True, unhealthy=False, status=404)
    assert start["status"] == 503                                             # 「無い」と保存させない


def test_unhealthy_without_failfast_only_disables_caching(entry, monkeypatch):
    start, _ = _run_send(entry, monkeypatch, failfast=False, unhealthy=True)
    assert start["status"] == 200 and dict(start["headers"])[b"cache-control"] == b"no-store"
    for kw in ({"method": "POST"}, {"path": "/ops/memory"}, {"status": 500}):
        start, _ = _run_send(entry, monkeypatch, failfast=True, unhealthy=True, **kw)
        assert start["status"] != 503 or kw.get("status") == 500, kw         # 画面の GET の 200/404 だけを 503 にする


def test_gateway_sets_request_flag_end_to_end(entry, monkeypatch):
    """要求の中（スレッドでも）で障害ですぐ返したら、入口がその要求を 503 にする。"""
    from starlette.testclient import TestClient

    from src.scraper import storage as S

    monkeypatch.setattr(S, "gcs_unhealthy_now", lambda: True)

    async def app(scope, receive, send):
        await asyncio.to_thread(S.note_degraded)
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": b"{}"})

    gw = entry.Gateway(app, app, app)
    r = TestClient(gw).get("/api/race/202611160101")
    assert r.status_code == 503 and r.headers["cache-control"] == "no-store"
    assert S.request_flag() is None


def test_gate_does_not_wait_while_pulls_fail(entry):
    waited = []

    class R:
        failing, wait_s = True, 8.0

        def should_wait(self):
            return True

        def start_if_due(self):
            class T:
                def join(self, s):
                    waited.append(s)
            return T()

    gw = entry.Gateway(None, None, None)
    gw.refresher = R()
    asyncio.run(gw._gate())
    assert waited == []
    gw.refresher.failing = False
    asyncio.run(gw._gate())
    assert waited == [8.0]


def test_cloudrun_defaults_enable_failfast_and_short_timeouts(entry):
    d = entry._DEFAULTS
    assert d["KEIBA_GCS_SHARED_STATE"] == "1" and d["KEIBA_GCS_FAIL_FAST"] == "1"
    assert float(d["KEIBA_GCS_TIMEOUT_SEC"]) <= 5 and float(d["KEIBA_GCS_RETRY_DEADLINE_SEC"]) <= 5


def test_prerender_skips_responses_built_during_degradation(monkeypatch, tmp_path):
    from src.api import prerender
    from src.scraper import storage as S

    monkeypatch.setenv("KEIBA_PRERENDER", "1")
    monkeypatch.setenv("KEIBA_PRERENDER_DIR", str(tmp_path))
    since = time.time_ns()
    prerender.put("race", "202611160101", b"{}", since_ns=since, hard=0)
    assert prerender._path("race", "202611160101").exists()
    since = time.time_ns()
    S.note_degraded()
    prerender.put("race", "202611160102", b"{}", since_ns=since, hard=0)
    assert not prerender._path("race", "202611160102").exists()


def test_request_refresher_marks_failing_and_clears_on_success(tmp_path):
    from src.ops import gcs_mirror as gm

    class Clock:
        t = 1000.0

        def __call__(self):
            return self.t

    class Boom:
        def list(self, prefix):
            raise RuntimeError("down")

    clk = Clock()
    r = gm.RequestRefresher(Boom(), tmp_path, interval=60, clock=clk)
    clk.t += 61
    r.start_if_due().join(5)
    assert r.failing is True

    class Ok:
        def list(self, prefix):
            return {}

        def download(self, key):
            raise FileNotFoundError(key)

    r.backend = Ok()
    clk.t += 61
    r.start_if_due().join(5)
    assert r.failing is False


def test_gcs_backend_passes_timeout_and_retry(monkeypatch):
    from src.ops import gcs_mirror as gm

    seen = {}

    class Blob:
        def download_as_bytes(self, **kw):
            seen["dl"] = kw
            return b"x"

    class Bucket:
        def blob(self, key):
            return Blob()

    class Client:
        def bucket(self, name):
            return Bucket()

        def list_blobs(self, bucket, **kw):
            seen["ls"] = kw
            return []

    monkeypatch.setenv("GCS_BUCKET", "b")
    monkeypatch.setenv("KEIBA_GCS_MIRROR_TIMEOUT_SEC", "7")
    monkeypatch.setattr("src.config.gcp_guard.assert_gcp_allowed", lambda *_: None)
    monkeypatch.setattr("src.config.gcp_credentials.storage_client", lambda: Client())
    be = gm.GcsBackend()
    assert be.download("k") == b"x" and be.list("p/") == {}
    assert seen["dl"]["timeout"] == 7.0 and seen["ls"]["timeout"] == 7.0
    assert seen["dl"]["retry"] is not None and seen["ls"]["retry"] is not None
