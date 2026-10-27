"""重い・副作用のある GET の防御: 血統傾向分析の無条件全件集計の拒否／キャッシュ／同時実行制限、upcoming-races が開発者以外で外部取得しない。"""

import json
import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("KEIBA_CLOUDRUN_LAZY", "1")      # cloudrun_entry を import するだけにする（アプリを組み立てない）

import src.api.app as A

client = TestClient(A.app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _clean():
    A._PED_STATS_CACHE.clear()
    yield
    A._PED_STATS_CACHE.clear()


def test_unfiltered_pedigree_query_is_rejected_without_computing(monkeypatch):
    called = []
    monkeypatch.setattr(A, "_pedigree_race_stats_query_impl", lambda **k: called.append(k))
    r = client.get("/api/pedigree-race-stats/query")
    assert r.status_code == 400 and "絞り込み条件" in r.json()["error"] and not called


def test_filtered_query_runs_in_thread_and_second_call_is_cached(monkeypatch):
    calls = []

    def fake(**kw):
        calls.append(kw)
        return A.JSONResponse({"total_entries": 3, "cat1": [], "cat2": [], "cat3": []})

    monkeypatch.setattr(A, "_pedigree_race_stats_query_impl", fake)
    r1 = client.get("/api/pedigree-race-stats/query?date_from=20260101&surface=芝")
    r2 = client.get("/api/pedigree-race-stats/query?date_from=20260101&surface=芝")
    assert r1.status_code == r2.status_code == 200 and r1.json() == r2.json() and len(calls) == 1
    assert calls[0]["date_from"] == "20260101" and calls[0]["top_n"] == 50


def test_busy_returns_503_with_retry_after(monkeypatch):
    monkeypatch.setattr(A, "_pedigree_race_stats_query_impl", lambda **k: A.JSONResponse({}))
    taken = [A._PED_STATS_SLOTS.acquire(blocking=False) for _ in range(2)]
    try:
        r = client.get("/api/pedigree-race-stats/query?date_from=20260101")
        assert r.status_code == 503 and r.headers.get("retry-after") == "10"
    finally:
        for t in taken:
            if t:
                A._PED_STATS_SLOTS.release()


def test_upcoming_races_never_fetches_remote_for_non_developers(monkeypatch):
    seen = {}

    class Boom:
        def __init__(self, *a, **k):
            seen["client"] = True
            raise AssertionError("netkeiba に取得に行ってはいけない")

    monkeypatch.setattr("src.scraper.client.NetkeibaClient", Boom)
    monkeypatch.setattr(A, "is_developer", lambda request: False)
    A._UPCOMING_RACES_DEFAULT_CACHE["payload"] = None
    r = client.get("/api/upcoming-races?start_date=2031-01-01&end_date=2031-01-03")
    assert r.status_code == 200 and r.json()["local_only"] is True and "client" not in seen


def test_flask_db_errors_become_503_but_other_errors_and_http_errors_keep_their_codes():
    from sqlalchemy.exc import ProgrammingError

    from src.api.flask_app import _fnum, create_app

    fa = create_app()

    @fa.get("/_t/db")
    def _db():
        raise ProgrammingError("select 1", {}, Exception("column users.is_member does not exist"))

    @fa.get("/_t/boom")
    def _boom():
        raise RuntimeError("boom")

    c = fa.test_client()
    r = c.get("/_t/db")
    assert r.status_code == 503 and r.get_json()["reason"] == "db_error" and r.headers["Retry-After"] == "30"
    assert c.get("/_t/missing").status_code == 404
    assert c.get("/_t/boom").status_code == 500
    assert _fnum(None) is None and _fnum("1.5") == 1.5


def test_backfill_module_can_be_imported_from_a_worker_thread():
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent("""
        import threading, sys
        err = []
        def go():
            try:
                import src.scraper.backfill
            except Exception as e:
                err.append(repr(e))
        t = threading.Thread(target=go); t.start(); t.join()
        sys.exit(1 if err else 0)
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-500:]


def test_warm_sequence_calls_latest_past_race_and_survives_failures():
    import asyncio

    from src.api.cloudrun_entry import warm_sequence

    seen = []

    async def get(path):
        seen.append(path)
        if path == "/api/scrape-dates":
            return 200, {"dates": ["20260101", "20261017", "20261031"]}
        if path == "/api/race-list/20261017":
            return 200, {"races": [{"race_id": "202605040101"}, {"race_id": "202605040102"}]}
        if path.endswith("/predictions"):
            raise RuntimeError("boom")                 # 失敗しても次へ進む
        return 200, {}

    called = asyncio.run(warm_sequence(get, today="20261020"))
    assert called == ["/api/v1/health", "/api/scrape-dates", "/api/race-list/20261017", "/api/race/202605040101/predictions", "/api/race/202605040101"]
    assert asyncio.run(warm_sequence(lambda p: asyncio.sleep(0, (500, None)), today="20261020")) == ["/api/v1/health", "/api/scrape-dates"]


def _lifespan(gateway):
    import asyncio

    async def go():
        sent = []
        msgs = iter([{"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}])

        async def receive():
            return next(msgs)

        async def send(m):
            sent.append(m["type"])

        await gateway({"type": "lifespan"}, receive, send)
        return sent

    return asyncio.run(go())


def test_gateway_lifespan_waits_for_warmup_but_never_blocks_startup_on_failure(monkeypatch):
    import asyncio

    from src.api import cloudrun_entry as C

    async def noop(*a):
        return None

    g = C.Gateway(noop, noop, noop)
    order = []

    async def warm():
        order.append("warm")

    g.warmup = warm
    assert _lifespan(g) == ["lifespan.startup.complete", "lifespan.shutdown.complete"] and order == ["warm"]

    async def bad():
        raise RuntimeError("x")

    g.warmup = bad
    assert _lifespan(g)[0] == "lifespan.startup.complete"

    monkeypatch.setattr(C, "WARMUP_TIMEOUT_SEC", 0.05)

    async def slow():
        await asyncio.sleep(5)

    g.warmup = slow
    assert _lifespan(g)[0] == "lifespan.startup.complete"
