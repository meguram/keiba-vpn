import asyncio
from datetime import datetime, timedelta, timezone

from src.api.response_cache import TtlCoalescingCache, race_list_ttl

JST = timezone(timedelta(hours=9))


def test_concurrent_requests_compute_once():
    async def run():
        c = TtlCoalescingCache()
        calls = []

        async def compute():
            calls.append(1)
            await asyncio.sleep(0.05)
            return {"n": len(calls)}

        res = await asyncio.gather(*[c.get_or_compute(("d", True), 30, compute) for _ in range(6)])
        return calls, res

    calls, res = asyncio.run(run())
    assert len(calls) == 1 and all(r == {"n": 1} for r in res)


def test_ttl_expiry_recomputes_and_keys_are_separate():
    async def run():
        t = [0.0]
        c = TtlCoalescingCache(clock=lambda: t[0])
        calls = []

        async def compute():
            calls.append(1)
            return len(calls)

        a = await c.get_or_compute("k", 10, compute)
        b = await c.get_or_compute("k", 10, compute)
        t[0] = 11.0
        d = await c.get_or_compute("k", 10, compute)
        e = await c.get_or_compute("other", 10, compute)
        return a, b, d, e

    assert asyncio.run(run()) == (1, 1, 2, 3)


def test_errors_are_not_cached():
    async def run():
        c = TtlCoalescingCache()
        n = {"i": 0}

        async def compute():
            n["i"] += 1
            if n["i"] == 1:
                raise RuntimeError("boom")
            return "ok"

        try:
            await c.get_or_compute("k", 10, compute)
        except RuntimeError:
            pass
        return await c.get_or_compute("k", 10, compute)

    assert asyncio.run(run()) == "ok"


def test_race_list_ttl_is_short_near_today_and_long_for_past():
    now = datetime(2026, 10, 19, 12, 0, tzinfo=JST)
    assert race_list_ttl("20261018", now) == 45.0 and race_list_ttl("20261021", now) == 45.0
    assert race_list_ttl("20260906", now) == 600.0
    assert race_list_ttl("bad", now) == 0.0
