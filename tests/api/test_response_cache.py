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


def test_stale_value_is_returned_immediately_and_refreshed_in_background():
    async def run():
        t = [0.0]
        c = TtlCoalescingCache(clock=lambda: t[0])
        calls = []

        async def compute():
            calls.append(1)
            await asyncio.sleep(0.05)
            return len(calls)

        first = await c.get_or_compute("k", 10, compute, stale_max=100)       # 1 回目は待って計算
        t[0] = 20.0                                                             # 期限切れ（stale_max 内）
        started = asyncio.get_running_loop().time()
        stale = await c.get_or_compute("k", 10, compute, stale_max=100)       # 古い値を待たずに返す
        waited = asyncio.get_running_loop().time() - started
        again = await c.get_or_compute("k", 10, compute, stale_max=100)       # 裏の再計算は 1 回だけ
        await asyncio.sleep(0.15)
        fresh = await c.get_or_compute("k", 10, compute, stale_max=100)       # 裏で差し替わっている
        return first, stale, again, fresh, waited, len(calls)

    first, stale, again, fresh, waited, n = asyncio.run(run())
    assert first == 1 and stale == 1 and again == 1 and fresh == 2
    assert waited < 0.04 and n == 2


def test_too_old_value_is_not_served_and_failed_refresh_keeps_the_stale_value():
    async def run():
        t = [0.0]
        c = TtlCoalescingCache(clock=lambda: t[0])
        state = {"fail": False, "n": 0}

        async def compute():
            state["n"] += 1
            if state["fail"]:
                raise RuntimeError("boom")
            return state["n"]

        await c.get_or_compute("k", 10, compute, stale_max=50)
        t[0] = 60.0                                  # stale_max を超えた → 待って計算し直す
        too_old = await c.get_or_compute("k", 10, compute, stale_max=50)
        t[0] = 75.0
        state["fail"] = True                         # 裏の再計算が失敗しても、古い値を返し続ける
        s1 = await c.get_or_compute("k", 10, compute, stale_max=50)
        await asyncio.sleep(0.02)
        s2 = await c.get_or_compute("k", 10, compute, stale_max=50)
        return too_old, s1, s2

    too_old, s1, s2 = asyncio.run(run())
    assert too_old == 2 and s1 == 2 and s2 == 2


def test_without_stale_max_behaviour_is_unchanged():
    async def run():
        t = [0.0]
        c = TtlCoalescingCache(clock=lambda: t[0])
        n = []

        async def compute():
            n.append(1)
            return len(n)

        a = await c.get_or_compute("k", 10, compute)
        t[0] = 11.0
        b = await c.get_or_compute("k", 10, compute)
        return a, b

    assert asyncio.run(run()) == (1, 2)


def test_warm_computes_once_and_skips_fresh_values():
    async def run():
        c = TtlCoalescingCache()
        n = []

        async def compute():
            n.append(1)
            return 1

        await c.warm("k", 30, compute)
        await c.warm("k", 30, compute)
        return len(n)

    assert asyncio.run(run()) == 1
