"""重い GET API の応答を短時間だけ使い回し、同じ要求が同時に来たときは 1 回だけ計算する（連打・同時表示の負荷対策）。"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Hashable

_JST = timezone(timedelta(hours=9))


class TtlCoalescingCache:
    def __init__(self, max_items: int = 512, clock: Callable[[], float] = time.monotonic) -> None:
        self._items: dict[Hashable, tuple[float, Any]] = {}
        self._locks: dict[Hashable, asyncio.Lock] = {}
        self._max = max_items
        self._clock = clock

    def _fresh(self, key: Hashable, ttl: float) -> tuple[bool, Any]:
        hit = self._items.get(key)
        if hit and (self._clock() - hit[0]) < ttl:
            return True, hit[1]
        return False, None

    async def get_or_compute(self, key: Hashable, ttl: float, compute: Callable[[], Awaitable[Any]]) -> Any:
        ok, val = self._fresh(key, ttl)
        if ok:
            return val
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            ok, val = self._fresh(key, ttl)      # 待っている間に別の要求が計算を終えていれば、それを使う
            if ok:
                return val
            val = await compute()
            if len(self._items) >= self._max:
                oldest = min(self._items, key=lambda k: self._items[k][0])
                self._items.pop(oldest, None)
                self._locks.pop(oldest, None)
            self._items[key] = (self._clock(), val)
            return val

    def clear(self) -> None:
        self._items.clear()


def race_list_ttl(date: str, now: datetime | None = None) -> float:
    """開催日別のレース一覧の保持秒数。当日前後は結果の有無が変わるので短く、過去の日は長く。"""
    now = now or datetime.now(_JST)
    try:
        d = datetime.strptime(date, "%Y%m%d").replace(tzinfo=_JST)
    except ValueError:
        return 0.0
    return 45.0 if abs((d.date() - now.date()).days) <= 2 else 600.0
