"""依存先（Redis など）の障害時に、待たずに素早く諦めるためのサーキットブレーカー。

閉(closed) → 連続失敗が閾値に達すると 開(open) → 一定時間後に 半開(half_open) で 1 件だけ試行し、
成功すれば閉、失敗すればまた開。開いている間は依存先に触れないため、障害時のレイテンシは
「タイムアウト × 件数」ではなく「ほぼ 0」になる。
"""
from __future__ import annotations

import threading
import time
from typing import Callable, TypeVar

T = TypeVar("T")
_UNSET = object()


class CircuitOpenError(RuntimeError):
    """ブレーカーが開いていて呼び出しを拒否した。"""


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 3,
        recovery_timeout: float = 10.0,
        trip_on: tuple[type[BaseException], ...] = (Exception,),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.trip_on = trip_on
        self._clock = clock
        self._lock = threading.Lock()
        self._failures = 0
        self._opened_at: float | None = None
        self._probe_in_flight = False

    @property
    def state(self) -> str:
        with self._lock:
            return self._state_locked()

    def _state_locked(self) -> str:
        if self._opened_at is None:
            return "closed"
        if self._clock() - self._opened_at >= self.recovery_timeout:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        with self._lock:
            state = self._state_locked()
            if state == "closed":
                return True
            if state == "half_open" and not self._probe_in_flight:
                self._probe_in_flight = True
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None
            self._probe_in_flight = False

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._opened_at is not None or self._failures >= self.failure_threshold:
                self._opened_at = self._clock()
            self._probe_in_flight = False

    def call(self, fn: Callable[[], T], *, fallback: Callable[[], T] | object = _UNSET) -> T:
        """``fn`` を実行する。開いている間は ``fallback()``（無ければ CircuitOpenError）。

        ``trip_on`` に当たる例外だけを失敗として数える（想定外のバグでは開かない）。例外は再送出する。
        """
        if not self.allow():
            if fallback is _UNSET:
                raise CircuitOpenError(f"circuit '{self.name}' is open")
            return fallback()  # type: ignore[operator]
        try:
            result = fn()
        except self.trip_on:
            self.record_failure()
            raise
        except BaseException:
            # 失敗として数えないが、半開の試行枠は返す
            with self._lock:
                self._probe_in_flight = False
            raise
        self.record_success()
        return result
