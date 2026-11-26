"""応答を作る段ごとの秒数を測り、Server-Timing の文字列とログに出す（計測用。中身は段の名前と時間だけ）。

使い方: ``tm = StageTimer()`` を作り、``with tm.stage("name"):`` で段を囲む。``tm.header()`` が Server-Timing の値。
段は複数のスレッド（asyncio.to_thread）から足されることがあるので、足すときだけ鍵を取る。
"""
from __future__ import annotations

import re
import threading
import time
from contextlib import contextmanager
from typing import Iterator

_TOKEN = re.compile(r"[^A-Za-z0-9_.-]")


def _token(name: str) -> str:
    """Server-Timing の名前は token（英数字と一部の記号）だけ。外れる文字は _ に替える。"""
    return _TOKEN.sub("_", name) or "_"


class StageTimer:
    def __init__(self) -> None:
        self._items: list[tuple[str, float]] = []
        self._lock = threading.Lock()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, (time.perf_counter() - t0) * 1000)

    def add(self, name: str, ms: float) -> None:
        with self._lock:
            self._items.append((_token(name), float(ms)))

    def items(self) -> list[tuple[str, float]]:
        with self._lock:
            return list(self._items)

    def header(self) -> str:
        return ", ".join(f"{n};dur={ms:.1f}" for n, ms in self.items())


def merge_server_timing(*values: str | None) -> str:
    """Server-Timing の値をつなぐ（空は飛ばす）。"""
    return ", ".join(v for v in values if v)
