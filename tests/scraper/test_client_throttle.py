"""待ちを「前の GET の開始から」で数えるスイッチ（NETKEIBA_THROTTLE_START_TO_START）のテスト。"""

from __future__ import annotations

import time
from unittest import mock

from src.scraper import client as cl
from src.scraper.client import NetkeibaClient

_CFG = {"netkeiba": {"min_interval": 1.0, "max_interval": 1.0}, "smartrc": {"min_interval": 2.0, "max_interval": 4.0}}


def _slept(monkeypatch, start_to_start: bool, since_last: float | None) -> float:
    monkeypatch.setattr(cl, "_THROTTLE_CONFIG", _CFG)
    monkeypatch.setattr(cl, "_THROTTLE_START_TO_START", start_to_start)
    monkeypatch.setattr(cl.random, "gauss", lambda mu, sigma: 0.0)
    c = NetkeibaClient(auto_login=False)
    c._last_request_time = 0 if since_last is None else time.time() - since_last
    with mock.patch.object(cl.time, "sleep") as sl:
        c._throttle("netkeiba")
    return sl.call_args.args[0]


def test_default_waits_full_interval_after_response(monkeypatch):
    assert _slept(monkeypatch, False, since_last=0.6) == 1.0


def test_start_to_start_subtracts_time_since_last_get(monkeypatch):
    assert 0.35 <= _slept(monkeypatch, True, since_last=0.6) <= 0.45


def test_start_to_start_does_not_sleep_when_already_slower_than_interval(monkeypatch):
    assert _slept(monkeypatch, True, since_last=3.0) == 0.0


def test_start_to_start_waits_full_interval_on_first_request(monkeypatch):
    assert _slept(monkeypatch, True, since_last=None) == 1.0
