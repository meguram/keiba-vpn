"""開催日でない日・作業が無い日に、無駄な netkeiba アクセスをしないための仕組みのテスト。"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone

import pytest

from src.scraper import auto_scrape as A
from src.scraper.client import NetkeibaClient

JST = timezone(timedelta(hours=9))
FUTURE = ["2026-10-17", "2026-10-18", "2026-10-24", "2026-10-25", "2026-11-14", "2026-12-26"]


# ── レース一覧の先読み範囲 ─────────────────────────────────────────────────────

def test_race_list_polls_only_near_dates_except_on_the_weekly_check_day():
    tue = date(2026, 10, 13)      # 火曜: 14 日先（10/27）までだけ
    assert A.race_list_poll_dates(FUTURE, tue, ahead_days=14) == ["2026-10-17", "2026-10-18", "2026-10-24", "2026-10-25"]
    mon = date(2026, 10, 12)      # 月曜: 先の日付も確認する
    assert A.race_list_poll_dates(FUTURE, mon, ahead_days=14) == FUTURE


def test_race_list_far_dates_are_checked_once_a_week():
    days = [date(2026, 10, 12) + timedelta(days=i) for i in range(7)]       # 月〜日
    far_checks = [d for d in days if "2026-12-26" in A.race_list_poll_dates(FUTURE, d, ahead_days=14)]
    assert far_checks == [date(2026, 10, 12)]


def test_race_list_ahead_days_can_be_set_by_env(monkeypatch):
    monkeypatch.setenv("KEIBA_RACE_LIST_AHEAD_DAYS", "3")
    assert A.race_list_poll_dates(FUTURE, date(2026, 10, 15)) == ["2026-10-17", "2026-10-18"]       # 木曜: 3 日先（10/18）まで


# ── バロメーターの試行上限 ─────────────────────────────────────────────────────

def test_barometer_retry_gives_up_after_the_cap_and_respects_the_interval():
    now = datetime(2026, 10, 20, 8, 30, tzinfo=JST)
    by_date = {"20261010": ["r1", "r2", "r3", "r4"]}
    state = {
        "race_barometer/r2": {"attempts": A.BAROMETER_MAX_ATTEMPTS, "last": (now - timedelta(days=3)).isoformat()},   # 上限到達
        "race_barometer/r3": {"attempts": 2, "last": (now - timedelta(hours=1)).isoformat()},                        # 間隔が短い
    }
    got = A.select_barometer_targets(by_date, lambda rid: rid == "r4", state, now)       # r4 は取得済み
    assert got == ["r1"]


# ── ログインは最初の通信まで遅らせる ───────────────────────────────────────────────

def test_login_is_deferred_until_the_session_is_used(monkeypatch):
    calls = []

    def fake_login(self):
        self._login_pending = False
        calls.append(1)
        self._logged_in = True
        return True

    monkeypatch.setenv("NETKEIBA_LAZY_LOGIN", "1")
    monkeypatch.setattr(NetkeibaClient, "login", fake_login)
    c = NetkeibaClient(auto_login=True)
    assert calls == []                      # 作業が無ければログインしない
    c._session
    c._session
    assert calls == [1]                     # 初めて通信に使うときに 1 回だけ


def test_eager_login_can_be_restored_by_env(monkeypatch):
    calls = []
    monkeypatch.setenv("NETKEIBA_LAZY_LOGIN", "0")
    monkeypatch.setattr(NetkeibaClient, "login", lambda self: calls.append(1) or True)
    NetkeibaClient(auto_login=True)
    assert calls == [1]


def test_auto_login_false_never_logs_in(monkeypatch):
    calls = []
    monkeypatch.setattr(NetkeibaClient, "login", lambda self: calls.append(1) or True)
    c = NetkeibaClient(auto_login=False)
    c._session
    assert calls == []


# ── キューのワーカーは runner を使い回す ──────────────────────────────────────────

def test_queue_worker_reuses_runner_within_ttl(monkeypatch):
    from src.scraper.job_queue import ScrapeJobQueue

    made = []

    def factory():
        made.append(object())
        return made[-1]

    ScrapeJobQueue._RUNNER_LOCAL.__dict__.clear()
    monkeypatch.setenv("SCRAPE_QUEUE_RUNNER_REUSE_SEC", "1800")
    a = ScrapeJobQueue._runner_for_thread(factory)
    b = ScrapeJobQueue._runner_for_thread(factory)
    assert a is b and len(made) == 1
    monkeypatch.setenv("SCRAPE_QUEUE_RUNNER_REUSE_SEC", "0")          # 0 は毎回作る（従来動作）
    c = ScrapeJobQueue._runner_for_thread(factory)
    assert c is not a and len(made) == 2
    ScrapeJobQueue._RUNNER_LOCAL.__dict__.clear()
