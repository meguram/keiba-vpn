"""開催日の待ち（_sleep_until）: time.sleep が壁時計より数 ms 早く起きても、予定の時刻を過ぎるまで待ち直す。

2026-11-22 の監査: 早起きのまま判定に進み、T-60 が T-45 に持ち越され、T-10・T-05 が捨てられた。
"""
from datetime import datetime, timedelta

import src.scraper.auto_scrape_queue as Q


class _Clock:
    """time.sleep(n) で壁時計が n − 5ms しか進まない（早起きする）偽の時計。"""

    def __init__(self, start: datetime):
        self.now = start
        self.sleeps: list[float] = []

    def sleep(self, sec: float) -> None:
        self.sleeps.append(sec)
        self.now += timedelta(seconds=max(0.0, sec - 0.005)) if sec > 0.01 else timedelta(seconds=sec)


def _patch(monkeypatch, clock: _Clock) -> None:
    class _DT(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock.now

    monkeypatch.setattr(Q, "datetime", _DT)
    monkeypatch.setattr(Q.time, "sleep", clock.sleep)


def test_sleep_until_resleeps_after_early_wakeup(monkeypatch):
    start = datetime(2026, 11, 22, 9, 0, tzinfo=Q._JST)
    at = start + timedelta(minutes=15)
    clock = _Clock(start)
    _patch(monkeypatch, clock)
    Q._sleep_until(at)
    assert clock.now >= at                 # 予定の時刻を過ぎてから戻る
    assert len(clock.sleeps) >= 2          # 早起きしたので、もう一度 sleep した


def test_sleep_until_past_time_does_not_sleep(monkeypatch):
    start = datetime(2026, 11, 22, 9, 0, tzinfo=Q._JST)
    clock = _Clock(start)
    _patch(monkeypatch, clock)
    Q._sleep_until(start - timedelta(seconds=1))
    assert clock.sleeps == []


def test_early_wakeup_does_not_shift_t60_to_t15(monkeypatch):
    """T-60 の時刻まで待ってから判定すると、T-60 の取得に入る（早起きで「まだ来ていない」と見なされて持ち越さない）。"""
    post = datetime(2026, 11, 22, 10, 0, tzinfo=Q._JST)
    race = {"race_id": "202605050101", "post_time": post}
    t60 = post - timedelta(minutes=60)
    t15 = post - timedelta(minutes=15)
    events = [{"race": race, "kind": "t60", "at": t60, "t15_at": t15}, {"race": race, "kind": "t15", "at": t15}]
    clock = _Clock(t60 - timedelta(minutes=5))
    _patch(monkeypatch, clock)
    Q._sleep_until(t60)
    t60_batch, t15_batch, skipped = Q.select_due_events(events, set(), Q.datetime.now(Q._JST))
    assert [x["race"]["race_id"] for x in t60_batch] == ["202605050101"]
    assert not t15_batch and not skipped
