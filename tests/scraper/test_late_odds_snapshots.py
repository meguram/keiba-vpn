"""発走直前（T-10 / T-5 / T-2）の単複オッズ取得（合成データのみ。外部アクセスなし）。"""

from __future__ import annotations

import os
from datetime import datetime
from unittest import mock
from zoneinfo import ZoneInfo

from src.scraper import queue_tasks
from src.scraper.auto_scrape_queue import (
    LATE_KIND_TASK,
    LATE_LEADS,
    LATE_TOLERANCE_SECONDS,
    ODDS_LATE_TASKS,
    build_odds_events,
    late_snapshots_enabled,
    select_due_events,
    select_due_late_events,
)
from src.scraper.odds_snapshot import ODDS_PAYLOAD_KEYS, merge_snapshot

JST = ZoneInfo("Asia/Tokyo")


def _race(rid, hh, mm):
    return {"race_id": rid, "post_time": datetime(2026, 5, 2, hh, mm, tzinfo=JST)}


def _t(hh, mm, ss=0):
    return datetime(2026, 5, 2, hh, mm, ss, tzinfo=JST)


def test_env_switch_default_off():
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("KEIBA_ODDS_LATE_SNAPSHOTS", None)
        assert late_snapshots_enabled() is False
    with mock.patch.dict(os.environ, {"KEIBA_ODDS_LATE_SNAPSHOTS": "1"}):
        assert late_snapshots_enabled() is True
    with mock.patch.dict(os.environ, {"KEIBA_ODDS_LATE_SNAPSHOTS": "0"}):
        assert late_snapshots_enabled() is False


def test_no_late_events_by_default():
    ev = build_odds_events([_race("A", 10, 5)])
    assert {e["kind"] for e in ev} == {"t60", "t15"}


def test_late_events_are_built_at_t10_t5_t2_in_time_order():
    ev = build_odds_events([_race("A", 10, 5)], late_leads=LATE_LEADS)
    got = [(e["kind"], e["at"].strftime("%H:%M")) for e in ev]
    assert got == [("t60", "09:05"), ("t15", "09:50"), ("t10", "09:55"), ("t05", "10:00"), ("t02", "10:03")]


def test_same_time_orders_t60_then_late_then_t15():
    # A の発走 10:15 → t15 = 10:00。B の発走 10:05 → t05 = 10:00。C の発走 11:00 → t60 = 10:00
    ev = build_odds_events([_race("A", 10, 15), _race("B", 10, 5), _race("C", 11, 0)], late_leads=LATE_LEADS)
    at_10 = [(e["race"]["race_id"], e["kind"]) for e in ev if e["at"].strftime("%H:%M") == "10:00"]
    assert at_10 == [("C", "t60"), ("B", "t05"), ("A", "t15")]


def test_select_due_events_ignores_late_kinds():
    ev = build_odds_events([_race("A", 10, 5)], late_leads=LATE_LEADS)
    t60, t15, skipped = select_due_events(ev, set(), _t(10, 1))
    # 10:01 は T-15（09:50）を過ぎている: t15 は実行、t60 はスキップ。t10 / t05 は select_due_events の対象外
    assert [e["kind"] for e in t15] == ["t15"] and t60 == [] and [e["kind"] for e in skipped] == ["t60"]


def test_select_due_late_events_runs_when_due_and_is_idempotent():
    ev = build_odds_events([_race("A", 10, 5)], late_leads=LATE_LEADS)
    done: set = set()
    due, stale = select_due_late_events(ev, done, _t(9, 55, 3))
    assert [e["kind"] for e in due] == ["t10"] and stale == []
    for e in due:
        done.add((e["race"]["race_id"], e["kind"]))
    assert select_due_late_events(ev, done, _t(9, 55, 4)) == ([], [])


def test_late_event_beyond_tolerance_is_stale():
    ev = build_odds_events([_race("A", 10, 5)], late_leads=LATE_LEADS)
    # t05（10:00）を 3 分遅れで見た: 許容（120 秒）を超えるので捨てる。t02（10:03）は 10:03 ちょうどで有効
    due, stale = select_due_late_events(ev, set(), _t(10, 3, 0))
    assert [e["kind"] for e in stale] == ["t10", "t05"]
    assert [e["kind"] for e in due] == ["t02"]
    assert LATE_TOLERANCE_SECONDS == 120


def test_late_event_after_post_time_is_stale():
    ev = build_odds_events([_race("A", 10, 5)], late_leads=LATE_LEADS)
    due, stale = select_due_late_events(ev, set(), _t(10, 5, 10))
    assert due == [] and {e["kind"] for e in stale} == {"t10", "t05", "t02"}


def test_changed_post_time_rebuilds_events_from_new_schedule():
    old = build_odds_events([_race("A", 10, 5)], late_leads=LATE_LEADS)
    new = build_odds_events([_race("A", 10, 15)], late_leads=LATE_LEADS)
    assert [e["at"].strftime("%H:%M") for e in old if e["kind"] == "t05"] == ["10:00"]
    assert [e["at"].strftime("%H:%M") for e in new if e["kind"] == "t05"] == ["10:10"]


def test_late_tasks_registered_everywhere():
    assert ODDS_LATE_TASKS == ["race_odds_t10", "race_odds_t05", "race_odds_t02"]
    assert set(LATE_KIND_TASK) == {"t10", "t05", "t02"}
    ids = {t["id"] for t in queue_tasks.TASK_CATALOG}
    for tid in ODDS_LATE_TASKS:
        assert tid in ids
        assert tid in queue_tasks._ODDS_PHASE_TASKS
    assert [queue_tasks._ODDS_PHASE_TASKS[t] for t in ODDS_LATE_TASKS] == ["t10", "t05", "t02"]


def test_late_snapshots_are_appended_to_history_without_losing_older_phases():
    doc = {"entries": [{"horse_number": 1, "odds": 3.0}]}
    cur = None
    for phase, odds in [("t60", 3.0), ("t15", 2.9), ("t10", 2.8), ("t05", 2.7), ("t02", 2.5)]:
        cur = merge_snapshot(cur, {"entries": [{"horse_number": 1, "odds": odds}]}, phase, ODDS_PAYLOAD_KEYS,
                             snapshot_at=f"2026-05-02T09:{ {'t60': 5, 't15': 50, 't10': 55, 't05': 58, 't02': 59}[phase]:02d}:00+09:00")
    assert cur["phase"] == "t02" and cur["entries"][0]["odds"] == 2.5
    assert [s["phase"] for s in cur["snapshots"]] == ["t60", "t15", "t10", "t05"]
    assert doc  # 引数は破壊されない


# ── 開催日ランナーのループ（偽の時計・偽の取得で通しで動かす）──────────────

class _Clock:
    def __init__(self, start):
        self.t = start


def _run_runner(monkeypatch, schedule, start, late_env, bundle_sec=0.0):
    """task_raceday_runner を偽の時計で最後まで回し、(時刻, race_id, タスク) の記録を返す。"""
    from datetime import timedelta

    from src.scraper import auto_scrape as _as
    from src.scraper import auto_scrape_queue as aq
    from src.scraper import raceday_pre_race_pipeline as pipe

    clock = _Clock(start)

    class _FakeDT:
        @staticmethod
        def now(tz=None):
            return clock.t

    calls: list[tuple[str, str, list[str]]] = []

    def fake_sleep(sec):
        clock.t = clock.t + timedelta(seconds=float(sec))

    def fake_direct(runner, race_id, date_fmt, tasks):
        calls.append((clock.t.strftime("%H:%M:%S"), race_id, list(tasks)))
        if "race_shutuba" in tasks:                    # T-15 のバンドルは重い
            clock.t = clock.t + timedelta(seconds=bundle_sec)
        return []

    monkeypatch.setenv("KEIBA_RACEDAY_DIRECT", "1")
    monkeypatch.setenv("KEIBA_AUTO_SCRAPE_USE_QUEUE", "1")
    if late_env:
        monkeypatch.setenv("KEIBA_ODDS_LATE_SNAPSHOTS", "1")
    else:
        monkeypatch.delenv("KEIBA_ODDS_LATE_SNAPSHOTS", raising=False)
    monkeypatch.setattr(aq, "datetime", _FakeDT)
    monkeypatch.setattr(aq.time, "sleep", fake_sleep)
    monkeypatch.setattr(aq.random, "uniform", lambda a, b: 0.0)
    monkeypatch.setattr(aq, "_run_race_tasks_direct", fake_direct)
    monkeypatch.setattr(aq, "_storage", lambda: object())
    monkeypatch.setattr(aq, "_queue", lambda: object())
    monkeypatch.setattr(aq, "_ensure_race_list_date", lambda *a, **k: True)
    monkeypatch.setattr(aq, "_fetch_race_schedule_storage", lambda *a, **k: schedule)
    monkeypatch.setattr(_as, "_load_race_calendar", lambda: {})
    monkeypatch.setattr(_as, "_today_jst", lambda: start.date())
    monkeypatch.setattr(_as, "_is_race_day", lambda *a, **k: True)
    monkeypatch.setattr(_as, "_get_race_day_venues", lambda *a, **k: ["東京"])
    monkeypatch.setattr(_as, "_save_status", lambda *a, **k: None)
    monkeypatch.setattr(pipe, "refresh_jra_baba_live", lambda: True)
    monkeypatch.setattr(pipe, "trigger_pre_race_predict", lambda *a, **k: None)
    monkeypatch.setattr("src.scraper.run.ScraperRunner", lambda *a, **k: object())
    aq.task_raceday_runner()
    return calls


def test_runner_takes_late_odds_on_time_when_enabled(monkeypatch):
    sched = [_race("A", 10, 5), _race("B", 10, 15)]
    calls = _run_runner(monkeypatch, sched, _t(8, 30), late_env=True, bundle_sec=0)
    assert {(c[1], c[2][0]) for c in calls if c[2][0] in ODDS_LATE_TASKS} == {
        (r, t) for r in ("A", "B") for t in ODDS_LATE_TASKS
    }
    # 取得は予定時刻ちょうど（許容 120 秒以内）
    at = {("A", "race_odds_t10"): "09:55:00", ("A", "race_odds_t05"): "10:00:00", ("A", "race_odds_t02"): "10:03:00",
          ("B", "race_odds_t10"): "10:05:00", ("B", "race_odds_t05"): "10:10:00", ("B", "race_odds_t02"): "10:13:00"}
    got = {(c[1], c[2][0]): c[0] for c in calls if c[2][0] in ODDS_LATE_TASKS}
    for k, v in at.items():
        assert got[k] == v, (k, got[k], v)


def test_runner_late_comes_before_t15_bundle_at_same_time(monkeypatch):
    # A の発走 10:15 → T-15 = 10:00。B の発走 10:05 → T-5 = 10:00。同時刻は直前オッズが先
    sched = [_race("A", 10, 15), _race("B", 10, 5)]
    calls = _run_runner(monkeypatch, sched, _t(8, 30), late_env=True)
    at_10 = [(c[1], c[2][0]) for c in calls if c[0] == "10:00:00"]
    assert at_10.index(("B", "race_odds_t05")) < at_10.index(("A", "race_shutuba"))


def test_runner_does_not_take_late_odds_when_disabled(monkeypatch):
    sched = [_race("A", 10, 5)]
    calls = _run_runner(monkeypatch, sched, _t(8, 30), late_env=False)
    assert not [c for c in calls if c[2][0] in ODDS_LATE_TASKS]
    assert [c[2][0] for c in calls] == ["race_odds_t60", "race_shutuba"]       # 従来どおり（T-60 → T-15）


def test_runner_waits_for_imminent_late_odds_before_a_heavy_bundle(monkeypatch):
    # 同じ時刻に T-15 バンドルが 3 レース（各 150 秒）。A の T-2（10:03）はバンドルの最中に過ぎるはずだが、
    # 次のバンドルに入る前（10:02:30）に 30 秒待って 10:03:00 に取る
    sched = [_race("A", 10, 5), _race("X", 10, 15), _race("Y", 10, 15), _race("Z", 10, 15)]
    calls = _run_runner(monkeypatch, sched, _t(8, 30), late_env=True, bundle_sec=150)
    got = {(c[1], c[2][0]): c[0] for c in calls}
    assert got[("A", "race_odds_t05")] == "10:00:00"
    assert got[("A", "race_odds_t02")] == "10:03:00"
    assert got[("Y", "race_shutuba")] == "10:03:00" and got[("X", "race_shutuba")] == "10:00:00"
    # 取りこぼし（スキップ）が出るのは、バンドルが長くて予定から 120 秒を超えた場合だけ
    for (rid, task), at in got.items():
        if task in ODDS_LATE_TASKS:
            assert at is not None
