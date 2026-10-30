from datetime import datetime

from src.scraper import backfill_guard as bg


class _Resp:
    def __init__(self, code=200, sec=0.3):
        self.status_code = code

        class _E:
            def total_seconds(self_inner):
                return sec

        self.elapsed = _E()


def _at(h, m=0, day=14):  # 2026-10-14 は水曜
    return datetime(2026, 10, day, h, m, tzinfo=bg.JST)


def _limited(monkeypatch, total=2700, cap_max=1500):
    monkeypatch.setattr(bg, "TOTAL_DAILY_LIMIT", total)
    monkeypatch.setattr(bg, "CAP_MAX", cap_max)


def test_default_has_no_caps():
    assert bg.TOTAL_DAILY_LIMIT == 0 and bg.CAP_MAX == 0 and bg.PACE_PER_HOUR == 0
    assert bg.stage_cap({"nights": []}) == bg.UNLIMITED and bg.daily_budget(10**6) == bg.UNLIMITED
    assert bg.MAX_BAD_IN_RECENT == 1 and bg.MAX_TRANSPORT_ERRORS == 1


def test_stage_cap_halves_first_night_and_after_unhealthy(monkeypatch):
    _limited(monkeypatch)
    assert bg.stage_cap({"nights": []}) == bg.CAP_MAX // 2
    assert bg.stage_cap({"nights": [{"healthy": True}]}) == bg.CAP_MAX
    assert bg.stage_cap({"nights": [{"healthy": True}, {"healthy": False}]}) == bg.CAP_MAX // 2


def test_window_and_raceday():
    assert bg.in_window(_at(3), "02:30-05:45") and not bg.in_window(_at(6), "02:30-05:45")
    assert not bg.is_raceday(_at(3, day=14)) and bg.is_raceday(_at(3, day=17))   # 10/17 は土曜


def test_cap_reached_stops(tmp_path):
    g = bg.NightGuard(cap=5, window="00:00-23:59", clock=lambda: _at(3), state_path=tmp_path / "s.json", sleep=lambda s: None, pace_per_hour=0, yield_to_others=False)
    for _ in range(4):
        g("https://db.netkeiba.com/race/1/", _Resp())
    assert g.should_stop() is None
    g("https://db.netkeiba.com/race/1/", _Resp())
    assert "上限" in g.should_stop()


def test_other_hosts_are_not_counted(tmp_path):
    g = bg.NightGuard(cap=2, window="00:00-23:59", clock=lambda: _at(3), state_path=tmp_path / "s.json", sleep=lambda s: None, pace_per_hour=0, yield_to_others=False)
    g("https://example.com/x", _Resp())
    assert g.requests == 0


def test_bad_statuses_stop_the_night(tmp_path):
    g = bg.NightGuard(cap=1000, window="00:00-23:59", clock=lambda: _at(3), state_path=tmp_path / "s.json", sleep=lambda s: None, pace_per_hour=0, yield_to_others=False)
    g("https://db.netkeiba.com/race/1/", _Resp(200))
    assert g.stop_reason is None
    g("https://db.netkeiba.com/race/1/", _Resp(503))
    assert g.stop_reason and not g.healthy()          # エラー 1 件で即中断


def test_400_is_bad_and_error_writes_halt_mark(tmp_path):
    g = bg.NightGuard(cap=1000, window="00:00-23:59", clock=lambda: _at(3), state_path=tmp_path / "s.json", sleep=lambda s: None, pace_per_hour=0, yield_to_others=False)
    g("https://db.netkeiba.com/race/1/", _Resp(400))
    assert g.stop_reason
    g.record({})
    assert (tmp_path / "backfill_past_halt.json").exists()


def test_clean_night_writes_no_halt_mark(tmp_path):
    g = bg.NightGuard(cap=1000, window="00:00-23:59", clock=lambda: _at(3), state_path=tmp_path / "s.json", sleep=lambda s: None, pace_per_hour=0, yield_to_others=False)
    g("https://db.netkeiba.com/race/1/", _Resp(200))
    g.record({})
    assert not (tmp_path / "backfill_past_halt.json").exists()


def test_404_is_not_bad(tmp_path):
    g = bg.NightGuard(cap=1000, window="00:00-23:59", clock=lambda: _at(3), state_path=tmp_path / "s.json", sleep=lambda s: None, pace_per_hour=0, yield_to_others=False)
    for _ in range(10):
        g("https://db.netkeiba.com/race/1/", _Resp(404))
    assert g.stop_reason is None and g.healthy()


def test_latency_spike_stops(tmp_path):
    g = bg.NightGuard(cap=1000, window="00:00-23:59", clock=lambda: _at(3), state_path=tmp_path / "s.json", sleep=lambda s: None, pace_per_hour=0, yield_to_others=False)
    for _ in range(bg.LATENCY_WINDOW):
        g("https://db.netkeiba.com/race/1/", _Resp(200, 0.4))
    assert g.stop_reason is None
    for _ in range(bg.LATENCY_WINDOW):
        g("https://db.netkeiba.com/race/1/", _Resp(200, 3.5))
    assert g.stop_reason and "遅延" in g.stop_reason


def test_transport_errors_stop(tmp_path):
    g = bg.NightGuard(cap=1000, window="00:00-23:59", clock=lambda: _at(3), state_path=tmp_path / "s.json", sleep=lambda s: None, pace_per_hour=0, yield_to_others=False)
    g.on_error("https://db.netkeiba.com/x", TimeoutError("t"))
    assert g.stop_reason          # 通信エラー 1 件で即中断


def test_outside_window_stops(tmp_path):
    g = bg.NightGuard(cap=10, window="02:30-05:45", clock=lambda: _at(6), state_path=tmp_path / "s.json", sleep=lambda s: None, pace_per_hour=0, yield_to_others=False)
    assert "時間窓" in g.should_stop()


def test_record_persists_and_drives_next_cap(tmp_path, monkeypatch):
    _limited(monkeypatch)
    p = tmp_path / "s.json"
    g = bg.NightGuard(cap=1000, window="00:00-23:59", clock=lambda: _at(3), state_path=p, sleep=lambda s: None, pace_per_hour=0, yield_to_others=False)
    g("https://db.netkeiba.com/race/1/", _Resp())
    g.record({"ended_by": "x"})
    st = bg.load_state(p)
    assert st["nights"][-1]["requests"] == 1 and st["nights"][-1]["healthy"]
    assert bg.stage_cap(st) == bg.CAP_MAX


def test_wait_for_quiet_gives_up():
    calls = []
    assert bg.wait_for_quiet(max_wait_sec=0, poll=0, busy=lambda: True, sleep=lambda s: calls.append(s)) is False
    assert bg.wait_for_quiet(max_wait_sec=5, poll=0, busy=lambda: False) is True


def test_daily_budget_and_caps(monkeypatch):
    _limited(monkeypatch)
    assert bg.CAP_MAX == 1500 and bg.stage_cap({"nights": []}) == 750
    assert bg.daily_budget(0) == bg.TOTAL_DAILY_LIMIT - bg.EXPECTED_WEEKDAY_OPS
    assert bg.daily_budget(10_000) == 0


def test_request_counter(tmp_path, monkeypatch):
    from src.scraper import request_counter as rc

    monkeypatch.setattr(rc, "DIR", tmp_path)
    assert rc.count_today() == 0
    rc.count_request(3)
    rc.count_request()
    assert rc.count_today() == 4


def test_aimd_slows_on_bad_and_recovers(tmp_path, monkeypatch):
    monkeypatch.setattr(bg, "MAX_BAD_IN_RECENT", 3)       # AIMD は『中断しきい値に達する前』の調整なので、しきい値を戻して確かめる
    slept = []
    g = bg.NightGuard(cap=10_000, window="00:00-23:59", clock=lambda: _at(3), state_path=tmp_path / "s.json", sleep=slept.append, pace_per_hour=0, yield_to_others=False)
    for _ in range(bg.LATENCY_WINDOW):
        g("https://db.netkeiba.com/race/1/", _Resp(200, 0.3))     # 基準
    assert g.extra_delay == 0 and not slept
    g("https://db.netkeiba.com/race/1/", _Resp(503, 0.3))         # 悪化 → 待ちが入る
    assert g.extra_delay == bg.AIMD_FIRST_DELAY and slept == [bg.AIMD_FIRST_DELAY]
    for _ in range(bg.AIMD_CLEAN_STEP):
        g("https://db.netkeiba.com/race/1/", _Resp(200, 0.3))
    assert g.extra_delay == bg.AIMD_FIRST_DELAY - bg.AIMD_RECOVER_SEC   # 加法的に戻る


def test_constant_pace_and_yield(tmp_path):
    slept: list[float] = []
    busy = iter([True, True, False])
    g = bg.NightGuard(cap=1000, window="00:00-24:00", clock=lambda: _at(15), state_path=tmp_path / "s.json", sleep=slept.append,
                      pace_per_hour=60, yield_to_others=True)
    g.busy = lambda: next(busy)
    g("https://db.netkeiba.com/race/1/", _Resp(200))
    assert abs(sum(s for s in slept if s <= 5.0) - 60.0) < 1e-6      # 60 件/時＝60 秒間隔（5 秒刻み）
    assert slept.count(15.0) == 2                                     # 他の取得が動いている間は 15 秒ずつ譲る


def test_all_day_window_and_total_limit(tmp_path, monkeypatch):
    _limited(monkeypatch)
    assert bg.in_window(_at(15), "00:00-24:00") and bg.in_window(_at(23), "00:00-24:00")
    g = bg.NightGuard(cap=1000, clock=lambda: _at(15), state_path=tmp_path / "s.json", sleep=lambda s: None, pace_per_hour=0, yield_to_others=False)
    monkeypatch.setattr("src.scraper.request_counter.count_today", lambda day=None: bg.TOTAL_DAILY_LIMIT)
    assert "日の総件数" in (g.should_stop() or "")


def test_other_fetch_pattern_covers_netkeiba_fetchers():
    import re

    for args in ["python -m src.scripts.scraping.fetch_sire_lines", "python -m src.scripts.scraping.backfill_horse_training_recent --limit 5",
                 "python -m src.scraper.auto_scrape"]:
        assert re.search(bg.OTHER_FETCH_PATTERN, args), args
    assert not re.search(bg.OTHER_FETCH_PATTERN, "python main.py --flask-api")


def test_shutdown_signal_breaks_yield_wait(tmp_path, monkeypatch):
    from src.scraper import backfill as bf

    g = bg.NightGuard(cap=0, state_path=tmp_path / "s.json", sleep=lambda s: None, yield_to_others=True)
    g.busy = lambda: True
    monkeypatch.setattr(bf, "_GUARD", g)
    monkeypatch.setattr(bf, "_shutdown_requested", False)
    bf._signal_handler(15, None)
    g._pace()   # 他の取得が動き続けていても、待ちを抜けて戻る
    assert g.stop_reason and bf._should_stop()


def test_yield_wait_logs_periodically(tmp_path, caplog):
    import logging

    n = int(bg.YIELD_LOG_EVERY_SEC // 15) * 2 + 1          # 2 回分の間隔より少し長く譲る
    busy = iter([True] * n + [False])
    g = bg.NightGuard(cap=0, state_path=tmp_path / "s.json", sleep=lambda s: None, pace_per_hour=0, yield_to_others=True)
    g.busy = lambda: next(busy)
    with caplog.at_level(logging.INFO, logger=bg.__name__):
        g._pace()
    assert sum("待機中" in r.getMessage() for r in caplog.records) == 3   # 0 分・10 分・20 分（固まり検知の 40 分より短い間隔で出る）


def test_other_fetchers_do_not_wait_for_past_backfill():
    """--phase past は他の取得に譲る。相手もこちらを待つと、お互いに待ち合って止まる（2026-10-29）。"""
    import re

    from src.scripts.scraping import backfill_horse_race_remarks as remarks
    from src.scripts.scraping import backfill_horse_training_recent as training

    past = "/opt/conda/bin/python3 -m src.scraper.backfill --phase past"
    assert re.search(bg.OTHER_FETCH_PATTERN, "/opt/conda/bin/python3 -m src.scripts.scraping.backfill_horse_training_recent --from 2026-04-01")
    for pattern in (training.OTHER_SCRAPERS, remarks.OTHER_SCRAPERS):
        assert not re.search(pattern, past)
        assert re.search(pattern, "/opt/conda/bin/python3 -m src.scraper.backfill --phase horse")   # 他の段階は従来どおり待つ
        assert not re.search(pattern, "/opt/conda/bin/python3 -m src.scripts.scraping.fetch_status")   # 状況表示（読み取りのみ）は待つ対象にしない


def test_default_runs_alongside_other_fetchers(tmp_path):
    """既定では他の取得に譲らず並走する（BACKFILL_PAST_YIELD=1 で譲る）。"""
    assert bg.YIELD_TO_OTHERS is False
    slept: list[float] = []
    g = bg.NightGuard(cap=1000, state_path=tmp_path / "s.json", sleep=slept.append, pace_per_hour=0)
    g.busy = lambda: True
    g("https://db.netkeiba.com/race/1/", _Resp(200))
    assert slept == []
