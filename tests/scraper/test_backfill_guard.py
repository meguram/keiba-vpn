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


def test_stage_cap_halves_first_night_and_after_unhealthy():
    assert bg.stage_cap({"nights": []}) == bg.CAP_MAX // 2
    assert bg.stage_cap({"nights": [{"healthy": True}]}) == bg.CAP_MAX
    assert bg.stage_cap({"nights": [{"healthy": True}, {"healthy": False}]}) == bg.CAP_MAX // 2


def test_window_and_raceday():
    assert bg.in_window(_at(3), "02:30-05:45") and not bg.in_window(_at(6), "02:30-05:45")
    assert not bg.is_raceday(_at(3, day=14)) and bg.is_raceday(_at(3, day=17))   # 10/17 は土曜


def test_cap_reached_stops(tmp_path):
    g = bg.NightGuard(cap=5, window="00:00-23:59", clock=lambda: _at(3), state_path=tmp_path / "s.json")
    for _ in range(4):
        g("https://db.netkeiba.com/race/1/", _Resp())
    assert g.should_stop() is None
    g("https://db.netkeiba.com/race/1/", _Resp())
    assert "上限" in g.should_stop()


def test_other_hosts_are_not_counted(tmp_path):
    g = bg.NightGuard(cap=2, window="00:00-23:59", clock=lambda: _at(3), state_path=tmp_path / "s.json")
    g("https://example.com/x", _Resp())
    assert g.requests == 0


def test_bad_statuses_stop_the_night(tmp_path):
    g = bg.NightGuard(cap=1000, window="00:00-23:59", clock=lambda: _at(3), state_path=tmp_path / "s.json")
    for code in (503, 200, 503, 429):
        g("https://db.netkeiba.com/race/1/", _Resp(code))
    assert g.stop_reason and not g.healthy()


def test_404_is_not_bad(tmp_path):
    g = bg.NightGuard(cap=1000, window="00:00-23:59", clock=lambda: _at(3), state_path=tmp_path / "s.json")
    for _ in range(10):
        g("https://db.netkeiba.com/race/1/", _Resp(404))
    assert g.stop_reason is None and g.healthy()


def test_latency_spike_stops(tmp_path):
    g = bg.NightGuard(cap=1000, window="00:00-23:59", clock=lambda: _at(3), state_path=tmp_path / "s.json")
    for _ in range(bg.LATENCY_WINDOW):
        g("https://db.netkeiba.com/race/1/", _Resp(200, 0.4))
    assert g.stop_reason is None
    for _ in range(bg.LATENCY_WINDOW):
        g("https://db.netkeiba.com/race/1/", _Resp(200, 3.5))
    assert g.stop_reason and "遅延" in g.stop_reason


def test_transport_errors_stop(tmp_path):
    g = bg.NightGuard(cap=1000, window="00:00-23:59", clock=lambda: _at(3), state_path=tmp_path / "s.json")
    g.on_error("https://db.netkeiba.com/x", TimeoutError("t"))
    assert g.stop_reason is None
    g.on_error("https://db.netkeiba.com/x", TimeoutError("t"))
    assert g.stop_reason


def test_outside_window_stops(tmp_path):
    g = bg.NightGuard(cap=10, window="02:30-05:45", clock=lambda: _at(6), state_path=tmp_path / "s.json")
    assert "時間窓" in g.should_stop()


def test_record_persists_and_drives_next_cap(tmp_path):
    p = tmp_path / "s.json"
    g = bg.NightGuard(cap=1000, window="00:00-23:59", clock=lambda: _at(3), state_path=p)
    g("https://db.netkeiba.com/race/1/", _Resp())
    g.record({"ended_by": "x"})
    st = bg.load_state(p)
    assert st["nights"][-1]["requests"] == 1 and st["nights"][-1]["healthy"]
    assert bg.stage_cap(st) == bg.CAP_MAX


def test_wait_for_quiet_gives_up():
    calls = []
    assert bg.wait_for_quiet(max_wait_sec=0, poll=0, busy=lambda: True, sleep=lambda s: calls.append(s)) is False
    assert bg.wait_for_quiet(max_wait_sec=5, poll=0, busy=lambda: False) is True
