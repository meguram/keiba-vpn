from datetime import datetime

from src.scraper import backfill_guard as bg
from src.scraper.parsers import OddsParser


class _R:
    def __init__(self, code, body=None):
        self.status_code = code
        self._body = body

    def json(self):
        if self._body is None:
            raise ValueError("empty body")
        return self._body


class _Sess:
    def __init__(self, code, body=None, raises=False):
        self.code, self.body, self.raises = code, body, raises

    def get(self, url, timeout=0):
        if self.raises:
            raise TimeoutError("x")
        return _R(self.code, self.body)


def test_pair_odds_records_http_400_for_all_four_calls():
    p = OddsParser()
    out = p.parse_pair_odds_from_api(_Sess(400), "201904020701")
    assert not out["umaren"] and p.last_api_codes == (400, 400, 400, 400)


def test_pair_odds_records_exception_as_zero_and_empty_json_as_200():
    p = OddsParser()
    p.parse_pair_odds_from_api(_Sess(200, raises=True), "x")
    assert p.last_api_codes == (0, 0, 0, 0)
    p.parse_pair_odds_from_api(_Sess(200, {"status": "result", "data": {"odds": {}}}), "x")
    assert p.last_api_codes == (200, 200, 200, 200)


def test_win_place_records_codes():
    p = OddsParser()
    out = p.parse_from_api(_Sess(400), "x")
    assert out["entries"] == [] and p.last_api_codes == (400, 400)


def test_guard_counts_api_400_as_bad_and_stops(tmp_path):
    g = bg.NightGuard(cap=10**9, window="00:00-24:00", clock=lambda: datetime(2026, 10, 14, 12, tzinfo=bg.JST),
                      state_path=tmp_path / "s.json", sleep=lambda s: None, pace_per_hour=0, yield_to_others=False)
    g.add_api_codes((200, 200, 200, 200))
    assert g.stop_reason is None and g.bad == 0
    g.add_api_codes((400, 400, 400, 400))
    assert g.bad == 4 and g.status_counts[400] == 4 and "403/429/5xx" in (g.stop_reason or "")


def test_guard_counts_exception_as_transport_error(tmp_path):
    g = bg.NightGuard(cap=10**9, window="00:00-24:00", clock=lambda: datetime(2026, 10, 14, 12, tzinfo=bg.JST),
                      state_path=tmp_path / "s.json", sleep=lambda s: None, pace_per_hour=0, yield_to_others=False)
    g.add_api_codes((0,))
    assert g.transport_errors == 1 and g.bad == 0
