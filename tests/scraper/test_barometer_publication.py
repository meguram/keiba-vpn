"""バロメーター(走行データ): レース翌週の金曜18時頃に公開・障害レースは対象外。未公開で無駄な取得や N/A 登録をしない。"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.scraper import run as R
from src.scraper.parsers import BarometerParser

JST = timezone(timedelta(hours=9))


@pytest.mark.parametrize("race,publish", [
    (date(2026, 10, 11), datetime(2026, 10, 16, 18, 0, tzinfo=JST)),   # 日曜 → 翌週金曜
    (date(2026, 10, 10), datetime(2026, 10, 16, 18, 0, tzinfo=JST)),   # 土曜
    (date(2026, 10, 12), datetime(2026, 10, 16, 18, 0, tzinfo=JST)),   # 月曜(祝日開催)も同じ週の扱い
    (date(2026, 9, 6), datetime(2026, 9, 11, 18, 0, tzinfo=JST)),
    (date(2026, 10, 9), datetime(2026, 10, 16, 18, 0, tzinfo=JST)),    # 金曜(同じ週の月〜日の範囲)
])
def test_publish_time_is_friday_18_of_the_following_week(race, publish):
    assert R.ScraperRunner.barometer_publish_at(race) == publish


def _runner(storage_docs, parser_result=None, now=None, monkeypatch=None):
    r = R.ScraperRunner.__new__(R.ScraperRunner)
    calls = {"parse": 0, "saved": [], "markers": []}

    class S:
        def exists(self, cat, key):
            return False

        def load(self, cat, key):
            return storage_docs.get((cat, key))

        def save(self, cat, key, data):
            calls["saved"].append((cat, key))
            return True

    class P:
        def parse_from_api(self, session, rid, html_out=None):
            calls["parse"] += 1
            if html_out is not None:
                html_out.append("<html>laps</html>")
            return parser_result or {"race_id": rid, "entries": []}

    r.storage, r.barometer_parser = S(), P()
    r.client = SimpleNamespace(is_logged_in=True, _session=object())
    r._save_not_available_marker = lambda cat, rid, reason="empty_response", force=False: calls["markers"].append((reason, force))
    if monkeypatch and now:
        class FakeDT(datetime):
            @classmethod
            def now(cls, tz=None):
                return now

        monkeypatch.setattr("datetime.datetime", FakeDT, raising=False)
    return r, calls


FLAT = {("race_result", "202609040212"): {"date": "2026-09-06", "surface": "芝", "race_name": "セントウルS"}}
STEEPLE = {("race_result", "202606040201"): {"date": "2026-09-06", "surface": "障", "race_name": "障害3歳以上未勝利"}}


def test_not_requested_before_publication_time(monkeypatch):
    r, calls = _runner({("race_result", "r1"): {"date": "2999-01-03", "surface": "芝"}})  # 遠い未来 = 必ず公開前
    assert r.scrape_barometer("r1", skip_existing=False) is None
    assert calls["parse"] == 0 and calls["saved"] == [] and calls["markers"] == []


def test_steeplechase_is_marked_not_applicable_without_request():
    r, calls = _runner(STEEPLE)
    assert r.scrape_barometer("202606040201", skip_existing=False) is None
    assert calls["parse"] == 0 and calls["markers"] == [("steeplechase", True)]


def test_published_flat_race_is_saved():
    ok = {"race_id": "202609040212", "entries": [{"horse_number": 1}]}
    r, calls = _runner(FLAT, parser_result=ok)
    assert r.scrape_barometer("202609040212", skip_existing=False) == ok
    assert calls["saved"][0] == ("race_barometer", "202609040212") and calls["markers"] == []


def test_not_published_notice_is_neither_saved_nor_marked():
    r, calls = _runner(FLAT, parser_result={"race_id": "202609040212", "entries": [], "not_published": True})
    assert r.scrape_barometer("202609040212", skip_existing=False) is None
    assert calls["parse"] == 1 and calls["saved"] == [] and calls["markers"] == []


def test_plain_empty_response_still_goes_through_marker_logic():
    r, calls = _runner(FLAT)
    assert r.scrape_barometer("202609040212", skip_existing=False) is None
    assert calls["markers"] == [("empty_response", False)]


def test_parser_flags_the_not_published_page():
    html = "<html>" + "x" * 200 + "<p>レース翌週の金曜日18時頃に公開予定です。※障害レースを除く</p></html>"

    class Resp:
        status_code, text = 200, html

    class Sess:
        def get(self, *a, **k):
            return Resp()

    out = BarometerParser().parse_from_api(Sess(), "202605040412")
    assert out["entries"] == [] and out.get("not_published") is True


def _laps(monkeypatch, result=None, exc=None):
    from src.scraper import parsers as P

    def fake(self, html, race_id=""):
        if exc:
            raise exc
        return result
    monkeypatch.setattr(P.DbHorseLaptimeTableParser, "parse", fake)


def test_horse_lap_is_saved_from_the_same_page(monkeypatch):
    """馬別ラップ（走法の軸の元データ）は、バロメーターと同じ HTML から保存する。取得は増えない（parse_from_api は 1 回）。"""
    _laps(monkeypatch, {"race_id": "202609040212", "entries": [
        {"horse_id": "h1", "finish_order": 1, "sections": [{"lap_time": 12.3}]},
        {"horse_id": "h2", "finish_order": 0, "sections": []}]})        # 中止の馬にラップが無いのは正常
    ok = {"race_id": "202609040212", "entries": [{"horse_number": 1}, {"horse_number": 2}]}
    r, calls = _runner(FLAT, parser_result=ok)
    assert r.scrape_barometer("202609040212", skip_existing=False) == ok
    assert calls["parse"] == 1
    assert calls["saved"] == [("race_barometer", "202609040212"), ("horse_lap", "202609040212")]


def test_horse_lap_without_laps_is_not_saved(monkeypatch):
    _laps(monkeypatch, {"race_id": "202609040212", "entries": [{"horse_id": "h1", "finish_order": 1, "sections": [{"lap_time": None}]}]})
    r, calls = _runner(FLAT, parser_result={"race_id": "202609040212", "entries": [{"horse_number": 1}]})
    r.scrape_barometer("202609040212", skip_existing=False)
    assert calls["saved"] == [("race_barometer", "202609040212")]


def test_partial_horse_lap_is_not_saved(monkeypatch):
    """会員状態が効いていない応答（先頭の数頭だけ）は保存しない。"""
    _laps(monkeypatch, {"race_id": "202609040212", "entries": [{"horse_id": "h1", "finish_order": 1, "sections": [{"lap_time": 12.3}]}]})
    r, calls = _runner(FLAT, parser_result={"race_id": "202609040212", "entries": [{"horse_number": 1}, {"horse_number": 2}]})
    r.scrape_barometer("202609040212", skip_existing=False)
    assert calls["saved"] == [("race_barometer", "202609040212")]


def test_horse_lap_failure_does_not_stop_barometer(monkeypatch):
    _laps(monkeypatch, exc=ValueError("broken"))
    ok = {"race_id": "202609040212", "entries": [{"horse_number": 1}]}
    r, calls = _runner(FLAT, parser_result=ok)
    assert r.scrape_barometer("202609040212", skip_existing=False) == ok
    assert calls["saved"] == [("race_barometer", "202609040212")]


def test_parser_hands_back_the_html():
    html = "<html>" + "x" * 200 + "</html>"

    class Resp:
        status_code, text = 200, html

    class Sess:
        def get(self, *a, **k):
            return Resp()

    box: list[str] = []
    BarometerParser().parse_from_api(Sess(), "202605040412", html_out=box)
    assert box == [html]
