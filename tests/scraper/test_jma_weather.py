from datetime import date
from pathlib import Path

import pytest

from src.scraper import jma_weather as jw

FX = Path(__file__).parent / "fixtures"


def test_parse_hourly_s1_and_a1():
    s1 = jw.parse_hourly((FX / "jma_hourly_s1_sapporo_20261018.html").read_text(encoding="utf-8"))
    assert len(s1) == 24 and s1[0]["hour"] == 1 and s1[-1]["hour"] == 24
    r = s1[12]
    assert r["hour"] == 13 and r["wind_dir"] == "東南東" and r["wind_dir_deg"] == 112.5 and r["wind_speed_ms"] == 4.2 and r["temp_c"] == 15.4
    a1 = jw.parse_hourly((FX / "jma_hourly_a1_toyonaka_20261018.html").read_text(encoding="utf-8"))
    assert len(a1) == 24 and a1[12]["wind_speed_ms"] == 2.8 and a1[12]["wind_dir"] == "東南東" and a1[0]["humidity_pct"] is None      # アメダスの『平均風速』も読める・未観測は None


def test_parse_hourly_errors_on_structure_change():
    with pytest.raises(jw.JmaError):
        jw.parse_hourly("<html><body>メンテナンス中</body></html>")


def test_num_flags_and_missing():
    assert jw._num("5.4 )") == (5.4, ")") and jw._num("12.0 ]") == (12.0, "]")
    assert jw._num("×") == (None, "") and jw._num("///") == (None, "") and jw._num("--") == (0.0, "")


def test_parse_stations_and_nearest():
    st = jw.parse_stations((FX / "jma_prefecture_14.html").read_text(encoding="utf-8", errors="ignore"), 14)
    assert any(s["name"] == "札幌" and s["type"] == "s" and s["block_no"] == "47412" for s in st)
    assert all(s["ended"] for s in st if s["name"] == "阿曽岩")                                  # 終了年が入っている観測所は現行ではない
    near = jw.nearest_station(jw.COURSES[0], st)
    assert near["name"] == "札幌" and near["distance_km"] < 3


def test_wind_directions_cover_16_points():
    assert len(jw.WIND_DEG) == 16 and jw.WIND_DEG["北"] == 0.0 and jw.WIND_DEG["西"] == 270.0


class _Store:
    def __init__(self):
        self.d = {}

    def load(self, c, k, bypass_cache=False):
        return self.d.get((c, k))

    def save(self, c, k, data):
        self.d[(c, k)] = data
        return True

    def list_keys(self, c):
        return [k for (cc, k) in self.d if cc == c]


def test_fetch_range_skips_stored_days_and_saves(monkeypatch):
    store = _Store()
    html = (FX / "jma_hourly_s1_sapporo_20261018.html").read_text(encoding="utf-8")
    calls = []
    monkeypatch.setattr(jw, "_get", lambda url, session: calls.append(url) or html)
    monkeypatch.setattr(jw, "load_stations", lambda: {"01": {"station": {"type": "s", "prec_no": 14, "block_no": "47412", "name": "札幌", "lat": 43.06, "lon": 141.33, "distance_km": 1.5}}})
    r = jw.fetch_range(date(2020, 1, 1), date(2020, 1, 2), store=store)
    assert r == {"fetched": 2, "skipped": 0} and len(calls) == 2
    r2 = jw.fetch_range(date(2020, 1, 1), date(2020, 1, 2), store=store)
    assert r2 == {"fetched": 0, "skipped": 2} and len(calls) == 2                      # 取得済みの日は取り直さない
    assert jw.report(store, years=[2020]) == {"01": {2020: 2}}


def test_save_failure_is_an_error():
    class _Bad(_Store):
        def save(self, c, k, data):
            return False

    with pytest.raises(jw.JmaError):
        jw.save_month({"venue_code": "01", "month": "2020-01", "days": {}}, _Bad())


def test_month_object_layout_and_observed_at():
    html = (FX / "jma_hourly_s1_sapporo_20261018.html").read_text(encoding="utf-8")
    recs = jw.with_observed_at(jw.parse_hourly(html), date(2026, 10, 18))
    assert recs[0]["observed_at"] == "2026-10-18T01:00:00+09:00" and recs[-1]["observed_at"] == "2026-10-19T00:00:00+09:00"      # 24 時は翌日 0 時
    store = _Store()
    import types
    jw_get = lambda url, session: html      # noqa: E731
    mp = pytest.MonkeyPatch()
    mp.setattr(jw, "_get", jw_get)
    mp.setattr(jw, "load_stations", lambda: {"01": {"station": {"type": "s", "prec_no": 14, "block_no": "47412", "name": "札幌", "lat": 43.06, "lon": 141.33, "distance_km": 1.5}}})
    try:
        jw.fetch_range(date(2020, 1, 31), date(2020, 2, 1), store=store)
    finally:
        mp.undo()
    assert sorted(k for _, k in store.d) == ["01_202001", "01_202002"]                 # 月ごとに別オブジェクト
    day = store.d[("jma_weather_hourly", "01_202001")]["days"]["2020-01-31"]
    assert day["station"]["name"] == "札幌" and day["fetched_at"] and len(day["hours"]) == 24
