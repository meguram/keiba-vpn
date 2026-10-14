from datetime import datetime, timedelta, timezone

from src.scraper import recent_retry as rr

JST = timezone(timedelta(hours=9))
NOW = datetime(2026, 10, 14, 6, 0, tzinfo=JST)


def _cal(*dates):
    return {"race_days": [{"date": d} for d in dates]}


def test_due_dates_are_past_days_within_lookback():
    cal = _cal("2026-09-01", "2026-09-20", "2026-10-10", "2026-10-11", "2026-10-12", "2026-10-14", "2026-10-17")
    assert rr.due_dates(cal, NOW) == ["20261010", "20261011", "20261012"]  # 当日(10/14)・未来・21日より前は除く


def test_should_try_respects_interval_and_cap():
    assert rr.should_try({}, "a", NOW)
    st = {"a": {"attempts": 1, "last": (NOW - timedelta(hours=1)).isoformat()}}
    assert not rr.should_try(st, "a", NOW)
    st["a"]["last"] = (NOW - timedelta(hours=4)).isoformat()
    assert rr.should_try(st, "a", NOW)
    st["a"]["attempts"] = rr.MAX_ATTEMPTS
    assert not rr.should_try(st, "a", NOW)


def test_select_targets_skips_existing_orders_by_category_then_date_and_caps():
    by_date = {"20261011": ["r1", "r2"], "20261010": ["r0"]}
    have = {("race_result", "r2"), ("race_index", "r0")}
    got = rr.select_targets(by_date, lambda c, r: (c, r) in have, {}, NOW)
    cats = [(c, r) for c, _, r in got]
    assert cats[:2] == [("race_result", "r0"), ("race_result", "r1")]  # 確定結果が先、日付の古い順
    assert ("race_result", "r2") not in cats and ("race_index", "r0") not in cats
    assert len(rr.select_targets(by_date, lambda c, r: False, {}, NOW, max_requests=3)) == 3


def test_select_targets_skips_recently_tried():
    st = {"race_result/r1": {"attempts": 2, "last": (NOW - timedelta(minutes=30)).isoformat()}}
    got = rr.select_targets({"20261011": ["r1"]}, lambda c, r: False, st, NOW, categories=[("race_result", "scrape_race_result")])
    assert got == []


def test_state_roundtrip(tmp_path):
    p = tmp_path / "s.json"
    assert rr.load_state(p) == {}
    rr.save_state({"a": {"attempts": 1}}, p)
    assert rr.load_state(p) == {"a": {"attempts": 1}}
