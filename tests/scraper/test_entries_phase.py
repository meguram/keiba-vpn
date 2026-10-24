"""出馬表の段階の判定・差分・収集スケジュール（純粋関数）のテスト。"""

from datetime import date, datetime

from src.scraper import entries_phase as EP

JST = EP.JST


def _ent(n, **kw):
    base = {"horse_id": f"h{n}", "horse_name": f"馬{n}", "horse_number": 0, "bracket_number": 0, "jockey_id": "", "jockey_weight": 0, "weight": 0}
    base.update(kw)
    return base


def _card(entries, **kw):
    return {"race_id": "202608040801", "entries": entries, "start_time": "", **kw}


def test_none_when_no_entries():
    assert EP.classify_entries(_card([]), "20261024", datetime(2026, 10, 21, 20, 0, tzinfo=JST))["phase"] == EP.PHASE_NONE
    assert EP.classify_entries(None, "20261024")["phase"] == EP.PHASE_NONE


def test_registered_before_thursday_noon():
    card = _card([_ent(i) for i in range(1, 15)])
    r = EP.classify_entries(card, "20261024", datetime(2026, 10, 21, 20, 0, tzinfo=JST))
    assert r["phase"] == EP.PHASE_REGISTERED and r["n_entries"] == 14 and r["n_numbered"] == 0
    assert "暫定" in r["label"]


def test_entries_after_thursday_noon_for_saturday_and_sunday():
    card = _card([_ent(i) for i in range(1, 9)])
    assert EP.classify_entries(card, "20261024", datetime(2026, 10, 22, 12, 0, tzinfo=JST))["phase"] == EP.PHASE_ENTRIES
    assert EP.classify_entries(card, "20261024", datetime(2026, 10, 22, 11, 59, tzinfo=JST))["phase"] == EP.PHASE_REGISTERED
    # 日曜開催も同じ週の木曜が基準
    assert EP.classify_entries(card, "20261025", datetime(2026, 10, 22, 12, 30, tzinfo=JST))["phase"] == EP.PHASE_ENTRIES
    assert EP.classify_entries(card, "20261025", datetime(2026, 10, 21, 23, 0, tzinfo=JST))["phase"] == EP.PHASE_REGISTERED


def test_final_and_weighed():
    ents = [_ent(i, horse_number=i, bracket_number=1 + (i - 1) // 2, jockey_id=f"j{i}") for i in range(1, 9)]
    now = datetime(2026, 10, 23, 17, 0, tzinfo=JST)
    assert EP.classify_entries(_card(ents), "20261024", now)["phase"] == EP.PHASE_FINAL
    for e in ents:
        e["weight"] = 480
    assert EP.classify_entries(_card(ents), "20261024", datetime(2026, 10, 24, 9, 0, tzinfo=JST))["phase"] == EP.PHASE_WEIGHED


def test_partial_numbers_are_not_final():
    ents = [_ent(1, horse_number=1), _ent(2, horse_number=0), _ent(3, horse_number=3)]
    assert EP.classify_entries(_card(ents), "20261024", datetime(2026, 10, 23, 17, 0, tzinfo=JST))["phase"] == EP.PHASE_ENTRIES
    # 重複した馬番も確定とみなさない
    dup = [_ent(1, horse_number=1), _ent(2, horse_number=1)]
    assert EP.classify_entries(_card(dup), "20261024", datetime(2026, 10, 23, 17, 0, tzinfo=JST))["phase"] != EP.PHASE_FINAL


def test_signature_changes_on_scratch_and_jockey_not_on_weight():
    a = _card([_ent(1, horse_number=1, jockey_id="j1"), _ent(2, horse_number=2, jockey_id="j2")])
    b = _card([_ent(1, horse_number=1, jockey_id="j1", weight=480), _ent(2, horse_number=2, jockey_id="j2", weight=500)])
    assert EP.entries_signature(a) == EP.entries_signature(b)
    c = _card([_ent(1, horse_number=1, jockey_id="j1")])
    assert EP.entries_signature(a) != EP.entries_signature(c)
    d = _card([_ent(1, horse_number=1, jockey_id="j9"), _ent(2, horse_number=2, jockey_id="j2")])
    assert EP.entries_signature(a) != EP.entries_signature(d)


def test_diff_entries():
    old = _card([_ent(1, jockey_id="j1", jockey_name="A", jockey_weight=55), _ent(2, jockey_id="j2", jockey_name="B")])
    new = _card([_ent(1, jockey_id="j7", jockey_name="C", jockey_weight=56), _ent(3, jockey_id="j3")])
    types = {(c["type"], c["horse_id"]) for c in EP.diff_entries(old, new)}
    assert ("removed", "h2") in types and ("added", "h3") in types
    assert ("jockey_change", "h1") in types and ("weight_carried_change", "h1") in types
    # 騎手が未定（空）から決まるのは変更ではない
    old2 = _card([_ent(1, jockey_id="")])
    new2 = _card([_ent(1, jockey_id="j1")])
    assert EP.diff_entries(old2, new2) == []


SCHEDULE = {
    "window_minutes": 25,
    "checkpoints": {"sun": ["18:30"], "thu": ["12:30", "15:30"], "fri": ["16:05", "16:30"], "sat": ["10:05"]},
    "critical": {"fri": ["16:05"], "sat": ["10:05"]},
}


def test_due_checkpoint_window_and_done():
    fri = lambda h, m: datetime(2026, 10, 23, h, m, tzinfo=JST)  # noqa: E731
    assert EP.due_checkpoint(fri(16, 4), SCHEDULE) is None
    cp = EP.due_checkpoint(fri(16, 5), SCHEDULE)
    assert cp and cp["label"] == "2026-10-23 16:05" and cp["critical"] is True
    assert EP.due_checkpoint(fri(16, 29), SCHEDULE)["label"] == "2026-10-23 16:05"      # 窓内（25 分）かつ 16:30 は未到達
    assert EP.due_checkpoint(fri(16, 31), SCHEDULE)["label"] == "2026-10-23 16:30"      # 新しい方を優先
    assert EP.due_checkpoint(fri(16, 31), SCHEDULE)["critical"] is False
    assert EP.due_checkpoint(fri(16, 10), SCHEDULE, ["2026-10-23 16:05"]) is None       # 実施済み
    assert EP.due_checkpoint(fri(17, 0), SCHEDULE) is None                                # 窓の外


def test_due_checkpoint_handles_utc_input():
    utc = datetime(2026, 10, 23, 7, 5, tzinfo=EP.ZoneInfo("UTC"))                         # = 16:05 JST
    assert EP.due_checkpoint(utc, SCHEDULE)["label"] == "2026-10-23 16:05"


def test_target_dates():
    out = EP.target_dates(["20261017", "20261024", "20261025", "20261101", "20261114"], date(2026, 10, 21), 14)
    assert out == ["20261024", "20261025", "20261101"]
    assert EP.target_dates(["20261101"], date(2026, 10, 21), 10) == []


def test_should_refetch_policy():
    base = dict(today=date(2026, 10, 23), list_count_changed=False)
    d = date(2026, 10, 24)
    assert EP.should_refetch(race_date=d, prev_phase=None, critical=False, **base)
    assert EP.should_refetch(race_date=d, prev_phase=EP.PHASE_REGISTERED, critical=False, **base)
    assert EP.should_refetch(race_date=d, prev_phase=EP.PHASE_ENTRIES, critical=False, **base)
    assert not EP.should_refetch(race_date=d, prev_phase=EP.PHASE_FINAL, critical=False, **base)       # 確定後は critical だけ
    assert EP.should_refetch(race_date=d, prev_phase=EP.PHASE_FINAL, critical=True, **base)
    assert not EP.should_refetch(race_date=date(2026, 10, 31), prev_phase=EP.PHASE_FINAL, critical=True, **base)   # まだ先の週
    assert EP.should_refetch(race_date=d, prev_phase=EP.PHASE_FINAL, critical=False, today=base["today"], list_count_changed=True)
