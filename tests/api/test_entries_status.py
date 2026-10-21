"""出馬表の段階の API 付与（entries_status）と、馬番が無い段階の応答（entries_only_predictions）のテスト。"""

from datetime import datetime

import pytest

from src.api import entries_status as ESt
from src.api.entries_only_predictions import entries_only_predictions
from src.scraper import entries_phase as EP
from src.scraper import entries_sync as ES

JST = EP.JST


class FakeStorage:
    def __init__(self, cards, odds=None):
        self.cards, self.odds = cards, odds or {}

    def load(self, category, key, bypass_cache=False):
        if category == "race_shutuba":
            return self.cards.get(key)
        if category == "race_odds":
            return self.odds.get(key)
        return None


def _ent(n, **kw):
    base = {"horse_id": f"h{n}", "horse_name": f"馬{n}", "horse_number": 0, "bracket_number": 0, "jockey_id": "", "jockey_weight": 0, "weight": 0}
    base.update(kw)
    return base


@pytest.fixture(autouse=True)
def _state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_ENTRIES_STATE_DIR", str(tmp_path / "meta" / "entries_state"))


def test_provisional_entries_are_returned_with_null_numbers():
    rid = "202608040801"
    st = FakeStorage({rid: {"race_id": rid, "date": "2026-10-24", "entries": [_ent(1, jockey_name="A"), _ent(2)]}})
    r = entries_only_predictions(rid, st)
    assert r["status"] == "no_prediction" and r["provisional_entries"] is True and r["has_prediction"] is False
    assert [p["horse_number"] for p in r["predictions"]] == [None, None]
    assert all(p["horse_number_provisional"] for p in r["predictions"])
    assert r["predictions"][0]["jockey_name"] == "A" and "確定前" in r["message"]


def test_final_entries_keep_the_numbered_response_and_partial_numbers_are_provisional():
    rid = "202608040801"
    final = FakeStorage({rid: {"race_id": rid, "entries": [_ent(1, horse_number=1), _ent(2, horse_number=2)]}})
    r = entries_only_predictions(rid, final)
    assert [p["horse_number"] for p in r["predictions"]] == [1, 2] and "provisional_entries" not in r
    partial = FakeStorage({rid: {"race_id": rid, "entries": [_ent(1, horse_number=1), _ent(2)]}})
    assert entries_only_predictions(rid, partial)["provisional_entries"] is True
    none = FakeStorage({})
    assert entries_only_predictions(rid, none)["status"] == "no_entries"


def test_entries_status_from_state_file():
    rid = "202608040801"
    card = {"race_id": rid, "entries": [_ent(1), _ent(2)]}
    state, _ = ES.update_state(None, None, card, rid, "20261024", datetime(2026, 10, 22, 13, 0, tzinfo=JST), 2)
    ES._atomic_write(ES.state_path(rid), state)
    out = ESt.entries_status(rid, FakeStorage({}), now=datetime(2026, 10, 22, 14, 0, tzinfo=JST))
    assert out["phase"] == "entries" and out["label"] == "出走馬決定" and out["source"] == "state"
    assert out["updated_at"].startswith("2026-10-22T13:00") and out["is_future"] is True and out["race_date"] == "2026-10-24"


def test_entries_status_fallback_from_card_and_none():
    rid = "202608040801"
    card = {"race_id": rid, "date": "2026-10-24", "entries": [_ent(n, horse_number=n, bracket_number=n) for n in (1, 2)]}
    out = ESt.entries_status(rid, FakeStorage({rid: card}), now=datetime(2026, 10, 23, 17, 0, tzinfo=JST))
    assert out["phase"] == "final" and out["source"] == "card"
    none = ESt.entries_status("202608040802", FakeStorage({}))
    assert none["phase"] == "none" and none["updated_at"] is None


def test_attach_is_backward_compatible_and_race_list_fields():
    rid = "202608040801"
    resp = {"status": "ok", "predictions": [{"horse_number": 1}]}
    out = ESt.attach_entries_status(resp, rid, FakeStorage({}))
    assert out["predictions"] == resp["predictions"] and out["entries_status"]["phase"] == "none"
    assert ESt.attach_entries_status("not a dict", rid) == "not a dict"
    card = {"race_id": rid, "entries": [_ent(1)]}
    state, _ = ES.update_state(None, None, card, rid, "20261024", datetime(2026, 10, 21, 20, 0, tzinfo=JST), 1)
    ES._atomic_write(ES.state_path(rid), state)
    f = ESt.race_list_fields([rid, "202608040802"])
    assert set(f) == {rid} and f[rid]["entries_phase"] == "registered" and f[rid]["entries_label"] == "特別登録（暫定）"
