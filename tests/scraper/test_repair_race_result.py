from src.scripts.data.repair_race_result_from_on_time import repair_entries


def _rr(**kw):
    e = {"horse_number": 1, "finish_position": 1, "last_3f": 94.0, "passing_order": "105", "weight": 1, "weight_change": 0,
         "odds": 0, "popularity": 108}
    e.update(kw)
    return {"entries": [e, {**e, "horse_number": 2, "finish_position": 2}]}


def _ot():
    good = {"last_3f": 33.2, "passing_order": "2-2", "weight": 492, "weight_change": 4, "odds": 3.4, "popularity": 1}
    return {"entries": [{"horse_number": 1, **good}, {"horse_number": 2, **good, "last_3f": 34.0}]}


def test_invalid_fields_are_replaced_and_valid_ones_kept():
    rr = _rr()
    rr["entries"][1].update(last_3f=35.5)  # 正常な値は触らない
    fixed = repair_entries(rr, _ot())
    e1, e2 = rr["entries"]
    assert (e1["last_3f"], e1["passing_order"], e1["weight"], e1["odds"], e1["popularity"]) == (33.2, "2-2", 492, 3.4, 1)
    assert e2["last_3f"] == 35.5 and e2["passing_order"] == "2-2"
    assert fixed["last_3f"] == 1 and fixed["passing_order"] == 2


def test_nothing_changes_when_on_time_is_also_invalid():
    ot = {"entries": [{"horse_number": 1, "last_3f": 0.0, "passing_order": "", "weight": 0, "odds": 0, "popularity": 0}]}
    rr = _rr()
    assert sum(repair_entries(rr, ot).values()) == 0
