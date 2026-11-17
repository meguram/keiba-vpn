"""過去該当馬の足すだけの更新（picks.append_history）。"""

import pytest

from src.research.graded import picks as PK
from src.research.graded.versions import V0_DATA_END, V0_ID

UNTIL = "2026-11-16"
assert V0_DATA_END < UNTIL


def _h(hn, fin=1.0):
    return {"hn": hn, "name": f"馬{hn}", "odds": 3.0, "pop": 1, "fin": fin, "tan": 300, "fuku": 130}


def _race(rid, date, hm=1, kill=(5,), winners=True):
    return {"race_id": rid, "date": date, "venue": "東京", "field": 10, "pay_ok": True, "winners": [_h(1)] if winners else [], "base": _h(hm),
            "B": {"honmei": _h(hm), "kill": [_h(k, 9.0) for k in kill]}}


def _hist(races, until=UNTIL):
    return {"sha256_rules": "s", "frozen_at": "f", "data_until": until, "rkeys": {"k": {"rkey": "k", "races": races}}}


def _pk(date="2026-11-22", vid="v1-k-2026", **kw):
    return {"sha256_rules": "s", "frozen_at": "f", "rkey": "k", "date": date, "data_until": "2026-11-21", "entries_sig": "e",
            "version": {"id": vid}, "base": {"hn": 2}, "B": {"honmei": {"hn": 4}, "kill": [{"hn": 8}, {"hn": 9}]}} | kw


def test_append_keeps_old_and_adds_only_new():
    old = _hist([_race("r1", "2025-11-20", hm=3), _race("r3", UNTIL)])
    new = _hist([_race("r1", "2025-11-20", hm=7), _race("r2", "2026-11-22"), _race("r0", "2026-11-10"), _race("r4", UNTIL),
                 _race("r5", "2026-11-23", winners=False), _race("r3", UNTIL, hm=9)])
    new["rkeys"]["x"] = {"rkey": "x", "races": [_race("r9", "2026-11-22")]}
    res, added = PK.append_history(old, new, {}, lambda r, hn: _h(hn))
    rs = res["rkeys"]["k"]["races"]
    assert sorted(added) == ["r2", "r4"]                                                  # 同じ日の取り込み漏れは足す・結果の無い回は足さない
    assert rs[0]["B"]["honmei"]["hn"] == 3 and "version" not in rs[0]
    assert next(x for x in rs if x["race_id"] == "r3")["B"]["honmei"]["hn"] == 1          # 既存の回は変えない
    assert next(x for x in rs if x["race_id"] == "r2")["version"] == {"id": V0_ID, "source": "記録の規則（前日の計算なし）"}
    assert "x" not in res["rkeys"] and len(old["rkeys"]["k"]["races"]) == 2


def test_append_uses_pick_incl_scratch_and_v0():
    new = _hist([_race("r2", "2026-11-22", hm=1)])
    look = lambda r, hn: None if hn in (4, 9) else _h(hn, 2.0)                           # 4・9 は取消
    res, _ = PK.append_history(_hist([]), new, {"r2": _pk()}, look, lambda r: "e")
    x = res["rkeys"]["k"]["races"][0]
    assert x["B"]["honmei"] is None and [q["hn"] for q in x["B"]["kill"]] == [8] and x["base"]["hn"] == 2
    assert x["scratched"] == [4, 9] and x["version"] == {"id": "v1-k-2026", "source": "前日の計算"}
    res, _ = PK.append_history(_hist([]), new, {"r2": _pk(vid=V0_ID, B={"honmei": {"hn": 6}, "kill": None})}, look, lambda r: "e")
    x = res["rkeys"]["k"]["races"][0]
    assert x["B"] == {"honmei": _h(6, 2.0), "kill": None} and x["version"] == {"id": V0_ID, "source": "前日の計算"}   # V0 でも前日に出した馬
    res, _ = PK.append_history(_hist([]), new, {"r2": _pk(B={"honmei": {"hn": 6}, "kill": []})}, look, lambda r: "e")
    assert res["rkeys"]["k"]["races"][0]["B"]["kill"] == []


@pytest.mark.parametrize("bad", [{"sha256_rules": "t"}, {"rkey": "z"}, {"date": "2026-11-23"}, {"data_until": "2026-11-22"}, {"entries_sig": "old"}])
def test_append_ignores_unusable_pick(bad):
    res, _ = PK.append_history(_hist([]), _hist([_race("r2", "2026-11-22", hm=1)]), {"r2": _pk(**bad)}, lambda r, hn: _h(hn), lambda r: "e")
    x = res["rkeys"]["k"]["races"][0]
    assert x["B"]["honmei"]["hn"] == 1 and x["version"]["source"] == "記録の規則（前日の計算なし）"


def test_append_refuses_other_freeze():
    with pytest.raises(ValueError):
        PK.append_history(_hist([]), _hist([]) | {"sha256_rules": "t"}, {}, lambda r, hn: _h(hn))


def test_without_default_keeps_new_version_rows():
    from src.research.race_analysis import graded_picks as GP
    v = {"honmei": {"default": True}, "kill": {"default": True}}
    x = _race("r2", "2026-11-22")
    assert GP._without_default(x, v)["B"]["honmei"] is None
    assert GP._without_default(x | {"version": {"id": "v1-k"}}, v)["B"]["honmei"]["hn"] == 1
