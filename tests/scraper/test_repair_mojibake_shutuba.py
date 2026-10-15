from src.scripts.maintenance.repair_mojibake_shutuba import header_ok, is_mojibake, merge_weights


def test_detects_mojibake_in_name_or_entries():
    assert is_mojibake({"race_name": "2��", "entries": []})
    assert is_mojibake({"race_name": "x", "entries": [{"horse_name": "��"}]})
    assert not is_mojibake({"race_name": "2歳新馬", "entries": [{"horse_name": "ディープ"}]})
    assert not is_mojibake(None)


def test_header_ok_requires_course_date_and_clean_text():
    good = {"date": "2026-09-27", "surface": "芝", "distance": 1200, "race_name": "2歳新馬", "entries": [{"horse_name": "a"}]}
    assert header_ok(good)
    assert not header_ok({**good, "date": ""})
    assert not header_ok({**good, "distance": 0})
    assert not header_ok({**good, "race_name": "�"})


def test_merge_keeps_old_weight_only_when_new_has_none_and_same_horse():
    old = {"entries": [{"horse_number": 1, "horse_id": "a", "weight": 440, "weight_change": 2}, {"horse_number": 2, "horse_id": "b", "weight": 480, "weight_change": 0}]}
    new = {"entries": [{"horse_number": 1, "horse_id": "a", "weight": None}, {"horse_number": 2, "horse_id": "b", "weight": 500, "weight_change": 4},
                       {"horse_number": 3, "horse_id": "c", "weight": None}]}
    out = merge_weights(new, old)
    e = out["entries"]
    assert (e[0]["weight"], e[0]["weight_change"]) == (440, 2)
    assert (e[1]["weight"], e[1]["weight_change"]) == (500, 4)    # 新しい値を優先
    assert e[2]["weight"] is None                                  # 旧版に無い馬は触らない
    assert "mojibake_repaired_at" in out["_meta"]
