from src.research.pedigree import birth_subplans as bs


def test_split_names():
    assert bs.split_names("ノーザンテーストNorthern Taste(加)") == ("Northern Taste", "ノーザンテースト")
    assert bs.split_names("Halo (米)") == ("Halo", "")
    assert bs.split_names("サンデーサイレンス") == ("", "サンデーサイレンス")


def _wd(y=1979, m=5, d=17, p=11, ref=True):
    return {"q": "Q1", "year": y, "month": m, "day": d, "precision": p, "has_ref": ref}


def test_decide_wikidata_with_reference():
    st, v, _ = bs.decide(_wd(), None, None, 1979)
    assert (st, v) == ("adopted", (1979, 5, 17))


def test_decide_wikidata_needs_lead_agreement_without_reference():
    assert bs.decide(_wd(ref=False), None, None, 1979)[0] == "candidate"
    assert bs.decide(_wd(ref=False), (1979, 5, 17), None, 1979)[0] == "adopted"
    # 日付が食い違うときは採用せず保留（参照の有無によらない）
    assert bs.decide(_wd(ref=False), (1979, 5, 18), None, 1979)[0] == "candidate"
    assert bs.decide(_wd(ref=True), None, (1979, 5, 18), 1979)[0] == "candidate"
    # 冒頭文が月までしかないときは不一致扱いにしない
    assert bs.decide(_wd(ref=True), (1979, 5, None), None, 1979)[0] == "adopted"


def test_decide_rejects_placeholder_and_year_precision():
    assert bs.decide(_wd(m=1, d=1), None, None, 1979)[0] == "none"
    assert bs.decide(_wd(p=9, m=1, d=1), None, None, 1979)[0] == "none"
    assert bs.decide(_wd(y=1980), None, None, 1979)[0] == "none"


def test_decide_rejects_jan1_placeholder_in_lead():
    assert bs.decide(None, (1990, 1, 1), None, 1990)[0] == "none"
    assert bs.decide(None, None, (1990, 1, 1), 1990)[0] == "none"
    assert bs.decide(None, (1990, 1, 2), None, 1990)[0] == "adopted"


def test_decide_lead_only_and_year_mismatch():
    assert bs.decide(None, (1907, 1, None), None, 1907) == ("adopted", (1907, 1, None), "英語版 Wikipedia 冒頭文")
    assert bs.decide(None, (1908, 1, None), None, 1907)[0] == "none"
    assert bs.decide(None, (1907, None, None), None, 1907)[0] == "none"


def test_priority_list():
    zone = [
        {"stallion_id": "a", "min_gen": "2", "n_horses": "10"},
        {"stallion_id": "a", "min_gen": "5", "n_horses": "5"},
        {"stallion_id": "b", "min_gen": "4", "n_horses": "99"},
        {"stallion_id": "c", "min_gen": "1", "n_horses": "3"},
        {"stallion_id": "d", "min_gen": "1", "n_horses": "50"},
    ]
    anc = {
        "a": {"precision": "year", "birthday": "1970", "name": "A"},
        "b": {"precision": "year", "birthday": "1970", "name": "B"},
        "c": {"precision": "day", "birthday": "1970-01-02", "name": "C"},
        "d": {"precision": "year", "birthday": "1890", "name": "D"},
    }
    assert bs.priority_list(zone, anc, set()) == [("a", "A", 1970, 10)]
    assert bs.priority_list(zone, anc, {"a"}) == []
