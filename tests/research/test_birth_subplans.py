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


def test_priority_list_min_impact():
    zone = [
        {"stallion_id": "a", "min_gen": "1", "n_horses": "10"},
        {"stallion_id": "b", "min_gen": "1", "n_horses": "2"},
    ]
    anc = {k: {"precision": "year", "birthday": "1970", "name": k.upper()} for k in "ab"}
    assert [x[0] for x in bs.priority_list(zone, anc, set(), min_impact=3)] == ["a"]
    assert [x[0] for x in bs.priority_list(zone, anc, set())] == ["a", "b"]


def test_all_year_only_list():
    anc = {
        "a": {"precision": "year", "birthday": "1970", "name": "A"},
        "b": {"precision": "day", "birthday": "1970-01-02", "name": "B"},
        "c": {"precision": "year", "birthday": "1890", "name": "C"},
        "d": {"precision": "year", "birthday": "1980", "name": "D"},
    }
    assert [x[0] for x in bs.all_year_only_list(anc, set())] == ["a", "d"]
    assert [x[0] for x in bs.all_year_only_list(anc, {"a"})] == ["d"]
    assert [x[0] for x in bs.all_year_only_list(anc, set(), min_year=1800)] == ["a", "c", "d"]


def test_body_foaled_date():
    assert bs.body_foaled_date("Electioneer was an American racehorse, foaled April 20, 1904 in Kentucky.") == (1904, 4, 20)
    assert bs.body_foaled_date("He was born on 3 March 1931 at Mereworth.") == (1931, 3, 3)
    assert bs.body_foaled_date("foaled in 1904 in Kentucky") is None
    assert bs.body_foaled_date("born March 1904") is None


def _row(q, en, d, nref=1, ja=None):
    r = {"h": {"value": f"http://www.wikidata.org/entity/{q}"}, "en": {"value": en}, "d": {"value": d + "T00:00:00Z"}, "nref": {"value": str(nref)}}
    if ja:
        r["ja"] = {"value": ja}
    return r


def test_match_bulk_unique_and_placeholder():
    rows = [_row("Q1", "Harlan", "1989-03-02"), _row("Q2", "Twin", "1970-05-05"), _row("Q3", "Twin", "1970-06-06"),
            _row("Q4", "Placeholder", "1980-01-01"), _row("Q5", "X", "1950-04-01", ja="ボストニアン")]
    anc = {
        "a": {"precision": "year", "birthday": "1989", "name": "Harlan\xa0(米)"},
        "b": {"precision": "year", "birthday": "1970", "name": "Twin"},
        "c": {"precision": "year", "birthday": "1980", "name": "Placeholder"},
        "d": {"precision": "year", "birthday": "1950", "name": "ボストニアン"},
        "e": {"precision": "day", "birthday": "1989-03-02", "name": "Harlan"},
    }
    got = {m["horse_id"]: m for m in bs.match_bulk(rows, anc, set())}
    assert set(got) == {"a", "d"}          # b は同名同年が 2 頭で曖昧、c は 1 月 1 日、e は日付つき
    assert got["a"]["date"] == "1989-03-02" and got["a"]["nref"] == 1
    assert bs.match_bulk(rows, anc, {"a"}) and "a" not in {m["horse_id"] for m in bs.match_bulk(rows, anc, {"a"})}


def test_upsert_log_replaces_and_appends(tmp_path):
    p = tmp_path / "log.csv"
    p.write_text(",".join(bs.LOG_COLUMNS) + "\nh1,A,no_data,1990,,,,2026-10-17,x\nh2,B,adopted,1991,3,4,u,2026-10-17,y\n", encoding="utf-8")
    bs.upsert_log(p, [["h1", "A", "adopted", 1990, 5, 6, "u1", "2026-10-19", "z"], ["h3", "C", "no_data", 1992, "", "", "", "2026-10-19", "w"]])
    rows = [l.split(",")[:3] for l in p.read_text(encoding="utf-8").splitlines()[1:]]
    assert rows == [["h1", "A", "adopted"], ["h2", "B", "adopted"], ["h3", "C", "no_data"]]
