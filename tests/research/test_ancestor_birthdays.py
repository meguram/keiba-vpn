import pytest

from src.research.pedigree import ancestor_birthdays as ab


def _need(*ids):
    return [{"horse_id": i, "name": f"h{i}", "min_gen": "1", "n_pairs": "10"} for i in ids]


def test_parse_raw():
    assert ab.parse_raw("1961年5月27日") == (1961, 5, 27)
    assert ab.parse_raw("1935年") == (1935, None, None)
    assert ab.parse_raw("1935年3月") == (1935, 3, None)
    assert ab.parse_raw("") is None


def test_merge_precedence_and_unknown_reasons():
    nk = [
        {"horse_id": "a", "status": "ok", "birthday_raw": "1935年"},
        {"horse_id": "b", "status": "ok", "birthday_raw": "1927年"},
        {"horse_id": "c", "status": "ok", "birthday_raw": "1920年"},
        {"horse_id": "d", "status": "no_birthday"},
        {"horse_id": "e", "status": "error"},
        {"horse_id": "g", "status": "ok", "birthday_raw": "1961年5月27日"},
    ]
    wiki = [
        {"horse_id": "a", "status": "day", "wiki_year": 1935, "month": 1, "day": 24, "url": "u-a"},
        {"horse_id": "b", "status": "year_only", "wiki_year": 1927, "month": None, "day": None, "url": "u-b"},
        {"horse_id": "c", "status": "year_mismatch", "wiki_year": 1921, "month": 4, "day": 4, "url": "u-c"},
    ]
    manual = {"d": {"y": 1900, "mo": 2, "d": None, "url": "m", "name": ""}}
    table, unknown = ab.merge(_need("a", "b", "c", "d", "e", "f", "g"), nk, wiki, manual)
    t = {r["horse_id"]: r for r in table}
    assert t["a"]["birthday"] == "1935-01-24" and t["a"]["source"] == "netkeiba+wikipedia"
    assert t["b"]["birthday"] == "1927" and t["b"]["precision"] == "year"
    assert t["c"]["birthday"] == "1920" and "1921" in t["c"]["wiki_conflict"]
    assert t["d"]["birthday"] == "1900-02" and t["d"]["source"] == "manual"
    assert t["g"]["precision"] == "day"
    assert {u["horse_id"]: u["reason"] for u in unknown} == {"e": "fetch_error", "f": "not_fetched"}


def test_manual_day_without_month_rejected(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text(",".join(ab.MANUAL_COLUMNS) + "\nx,n,1950,,12,u,\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ab.load_manual_births(p)
