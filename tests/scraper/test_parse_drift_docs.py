from src.scraper.parse_drift_docs import compare, field_rates


def _doc(i, **kw):
    d = {"race_id": f"r{i}", "weather": "晴", "entries": [{"horse_name": "a", "odds": 3.5}, {"horse_name": "b", "odds": 4.1}]}
    d.update(kw)
    return d


def test_detects_top_level_empty_and_entry_mojibake_and_empty_entries_field():
    base = field_rates([_doc(i) for i in range(20)])
    recent_docs = [_doc(i, weather="") for i in range(20)]
    for d in recent_docs:
        d["entries"] = [{"horse_name": "�", "odds": None}, {"horse_name": "�", "odds": None}]
    found = {(x["field"], x["kind"]) for x in compare(base, field_rates(recent_docs))}
    assert ("weather", "空") in found
    assert ("entries.horse_name", "文字化け") in found
    assert ("entries.odds", "空") in found


def test_stable_docs_have_no_findings():
    assert compare(field_rates([_doc(i) for i in range(20)]), field_rates([_doc(i) for i in range(30)])) == []
