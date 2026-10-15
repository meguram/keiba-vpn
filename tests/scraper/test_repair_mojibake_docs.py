from src.scripts.maintenance.repair_mojibake_docs import acceptable, has_mojibake


def test_has_mojibake_searches_nested_but_ignores_meta():
    assert has_mojibake({"entries": [{"comment": "ok"}, {"comment": "��"}]})
    assert has_mojibake({"a": {"b": ["x", "�"]}})
    assert not has_mojibake({"entries": [{"comment": "良い"}], "_meta": {"note": "�"}})
    assert not has_mojibake({"n": 3, "x": None})


def test_acceptable_requires_clean_and_not_fewer_entries():
    old = {"entries": [{}, {}, {}]}
    assert acceptable({"entries": [{}, {}, {}]}, old)
    assert acceptable({"entries": [{}] * 4}, old)
    assert not acceptable({"entries": [{}] * 2}, old)
    assert not acceptable({"entries": [{"a": "�"}] * 3}, old)
    assert not acceptable(None, old)


def test_acceptable_rejects_pages_that_lost_fields_even_with_same_rows():
    old = {"entries": [{"a": "�x", "training_date": "2026/07/22", "lap": "72.0", "comment": "�"}] * 4}
    lost = {"entries": [{"a": "x", "training_date": "", "lap": "", "comment": "ok"}] * 4}
    kept = {"entries": [{"a": "x", "training_date": "2026/07/22", "lap": "72.0", "comment": "ok"}] * 4}
    assert not acceptable(lost, old)
    assert acceptable(kept, old)
