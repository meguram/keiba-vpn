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
