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


# ── 標本をローカルの表から選ぶ（GCS の一覧を取らない。2026-11-13）──────────────────────────────

class _NoListStorage:
    def __init__(self):
        self.loaded = []

    def batch_list_blobs(self, *a):
        raise AssertionError("一覧を取ってはいけない")

    def load(self, cat, key, bypass_cache=False):
        self.loaded.append(key)
        return {"race_id": key, "race_name": "x", "entries": [{"horse_id": "h", "odds": 1.0}]}


def _tables(tmp_path, year, n):
    import pandas as pd
    d = tmp_path / str(year)
    d.mkdir(parents=True)
    rows = [{"race_id": f"{year}0501{i // 12 + 1:02d}{i % 12 + 1:02d}", "date": f"{year}-01-{i // 12 + 1:02d}"} for i in range(n)]
    pd.DataFrame(rows).to_parquet(d / "race_result_flat.parquet")
    return rows


def test_recent_sample_is_newest_races_without_listing(tmp_path):
    import random
    from src.scraper import parse_drift_docs as P

    rows = _tables(tmp_path, 2026, 120)
    ids = P.local_sample_ids("race_odds", "2026", 10, True, random.Random(0), tmp_path)
    newest = sorted(r["race_id"] for r in rows)[-30:]
    assert len(ids) == 10 and set(ids) <= set(newest)


def test_baseline_is_saved_and_reused(tmp_path, monkeypatch):
    import random
    from src.scraper import date_coverage as dc
    from src.scraper import parse_drift_docs as P

    monkeypatch.setattr(dc, "NOT_AVAILABLE_DIR", tmp_path / "meta" / "not_available")
    _tables(tmp_path, 2025, 60)
    st = _NoListStorage()
    rates, n = P.baseline_rates(st, "race_odds", "2025", 10, random.Random(0), tmp_path)
    assert n == 20 and len(st.loaded) == 20 and rates
    st2 = _NoListStorage()
    rates2, n2 = P.baseline_rates(st2, "race_odds", "2025", 10, random.Random(0), tmp_path)
    assert st2.loaded == [] and n2 == 20 and rates2 == rates           # 2 回目は読まない
