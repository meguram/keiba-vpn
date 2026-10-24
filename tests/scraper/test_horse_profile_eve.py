from src.scraper import horse_profile_eve as hp


class FakeStorage:
    def __init__(self, result=None, profile=None):
        self.result, self.profile = result or {}, profile or {}

    def load(self, cat, key, **kw):
        return {"horse_result": self.result, "horse_profile": self.profile}.get(cat, {}).get(key)


class FakeRunner:
    def __init__(self, storage, fail=()):
        self.storage, self.fail, self.calls = storage, set(fail), []

    def scrape_horse(self, hid, skip_existing=True, with_history=True, skip_pedigree=False):
        self.calls.append((hid, skip_existing, with_history, skip_pedigree))
        if hid in self.fail:
            raise RuntimeError("boom")
        self.storage.result[hid] = {"horse_id": hid, "owner": "X"}
        return self.storage.result[hid]


def test_classify_and_never_refetch_existing(monkeypatch):
    st = FakeStorage(result={"a": {"owner": "A"}, "b": {"owner": ""}}, profile={"b": {"owner": "B"}, "c": {"owner": ""}})
    st.result["e"] = {"owner": ""}                                  # horse_result はあるが owner が空（取得しない）
    got = hp.classify(st, ["a", "b", "c", "d", "e"])
    assert got == {"ok": ["a", "b"], "no_owner": ["e"], "missing": ["c", "d"]}


def test_fetch_missing_only_missing_and_continue_on_error(monkeypatch):
    monkeypatch.setenv("HORSE_PROFILE_EVE_MIN_SEC", "0")
    monkeypatch.setenv("HORSE_PROFILE_EVE_MAX_SEC", "0")
    st = FakeStorage(result={"a": {"owner": "A"}})
    runner = FakeRunner(st, fail={"x"})
    out = hp.fetch_missing(runner, ["d", "x"])
    assert out["saved"] == 1 and out["error"] == 1 and not out["aborted"]
    assert runner.calls[0] == ("d", True, True, True)       # 既存は上書きしない(skip_existing=True)


def test_owner_table_marks_missing():
    st = FakeStorage(result={"a": {"owner": "A"}})
    t = hp.owner_table(st, ["a", "z"])
    assert t.loc[t.horse_id == "a", "owner"].iloc[0] == "A" and t.loc[t.horse_id == "z", "owner_missing"].iloc[0] == 1


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("KEIBA_EVE_HORSE_PROFILE", raising=False)
    from src.scraper.auto_scrape_queue import _eve_horse_profile

    assert _eve_horse_profile(object(), []) == {"enabled": False}
