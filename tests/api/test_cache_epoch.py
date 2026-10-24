import os


def _fresh(tmp_path, monkeypatch):
    from src.utils import cache_epoch as ce

    monkeypatch.setattr(ce, "_FILE", tmp_path / "epoch.json")
    monkeypatch.setattr(ce, "_clearers", {})
    monkeypatch.setattr(ce, "_seen", {"mtime": -1.0, "checked": 0.0, "bumped": 0.0})
    return ce


def test_sync_clears_after_bump(tmp_path, monkeypatch):
    ce = _fresh(tmp_path, monkeypatch)
    ce.bump("init")
    hit = []
    ce.register("t", lambda: hit.append(1))
    assert ce.sync(force=True) is False          # 初回は基準を覚えるだけ
    ce.bump("x")
    os.utime(tmp_path / "epoch.json", (1, 2e9))   # mtime の粒度に依存しないよう確実に変える
    assert ce.sync(force=True) is True and hit == [1]
    assert ce.sync(force=True) is False


def test_bump_for_save_only_ui_categories_and_throttled(tmp_path, monkeypatch):
    ce = _fresh(tmp_path, monkeypatch)
    ce.bump_for_save("horse_pedigree_5gen")
    assert not (tmp_path / "epoch.json").exists()
    ce.bump_for_save("race_odds")
    assert (tmp_path / "epoch.json").exists()
    (tmp_path / "epoch.json").unlink()
    ce.bump_for_save("race_odds")                 # 20 秒以内は書かない
    assert not (tmp_path / "epoch.json").exists()


def test_failing_clearer_does_not_stop_others(tmp_path, monkeypatch):
    ce = _fresh(tmp_path, monkeypatch)
    ok = []
    ce.register("bad", lambda: 1 / 0)
    ce.register("good", lambda: ok.append(1))
    assert "bad" in ce.clear_all() and ok == [1]


def test_value_cache_is_cleared_by_epoch(tmp_path, monkeypatch):
    from src.api.v1 import delegates as D
    from src.utils import cache_epoch

    D._VALUE_CACHE[("x",)] = (0.0, {})
    cache_epoch.clear_all()
    assert not D._VALUE_CACHE


def test_past_date_ttl_is_long():
    from src.api.v1 import delegates as D

    assert D._value_ttl("20200101") == D._VALUE_TTL_PAST_SEC
    assert D._value_ttl("29991231") == D._VALUE_TTL_SEC
