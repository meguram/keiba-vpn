"""HybridStorage の load キャッシュの上限（件数・量・期限切れの掃除）。2026-11-15 のメモリの張り付きの再発防止。外部接続なし。"""

import pytest

from src.scraper import storage as S


@pytest.fixture
def make(monkeypatch, tmp_path):
    monkeypatch.setenv("KEIBA_ENV", "prod")
    monkeypatch.setenv("GCS_BUCKET", "")

    def _make(max_mb="0", max_entries="8000"):
        monkeypatch.setenv("LOAD_CACHE_MAX_MB", max_mb)
        monkeypatch.setenv("LOAD_CACHE_MAX_ENTRIES", max_entries)
        return S.HybridStorage(base_dir=str(tmp_path))

    return _make


def test_byte_limit_evicts_oldest_first(make):
    st = make(max_mb="1")
    for i in range(5):
        st._cache_put(f"race_result/{i}", {"i": i}, nbytes=300 * 1024)
    assert st.load_cache_bytes() <= 1024 * 1024
    assert list(st._load_cache) == ["race_result/2", "race_result/3", "race_result/4"]


def test_reput_moves_to_end_and_keeps_total(make):
    st = make(max_mb="1")
    st._cache_put("a/1", {}, nbytes=400 * 1024)
    st._cache_put("a/2", {}, nbytes=400 * 1024)
    st._cache_put("a/1", {}, nbytes=400 * 1024)            # 入れ直しは末尾へ（二重に数えない）
    assert st.load_cache_bytes() == 800 * 1024 and list(st._load_cache) == ["a/2", "a/1"]
    st._cache_put("a/3", {}, nbytes=400 * 1024)
    assert list(st._load_cache) == ["a/1", "a/3"]


def test_entry_limit_still_applies(make):
    st = make(max_entries="3")
    for i in range(5):
        st._cache_put(f"x/{i}", {"i": i})
    assert list(st._load_cache) == ["x/2", "x/3", "x/4"]


def test_expired_entries_are_swept(make, monkeypatch):
    st = make(max_mb="10")
    st._cache_put("race_result/old", {"a": 1}, nbytes=1000)
    k, (ts, d) = next(iter(st._load_cache.items()))
    st._load_cache[k] = (ts - st._mem_cache_ttl - 1, d)       # 期限切れにする
    st._load_cache_last_sweep -= 61
    st._cache_put("race_result/new", {"b": 1}, nbytes=1000)
    assert list(st._load_cache) == ["race_result/new"] and st.load_cache_bytes() == 1000


def test_invalidate_and_external_clear_keep_counts_right(make):
    st = make(max_mb="1")
    st._cache_put("c/1", {}, nbytes=500)
    st._cache_put("c/2", {}, nbytes=500)
    st.invalidate_load_cache("c", "1")
    assert st.load_cache_bytes() == 500
    st.invalidate_load_cache("c")
    assert st.load_cache_bytes() == 0
    st._cache_put("d/1", {}, nbytes=900 * 1024)
    st._load_cache.clear()                                  # 外から直接消された（src/scraper/export_tables.py など）
    st._cache_put("d/2", {}, nbytes=900 * 1024)
    assert list(st._load_cache) == ["d/2"] and st.load_cache_bytes() == 900 * 1024


def test_no_byte_limit_by_default(make):
    st = make()
    assert st._mem_cache_max_bytes == 0
    st._cache_put("e/1", {"x": "y" * 10000})
    assert st.load_cache_bytes() == 0                      # 量の上限が無ければ大きさを数えない（従来どおりの速さ）


def test_approx_mem_matches_measured_ratio():
    assert S._approx_mem(1000) == 2500
    assert S._approx_obj_mem({"a": "x" * 1000}) > 1000


def test_race_detail_cache_drops_expired(monkeypatch):
    monkeypatch.setenv("KEIBA_ENV", "dev")
    from src.api import app as A

    A._race_detail_cache.clear()
    now = A._time.time()
    A._race_detail_cache["old"] = (now - A._RACE_DETAIL_TTL - 1, {"x": 1})
    A._race_detail_cache_set("new", (now, {"y": 1}))
    assert list(A._race_detail_cache) == ["new"]
    A._race_detail_cache.clear()


def test_cloudrun_defaults_limit_caches(monkeypatch):
    monkeypatch.setenv("KEIBA_CLOUDRUN_LAZY", "1")
    from src.api import cloudrun_entry as E

    d = E._DEFAULTS
    assert 0 < float(d["LOAD_CACHE_MAX_MB"]) <= 512
    assert int(d["RACE_DETAIL_CACHE_MAX"]) <= 500 and float(d["KEIBA_PRERENDER_MAX_MB"]) <= 150
