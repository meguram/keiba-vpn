"""参照用データの更新を VPS の事前ロード係へ伝える目印(cache_marker)の書き手側のテスト。"""
import pytest

from src.utils import cache_epoch, cache_marker as cm


class FakeIO:
    def __init__(self):
        self.puts: list[tuple[str, bytes]] = []

    def put_bytes(self, name, body, content_type=None):
        self.puts.append((name, body))


@pytest.fixture
def io(monkeypatch):
    fake = FakeIO()
    monkeypatch.setenv("GCS_BUCKET", "bkt")
    monkeypatch.delenv("KEIBA_CACHE_MARKER", raising=False)
    monkeypatch.delenv("KEIBA_FETCH_SHADOW", raising=False)
    monkeypatch.setattr(cm, "_io", [fake])
    monkeypatch.setattr(cm, "_last", {})
    return fake


def test_enabled_switches(monkeypatch):
    monkeypatch.setenv("GCS_BUCKET", "bkt")
    monkeypatch.delenv("KEIBA_CACHE_MARKER", raising=False)
    monkeypatch.delenv("KEIBA_FETCH_SHADOW", raising=False)
    assert cm.enabled()
    monkeypatch.setenv("KEIBA_CACHE_MARKER", "0")
    assert not cm.enabled()
    monkeypatch.delenv("KEIBA_CACHE_MARKER")
    monkeypatch.setenv("KEIBA_FETCH_SHADOW", "1")
    assert not cm.enabled()
    monkeypatch.delenv("KEIBA_FETCH_SHADOW")
    monkeypatch.setenv("GCS_BUCKET", "")
    assert not cm.enabled()


def test_mark_puts_once_per_minute_and_rejects_bad_keys(io):
    assert cm.mark("202610250811", "t")
    assert not cm.mark("202610250811", "t")           # 同じキーは 60 秒に 1 回まで
    assert not cm.mark("abc")
    assert not cm.mark("2026102508")
    assert len(io.puts) == 1
    name = io.puts[0][0]
    assert name.startswith("ops/cache_dirty/") and name.endswith("/202610250811")


def test_mark_never_raises_when_put_fails(io, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("gcs down")

    monkeypatch.setattr(io, "put_bytes", boom)
    assert cm.mark("20261025") is False


def test_mark_from_save_only_for_ui_categories(io):
    assert cm.mark_from_save("race_result", "202610250811")
    assert not cm.mark_from_save("some_internal_category", "202610250812")
    assert len(io.puts) == 1


def test_keys_from_paths_extracts_races_and_dates_and_ignores_others():
    rels = [
        "data/local/inference/predictions_v3/20261025/202610250811.parquet",
        "data/page_reference/race_lists/20261026.json",
        "data/local/meta/entries_state/202610250812.json",
        "data/local/training/foo/202610250813.parquet",        # 画面に関係しない
    ]
    races, dates = cm.keys_from_paths(rels)
    assert races == ["202610250811", "202610250812"]
    assert dates == ["20261025", "20261026"]


def test_mark_from_paths_collapses_to_dates_on_bulk_upload(io):
    rels = [f"data/local/inference/predictions_v3/20261025/20261025{i:04d}.parquet" for i in range(cm._MAX_RACE_MARKS + 1)]
    assert cm.mark_from_paths(rels) == 1                    # 開催日の目印 1 つだけ
    assert io.puts[0][0].endswith("/20261025")


def test_mark_from_paths_noop_when_disabled(io, monkeypatch):
    monkeypatch.setenv("KEIBA_CACHE_MARKER", "0")
    assert cm.mark_from_paths(["data/page_reference/race_lists/20261026.json"]) == 0
    assert io.puts == []


def test_bump_marks_all_except_for_save_reasons(io, monkeypatch, tmp_path):
    monkeypatch.setattr(cache_epoch, "_FILE", tmp_path / "e.json")
    monkeypatch.setattr(cache_epoch, "_HARD_FILE", tmp_path / "h.json")
    cache_epoch.bump("save:race_result")
    assert io.puts == []
    cache_epoch.bump("deploy")
    assert [n.rsplit("/", 1)[1] for n, _ in io.puts] == ["_ALL"]
