import logging
import sys
import types

import src.scraper.auto_scrape as asc


def _install(monkeypatch, assign, calls=None):
    export = types.ModuleType("src.scraper.export_tables")

    def _export(*a, **k):
        if calls is not None:
            calls.append(k.get("keys"))
        return {"skipped": True, "reason": "既にエクスポート済み"}

    export.export_category_chunked = _export
    # その日のレースはローカルの記録から作る（GCS の全件一覧は取らない）
    export.local_universe = lambda year, day, kind: {"202605010101": day.isoformat(), "202605010201": "2026-10-18"}
    storage = types.ModuleType("src.scraper.storage")
    storage.HybridStorage = lambda *a, **k: object()
    engine = types.ModuleType("src.research.race.track_speed_engine")

    class _Eng:
        def __init__(self, *a, **k):
            pass

        def assign_races(self, **k):
            assign()

    engine.TrackSpeedEngine = _Eng
    for name, mod in (("src.scraper.export_tables", export), ("src.scraper.storage", storage),
                      ("src.research.race.track_speed_engine", engine)):
        monkeypatch.setitem(sys.modules, name, mod)


def test_no_confirmed_result_is_not_an_error(monkeypatch, caplog):
    def boom():
        raise ValueError("スコア付きレースが0件")

    _install(monkeypatch, boom)
    with caplog.at_level(logging.INFO, logger=asc.logger.name):
        asc._trigger_track_speed_for_date("2026-10-17")
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("見送る" in r.getMessage() for r in caplog.records)


def test_other_errors_are_still_errors(monkeypatch, caplog):
    def boom():
        raise RuntimeError("壊れた")

    _install(monkeypatch, boom)
    with caplog.at_level(logging.INFO, logger=asc.logger.name):
        asc._trigger_track_speed_for_date("2026-10-17")
    assert [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_only_that_days_races_are_passed_by_name(monkeypatch):
    calls = []
    _install(monkeypatch, lambda: None, calls)
    asc._trigger_track_speed_for_date("2026-10-17")
    assert calls == [["202605010101"]]           # keys を渡す（keys=None だと GCS の全件一覧になる）
