import pytest

from src.scripts.data import track_speed_daily as tsd


class _Engine:
    calls: list = []
    no_result_years: set = set()

    def __init__(self, base):
        pass

    def assign_races(self, years, date_min, date_max, progress_cb=None):
        _Engine.calls.append((years[0], date_min, date_max))
        if years[0] in _Engine.no_result_years:
            raise ValueError("no races")


@pytest.fixture(autouse=True)
def _fake_engine(monkeypatch):
    import src.research.race.track_speed_engine as mod

    _Engine.calls = []
    _Engine.no_result_years = set()
    monkeypatch.setattr(mod, "TrackSpeedEngine", _Engine)


def test_single_year_passes_range_as_is():
    assert tsd.run("20261020", "20261029") == {"2026": "ok"}
    assert _Engine.calls == [("2026", "20261020", "20261029")]


def test_year_boundary_is_clipped_per_year():
    assert tsd.run("20251225", "20260105") == {"2025": "ok", "2026": "ok"}
    assert _Engine.calls == [("2025", "20251225", "20251231"), ("2026", "20260101", "20260105")]


def test_no_results_is_skipped_not_raised():
    _Engine.no_result_years = {"2026"}
    assert tsd.run("20261020", "20261029") == {"2026": "no_results"}


def test_main_rejects_reversed_range():
    with pytest.raises(SystemExit):
        tsd.main(["--from", "20261029", "--to", "20261020"])


def test_main_rejects_bad_date():
    with pytest.raises(ValueError):
        tsd.main(["--from", "2026-10-20", "--to", "20261029"])


def test_main_ok():
    assert tsd.main(["--from", "20261020", "--to", "20261029"]) == 0
