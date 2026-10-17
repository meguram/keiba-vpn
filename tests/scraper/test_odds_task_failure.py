import pytest

from src.scraper.queue_tasks import _race_task


class _Runner:
    def __init__(self, result):
        self.result = result

    def scrape_odds(self, race_id, skip_existing=True, phase=None):
        return self.result

    scrape_pair_odds = scrape_odds


def test_empty_odds_is_a_failure_for_phase_tasks():
    with pytest.raises(RuntimeError, match="オッズを取得できなかった"):
        _race_task(_Runner(None), "202604040101", "race_odds_t60", "20261017", smart_skip=False)


def test_saved_odds_is_success():
    assert _race_task(_Runner({"entries": [1]}), "202604040101", "race_pair_odds_t15", "20261017", smart_skip=False) is False
