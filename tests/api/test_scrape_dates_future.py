"""/api/scrape-dates の picker_future_days（週間予測用）。未指定の従来動作（今日を含めない）は変えない。"""
from __future__ import annotations

import os
import unittest
import unittest.mock
from datetime import datetime, timedelta

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("GCS_BUCKET", "")
os.environ.setdefault("HORSE_NAME_INDEX_DISABLE_BOOTSTRAP", "1")

from fastapi.testclient import TestClient

from src.api import app as app_module

client = TestClient(app_module.app, raise_server_exceptions=False)


class _FakeStorage:
    gcs_enabled = False

    def __init__(self, stems: set[str]):
        self._stems = stems

    def load(self, kind: str, stem: str):
        return {"races": [{"race_id": f"{stem[:4]}0101010101"}]} if kind == "race_lists" and stem in self._stems else None


def _stem(offset: int) -> str:
    return (datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=offset)).strftime("%Y%m%d")


class TestPickerFutureDays(unittest.TestCase):
    def setUp(self):
        app_module._PICKER_SCRAPE_DATES_CACHE.clear()
        app_module._RACE_LIST_STEMS_CACHE.update(stems=None, ts=0.0)
        self.stems = {_stem(-2), _stem(0), _stem(3), _stem(8)}
        self._patches = [
            unittest.mock.patch.object(app_module, "_get_storage", lambda: _FakeStorage(self.stems)),
            unittest.mock.patch("src.scraper.monitor_future_eligible.date_has_at_least_one_jra_race", lambda stem, raw, meta: bool(raw)),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        app_module._PICKER_SCRAPE_DATES_CACHE.clear()
        app_module._RACE_LIST_STEMS_CACHE.update(stems=None, ts=0.0)

    def test_without_future_days_today_and_future_are_not_included(self):
        d = client.get("/api/scrape-dates?picker_past_days=10").json()["dates"]
        self.assertEqual(d, [_stem(-2)])

    def test_with_future_days_today_and_upcoming_are_included_in_descending_order(self):
        d = client.get("/api/scrape-dates?picker_past_days=10&picker_future_days=5").json()["dates"]
        self.assertEqual(d, [_stem(3), _stem(0), _stem(-2)])      # 8 日先は範囲外

    def test_future_days_zero_includes_only_today(self):
        d = client.get("/api/scrape-dates?picker_past_days=10&picker_future_days=0").json()["dates"]
        self.assertEqual(d, [_stem(0), _stem(-2)])

    def test_cache_key_distinguishes_future_days(self):
        a = client.get("/api/scrape-dates?picker_past_days=10").json()["dates"]
        b = client.get("/api/scrape-dates?picker_past_days=10&picker_future_days=5").json()["dates"]
        self.assertNotEqual(a, b)

    def test_future_days_out_of_range_is_rejected(self):
        self.assertEqual(client.get("/api/scrape-dates?picker_past_days=10&picker_future_days=31").status_code, 422)


if __name__ == "__main__":
    unittest.main()
