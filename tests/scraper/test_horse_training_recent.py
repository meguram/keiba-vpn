"""出走馬の調教ページの取り直し（最新 1 ページを保存済みの記録に合わせる）のテスト。"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from src.scraper import horse_training_recent as htr


def _e(date, race_date, tag=""):
    return {"date": date, "course": "栗坂", "time_raw": tag, "race_info": f"{race_date.replace('-', '/')}  東京11R 結果 ： 着"}


class MergeTest(unittest.TestCase):
    def test_replaces_same_race_date_and_keeps_older(self):
        existing = {"horse_id": "H", "total_items": 5, "pages_fetched": 3, "entries": [_e("2026-09-02", "2026-09-05", "old1"), _e("2026-08-01", "2026-08-08", "old2")]}
        fresh = {"horse_id": "H", "total_items": 6, "entries": [_e("2026-10-14", "2026-10-17", "new1"), _e("2026-09-02", "2026-09-05", "new2")]}
        m = htr.merge_recent(existing, fresh)
        tags = [e["time_raw"] for e in m["entries"]]
        self.assertEqual(tags, ["new1", "new2", "old2"])                   # 9/5 の行は新しいものに入れ替え、8/8 は残る
        self.assertEqual((m["total_items"], m["pages_fetched"]), (6, 3))

    def test_no_existing(self):
        m = htr.merge_recent(None, {"horse_id": "H", "total_items": 1, "entries": [_e("2026-10-14", "2026-10-17")]})
        self.assertEqual(len(m["entries"]), 1)
        self.assertEqual(m["pages_fetched"], 1)


class RefreshTest(unittest.TestCase):
    def _runner(self, entries, html="<table class='race_table_01'></table>"):
        r = mock.MagicMock()
        r.HORSE_TRAINING_URL = "https://x/{horse_id}"
        r.client.fetch.return_value = html
        r.training_parser.parse.return_value = {"entries": entries, "total_items": len(entries)}
        r.storage.load.return_value = None
        return r

    def test_saves_when_entries(self):
        r = self._runner([_e("2026-10-14", "2026-10-17")])
        out = htr.refresh_one(r, "H")
        self.assertEqual(out["status"], "saved")
        r._persist_horse_training.assert_called_once()

    def test_unreadable_page_is_not_saved(self):
        r = self._runner([])
        out = htr.refresh_one(r, "H")
        self.assertEqual(out["status"], "unreadable")                      # 表があるのに 0 件 → 保存しない
        r._persist_horse_training.assert_not_called()

    def test_empty_page_is_not_saved(self):
        r = self._runner([], html="<html></html>")
        self.assertEqual(htr.refresh_one(r, "H")["status"], "empty")
        r._persist_horse_training.assert_not_called()

    def test_one_failure_does_not_stop_others(self):
        r = self._runner([_e("2026-10-14", "2026-10-17")])
        calls = {"n": 0}

        def fetch(url, use_cache=False):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return "<table class='race_table_01'></table>"

        r.client.fetch.side_effect = fetch
        with mock.patch.object(htr.time, "sleep", lambda *_: None):
            st = htr.refresh_for_horses(r, ["A", "B"])
        self.assertEqual((st["error"], st["saved"]), (1, 1))


class EveSwitchTest(unittest.TestCase):
    def test_disabled_by_default(self):
        from src.scraper import auto_scrape_queue as aq

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("KEIBA_EVE_HORSE_TRAINING", None)
            self.assertEqual(aq._eve_horse_training(mock.MagicMock(), ["r1"]), {"enabled": False})


if __name__ == "__main__":
    unittest.main()
