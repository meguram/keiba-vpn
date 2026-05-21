"""race_result_availability のユニットテスト。"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from src.utils.race_result_availability import (
    _blob_has_entries,
    batch_race_result_status,
    race_result_status,
)


class TestBlobHasEntries(unittest.TestCase):
    def test_entries(self):
        self.assertTrue(_blob_has_entries({"entries": [{"rank": 1}]}))

    def test_empty(self):
        self.assertFalse(_blob_has_entries({}))
        self.assertFalse(_blob_has_entries(None))


class TestRaceResultStatus(unittest.TestCase):
    def test_confirmed(self):
        storage = MagicMock()
        storage.exists.side_effect = lambda cat, rid: cat == "race_result"
        storage.load.return_value = {"entries": [{"rank": 1}]}
        st = race_result_status(storage, "202603010301")
        self.assertTrue(st["viewable"])
        self.assertEqual(st["kind"], "confirmed")
        self.assertEqual(st["result_view_url"], "/race/202603010301")

    def test_flash_only(self):
        storage = MagicMock()
        storage.exists.side_effect = lambda cat, rid: cat == "race_result_on_time"
        storage.load.return_value = {"entries": [{"horse_number": 1}]}
        st = race_result_status(storage, "202603010301")
        self.assertTrue(st["viewable"])
        self.assertEqual(st["kind"], "flash")


if __name__ == "__main__":
    unittest.main()
