"""race_day_schedule 合成・スナップショット優先。"""

from __future__ import annotations

import unittest
from datetime import datetime

from src.scraper.auto_scrape_queue import _fetch_race_schedule_storage
from src.scraper.race_day_schedule import (
    schedule_payload_to_runtime_list,
    synthesize_race_day_schedule_payload,
)


class _MemStorage:
    def __init__(self, data: dict[tuple[str, str], dict | None]) -> None:
        self._data = data

    def load(self, category: str, key: str) -> dict | None:
        return self._data.get((category, key))


class TestRaceDaySchedule(unittest.TestCase):
    def test_synthesize_from_lists_and_shutuba(self) -> None:
        st = _MemStorage(
            {
                ("race_lists", "20230101"): {
                    "races": [
                        {"race_id": "202301010101", "venue": "中山", "round": 1, "race_name": "１Ｒ"},
                    ]
                },
                ("race_shutuba", "202301010101"): {"start_time": "10:05"},
            }
        )
        pl = synthesize_race_day_schedule_payload(st, "20230101")
        self.assertEqual(pl["date_fmt"], "20230101")
        self.assertEqual(len(pl["slots"]), 1)
        self.assertEqual(pl["slots"][0]["time_source"], "shutuba")
        rt = schedule_payload_to_runtime_list(pl)
        self.assertEqual(len(rt), 1)
        self.assertEqual(rt[0]["race_id"], "202301010101")
        self.assertEqual(rt[0]["start_time_str"], "10:05")
        self.assertIsInstance(rt[0]["post_time"], datetime)

    def test_fetch_prefers_race_day_schedule_snapshot(self) -> None:
        snap = {
            "date_fmt": "20230101",
            "slots": [
                {
                    "race_id": "202301010101",
                    "post_time_iso": "2023-01-01T11:15:00+09:00",
                    "start_time_str": "11:15",
                    "venue": "東",
                    "round": 2,
                    "race_name": "２Ｒ",
                }
            ],
        }
        st = _MemStorage({("race_day_schedule", "20230101"): snap})
        out = _fetch_race_schedule_storage(st, "20230101")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["race_id"], "202301010101")
        self.assertEqual(out[0]["start_time_str"], "11:15")
        self.assertEqual(
            out[0]["post_time"].strftime("%Y-%m-%dT%H:%M:%S"),
            "2023-01-01T11:15:00",
        )


if __name__ == "__main__":
    unittest.main()
