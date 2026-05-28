"""race.netkeiba.com RaceTable01（全着順）を RaceResultOnTimeParser が取り込めることの検証。"""

from __future__ import annotations

import unittest
from pathlib import Path

from src.scraper.parsers import RaceResultOnTimeParser
from src.scraper.race_result_on_time_normalize import normalize_race_live_result

_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "race_netkeiba_result_202605020302_racetable01_wrap_utf8.html"
)


class TestRaceResultOnTimeRaceTable01(unittest.TestCase):
    def test_parse_202605020302_snippet_14_entries(self):
        html = _FIXTURE.read_text(encoding="utf-8")
        p = RaceResultOnTimeParser()
        d = p.parse(html, race_id="202605020302")
        self.assertEqual(d.get("race_id"), "202605020302")
        entries = d.get("entries") or []
        self.assertEqual(len(entries), 14, "RaceTable01 の出走頭数と一致すること")
        e0 = entries[0]
        self.assertEqual(e0.get("finish_position"), 1)
        self.assertEqual(e0.get("horse_number"), 3)
        self.assertTrue(e0.get("horse_id"))
        self.assertTrue(e0.get("passing_order") or e0.get("finish_time"))

    def test_normalize_archive_sets_profile_and_nulls(self):
        html = _FIXTURE.read_text(encoding="utf-8")
        d = RaceResultOnTimeParser().parse(html, race_id="202605020302")
        normalize_race_live_result(d, profile="race_live_archive")
        self.assertEqual(d["_meta"]["result_schema_profile"], "race_live_archive")
        for ent in d["entries"]:
            self.assertIn("trainer_id", ent)


if __name__ == "__main__":
    unittest.main()
