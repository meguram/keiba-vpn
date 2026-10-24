import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from src.scraper import new_horse_pedigree as nhp


class _Storage:
    def __init__(self, cards):
        self.cards = cards

    def load(self, cat, key):
        return self.cards.get(key)


class _Runner:
    def __init__(self, fail=()):
        self.calls, self.fail = [], set(fail)

    def scrape_horse_pedigree_5gen(self, hid, skip_existing=False):
        self.calls.append(hid)
        return None if hid in self.fail else {"horse_id": hid}


class NewHorsePedigreeTest(unittest.TestCase):
    def test_upcoming_dates(self):
        days = ["2026-10-10", "2026-10-17", "2026-10-18", "2026-10-25"]
        self.assertEqual(nhp.upcoming_dates(days, date(2026, 10, 15)), ["20261017", "20261018"])
        self.assertEqual(nhp.upcoming_dates(days, date(2026, 10, 17)), ["20261018"])
        self.assertEqual(nhp.upcoming_dates(days, date(2026, 10, 17), horizon_days=8), ["20261018", "20261025"])

    def test_fetches_only_horses_without_ped_tbl(self):
        with tempfile.TemporaryDirectory() as t:
            base = Path(t)
            rl = base / "data/page_reference/race_lists"
            rl.mkdir(parents=True)
            (rl / "20261017.json").write_text(json.dumps({"races": [{"race_id": "202608040601"}, {"race_id": "202608040602"}]}))
            have = nhp.ped_tbl_path("2023100001", base)
            have.parent.mkdir(parents=True)
            have.write_bytes(b"x" * 100)
            cards = {"202608040601": {"entries": [{"horse_id": "2023100001"}, {"horse_id": "2023100002"}]}}
            runner = _Runner(fail={"2023100003"})
            cards["202608040601"]["entries"].append({"horse_id": "2023100003"})
            out = nhp.run(["20261017"], base_dir=base, storage=_Storage(cards), runner=runner, sleep=False)
            self.assertEqual(out["races"], 2)
            self.assertEqual(out["races_without_card"], 1)
            self.assertEqual(out["horses"], 3)
            self.assertEqual(out["missing"], 2)
            self.assertEqual(sorted(runner.calls), ["2023100002", "2023100003"])
            self.assertEqual(out["fetched"], 1)
            self.assertEqual(out["failed"], ["2023100003"])
            self.assertEqual(out["status"], "partial")

    def test_max_horses_defers_rest(self):
        with tempfile.TemporaryDirectory() as t:
            base = Path(t)
            rl = base / "data/page_reference/race_lists"
            rl.mkdir(parents=True)
            (rl / "20261017.json").write_text(json.dumps([{"race_id": "202608040601"}]))
            cards = {"202608040601": {"entries": [{"horse_id": f"20231000{i:02d}"} for i in range(5)]}}
            out = nhp.run(["20261017"], base_dir=base, storage=_Storage(cards), runner=_Runner(), max_horses=2, sleep=False)
            self.assertEqual((out["fetched"], out["deferred"]), (2, 3))


if __name__ == "__main__":
    unittest.main()
