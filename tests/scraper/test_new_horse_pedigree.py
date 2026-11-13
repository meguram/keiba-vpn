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
    def __init__(self, fail=(), existing=()):
        self.calls, self.fail, self.existing = [], set(fail), set(existing)
        self._last_pedigree_5gen_skip = False

    def scrape_horse_pedigree_5gen(self, hid, skip_existing=False):
        self.calls.append(hid)
        self._last_pedigree_5gen_skip = skip_existing and hid in self.existing      # GCS に 5 世代表があり、通信せずに飛ばした
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


    def test_skipped_existing_do_not_use_up_the_cap(self):
        """VPS には手元の ped_tbl が無く全馬が missing に見える。GCS にあって飛ばした馬は上限に数えない。"""
        with tempfile.TemporaryDirectory() as t:
            base = Path(t)
            rl = base / "data/page_reference/race_lists"
            rl.mkdir(parents=True)
            (rl / "20261017.json").write_text(json.dumps([{"race_id": "202608040601"}]))
            ids = [f"20231000{i:02d}" for i in range(6)]
            cards = {"202608040601": {"entries": [{"horse_id": h} for h in ids]}}
            runner = _Runner(existing=set(ids[2:]))                   # 4 頭は取得済み（新しい ID 側）、2 頭が本当の新馬
            out = nhp.run(["20261017"], base_dir=base, storage=_Storage(cards), runner=runner, max_horses=2, sleep=False)
            self.assertEqual((out["fetched"], out["skipped_existing"], out["deferred"]), (2, 4, 0))
            runner = _Runner(existing=set(ids))
            out = nhp.run(["20261017"], base_dir=base, storage=_Storage(cards), runner=runner, max_checks=3, sleep=False)
            self.assertEqual((len(runner.calls), out["deferred"]), (3, 3))               # 確かめる件数（GCS の読み）にも上限
            self.assertEqual(runner.calls, sorted(ids, reverse=True)[:3])               # 新しい ID（新馬）から確かめる

    def test_skip_path_with_network_counts_toward_cap(self):
        """飛ばす経路でも祖先・種付けのページを取ったら（client の通信回数が増えたら）上限に数える。"""
        class _Client:
            _request_count = 0

        class _NetRunner(_Runner):
            def __init__(self):
                super().__init__(existing=set(ids))
                self.client = _Client()

            def scrape_horse_pedigree_5gen(self, hid, skip_existing=False):
                self.client._request_count += 1
                return super().scrape_horse_pedigree_5gen(hid, skip_existing)

        ids = [f"20231000{i:02d}" for i in range(5)]
        with tempfile.TemporaryDirectory() as t:
            base = Path(t)
            rl = base / "data/page_reference/race_lists"
            rl.mkdir(parents=True)
            (rl / "20261017.json").write_text(json.dumps([{"race_id": "202608040601"}]))
            cards = {"202608040601": {"entries": [{"horse_id": h} for h in ids]}}
            runner = _NetRunner()
            out = nhp.run(["20261017"], base_dir=base, storage=_Storage(cards), runner=runner, max_horses=2, sleep=False)
            self.assertEqual((len(runner.calls), out["deferred"]), (2, 3))


if __name__ == "__main__":
    unittest.main()
