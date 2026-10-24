"""調教データの被覆の保険（training_coverage）と、穴埋めスクリプトの対象の数え方。"""

from __future__ import annotations

import unittest
from unittest import mock

from src.pipeline.inference import training_coverage as tc
from src.scripts.scraping import backfill_horse_training_recent as bf


def _rec(*race_dates: str) -> dict:
    entries = []
    for d in race_dates:
        entries.append({
            "race_info": f"{d.replace('-', '/')} 東京 1R テスト",
            "date": "2026-10-14",
            "course": "栗坂",
            "track_condition": "良",
            "lap_times": [52.0, 37.5, 24.0, 12.0],
            "position": "3",
        })
    return {"horse_id": "x", "total_items": len(entries), "pages_fetched": 1, "entries": entries}


class LevelTest(unittest.TestCase):
    def test_levels(self):
        self.assertEqual(tc.level_of(1.0), "ok")
        self.assertEqual(tc.level_of(0.9), "ok")
        self.assertEqual(tc.level_of(0.7), "low")
        self.assertEqual(tc.level_of(0.1), "poor")

    def test_summarize_empty_and_mixed(self):
        self.assertEqual(tc.summarize({})["level"], "unknown")
        s = tc.summarize({"a": (True, True), "b": (True, False), "c": (False, False), "d": (True, True)})
        self.assertEqual((s["n"], s["n_with_record"], s["n_with_block"], s["rate"], s["level"]), (4, 3, 2, 0.5, "low"))
        self.assertTrue(s["note"])


class AttachTest(unittest.TestCase):
    def test_disabled_by_default(self):
        resp = {"predictions": [{"horse_id": "1"}]}
        with mock.patch.dict("os.environ", {}, clear=False):
            self.assertIs(tc.attach_training_coverage(resp, "2026-10-18"), resp)

    def test_attach_with_storage(self):
        class St:
            def load(self, cat, h):
                return {"1": _rec("2026-10-18")}.get(h)

        resp = {"predictions": [{"horse_id": "1"}, {"horse_id": "2"}]}
        with mock.patch.dict("os.environ", {"KEIBA_TRAINING_COVERAGE": "1"}):
            out = tc.attach_training_coverage(resp, "20261018", St())
        cov = out["training_coverage"]
        self.assertEqual(cov["n"], 2)
        self.assertEqual(cov["n_with_block"], 1)
        self.assertEqual(cov["n_with_record"], 1)
        self.assertNotIn("training_coverage", resp)  # 元の応答は変えない


class BackfillTargetTest(unittest.TestCase):
    def test_missing_counts(self):
        starts = [("a", "2026-10-10"), ("a", "2026-09-01"), ("b", "2026-10-10")]
        fake = {"a": _rec("2026-10-10"), "b": None}
        with mock.patch("src.scripts.data.drift_coverage.block_dates", return_value={"a": {"2026-10-10"}, "b": set()}):
            out = bf.missing_counts(starts, load_many=lambda hs: fake)
        self.assertEqual(out, {"a": 1, "b": 1})

    def test_busy_pattern_ignores_itself_but_catches_other_scrapers(self):
        import re

        pat = re.compile(bf.OTHER_SCRAPERS)
        self.assertIsNone(pat.search("/opt/conda/bin/python3 -m src.scripts.scraping.backfill_horse_training_recent --limit 40"))
        self.assertIsNotNone(pat.search("python3 -m src.scripts.scraping.capture_dev_day --date 20260906"))
        self.assertIsNotNone(pat.search("python3 -m src.scripts.data.acq_profile_pilot --horse-ids 1"))
        self.assertIsNone(pat.search("python3 -m src.scripts.data.acq_eval run NJ"))      # 評価（取得しない）は待たない

    def test_years_between(self):
        self.assertEqual(bf.years_between("2026-04-01", "2027-01-05"), [2026, 2027])


if __name__ == "__main__":
    unittest.main()
