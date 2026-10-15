import unittest
from unittest import mock

from src.scraper import auto_scrape


class _FakeStorage:
    def __init__(self, saved):
        self.saved = saved

    def load(self, category, key):
        return self.saved.get(key)


class _FakeRunner:
    def __init__(self, saved, fetched):
        self.storage = _FakeStorage(saved)
        self.fetched = fetched
        self.calls = []

    def scrape_race_list(self, date):
        return [{"race_id": r} for r in self.fetched]

    def scrape_trainer_comment(self, rid, skip_existing=True, min_entries=0):
        self.calls.append((rid, skip_existing, min_entries))
        n = self.fetched[rid]
        return {"entries": [{}] * n} if n >= max(1, min_entries) else None


class RacedayCommentTest(unittest.TestCase):
    def test_saves_only_when_comments_exist_and_never_shrinks(self):
        saved = {"B": {"entries": [{}] * 10}, "C": {"entries": [{}] * 10}}
        runner = _FakeRunner(saved, {"A": 0, "B": 12, "C": 8, "D": 5})
        with mock.patch("src.scraper.run.ScraperRunner", return_value=runner), mock.patch("time.sleep"):
            out = auto_scrape.run_raceday_comment_for_date("20261017")
        self.assertEqual(out["races"], 4)
        self.assertEqual(out["saved"], 2)             # B(10→12) と D(新規 5)
        self.assertEqual(out["no_comment_yet"], 2)    # A は 0 件、C は減るので更新しない
        self.assertEqual(dict((r, m) for r, _, m in runner.calls), {"A": 1, "B": 10, "C": 10, "D": 1})
        self.assertTrue(all(not skip for _, skip, _ in runner.calls))


if __name__ == "__main__":
    unittest.main()
