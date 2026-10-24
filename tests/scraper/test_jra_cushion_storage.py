import unittest

from src.scraper.jra_cushion_storage import merge_year_records


class MergeYearRecordsTest(unittest.TestCase):
    def test_never_drops_existing_keys_and_prefers_new(self):
        old = [{"date": "2026-01-03", "venue_code": "06", "v": 1}, {"date": "2026-01-04", "venue_code": "06", "v": 2}]
        new = [{"date": "2026-01-04", "venue_code": "06", "v": 9}, {"date": "2026-01-10", "venue_code": "06", "v": 3}]
        out = merge_year_records(old, new)
        self.assertEqual([r["v"] for r in out], [1, 9, 3])

    def test_identical_is_unchanged(self):
        rows = [{"date": "2025-01-05", "venue_code": "06", "v": 1}]
        self.assertEqual(merge_year_records(rows, list(rows)), rows)


if __name__ == "__main__":
    unittest.main()
