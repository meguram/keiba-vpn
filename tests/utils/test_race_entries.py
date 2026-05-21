import unittest

from src.utils.race_entries import normalize_race_entries


class TestNormalizeRaceEntries(unittest.TestCase):
    def test_all_zero_assigns_sequential(self):
        entries = [
            {"horse_number": 0, "bracket_number": 0, "horse_name": "A"},
            {"horse_number": 0, "bracket_number": 0, "horse_name": "B"},
        ]
        out = normalize_race_entries(entries)
        self.assertEqual([e["horse_number"] for e in out], [1, 2])
        self.assertEqual(out[0]["bracket_number"], 1)
        self.assertEqual(out[1]["bracket_number"], 1)

    def test_valid_unique_unchanged(self):
        entries = [{"horse_number": 3}, {"horse_number": 7}]
        out = normalize_race_entries(entries)
        self.assertEqual([e["horse_number"] for e in out], [3, 7])


if __name__ == "__main__":
    unittest.main()
