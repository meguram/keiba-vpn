"""疑似特徴量ビルダーのテスト。"""
from __future__ import annotations

import unittest

from src.pipeline.features.pseudo_builder import META_COLUMNS, PseudoFeatureBuilder, get_feature_builder

RACE = {
    "race_id": "202605030811",
    "race_card": {"entries": [
        {"horse_id": f"H{i}", "horse_number": i, "horse_name": f"馬{i}"} for i in range(1, 17)
    ]},
}


class PseudoBuilderTest(unittest.TestCase):
    def test_shape_and_columns(self):
        df = PseudoFeatureBuilder(n_features=1000).build(RACE)
        self.assertEqual(df.shape, (16, 4 + 1000))
        self.assertEqual(list(df.columns[:4]), META_COLUMNS)
        self.assertEqual(df.columns[4], "pf_0000")
        self.assertEqual(df["pf_0000"].dtype, "float32")

    def test_deterministic_and_entry_specific(self):
        b = PseudoFeatureBuilder(n_features=20)
        a1, a2 = b.build(RACE), b.build(RACE)
        self.assertTrue(a1.equals(a2))
        self.assertFalse(a1.iloc[0, 4:].equals(a1.iloc[1, 4:]))

    def test_same_horse_in_different_race_differs(self):
        other = {**RACE, "race_id": "202605030812"}
        b = PseudoFeatureBuilder(n_features=20)
        self.assertFalse(b.build(RACE).iloc[0, 4:].equals(b.build(other).iloc[0, 4:]))

    def test_no_entries_gives_empty_frame(self):
        self.assertTrue(PseudoFeatureBuilder(5).build({"race_id": "x"}).empty)

    def test_factory_default_and_unknown(self):
        self.assertEqual(get_feature_builder().name, "pseudo")
        with self.assertRaises(NotImplementedError):
            get_feature_builder("real")


if __name__ == "__main__":
    unittest.main()
