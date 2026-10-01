"""FeatureStore.load_rows_for_keys（推論向け・行フィルタ読み込み）のテスト。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features.feature_store import FeatureStore


def _frame(years: list[str], n_races: int, col: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for y in years:
        for r in range(n_races):
            rid = f"{y}{r:08d}"
            for h in range(8):
                rows.append((rid, f"H{y}{r:04d}{h}", float(rng.random())))
    return pd.DataFrame(rows, columns=["race_id", "horse_id", col])


class LoadRowsForKeysTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.store = FeatureStore(base_dir=cls._tmp.name)
        cls.cols = ["f_a", "f_b", "f_c"]
        for i, c in enumerate(cls.cols):
            cls.store.save_feature_column(c, _frame(["2025", "2026"], 20, c, i), table_block="race_horse_tbl",
                                          merge_keys=["race_id", "horse_id"])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_matches_load_columns_for_selected_races(self):
        target = ["202600000003", "202600000011"]
        rows = self.store.load_rows_for_keys(self.cols, target)
        full = self.store.load_columns(self.cols, years=[2026])
        want = full[full["race_id"].isin(target)].sort_values(["race_id", "horse_id"]).reset_index(drop=True)
        got = rows.sort_values(["race_id", "horse_id"]).reset_index(drop=True)
        self.assertEqual(len(got), 16)
        pd.testing.assert_frame_equal(got[want.columns], want, check_dtype=False)

    def test_only_requested_rows_are_returned(self):
        rows = self.store.load_rows_for_keys(self.cols, ["202600000000"])
        self.assertEqual(set(rows["race_id"]), {"202600000000"})
        self.assertEqual(len(rows), 8)

    def test_year_is_inferred_from_race_id(self):
        rows = self.store.load_rows_for_keys(["f_a"], ["202500000001", "202600000001"])
        self.assertEqual(set(rows["race_id"]), {"202500000001", "202600000001"})

    def test_unknown_race_gives_empty_and_unknown_column_raises(self):
        self.assertEqual(len(self.store.load_rows_for_keys(self.cols, ["209900000001"])), 0)
        with self.assertRaises(FileNotFoundError):
            self.store.load_rows_for_keys(["no_such_col"], ["202600000001"])

    def test_empty_inputs(self):
        self.assertTrue(self.store.load_rows_for_keys([], ["202600000001"]).empty)
        self.assertTrue(self.store.load_rows_for_keys(self.cols, []).empty)


if __name__ == "__main__":
    unittest.main()
