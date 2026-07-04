"""pipeline.features.horse_pedigree_cross の unittest。"""

from __future__ import annotations

import unittest

import pandas as pd

from src.pipeline.features.horse_pedigree_cross import add_pedigree_cross_columns


class TestHorsePedigreeCross(unittest.TestCase):
    def _minimal_df(self, rows: list[dict]) -> pd.DataFrame:
        """add_pedigree_cross_columns が期待する最小列。"""
        return pd.DataFrame(rows)

    def test_duplicate_on_paternal_only(self) -> None:
        df = self._minimal_df(
            [
                {"path_fm": "F", "ancestor_horse_id": "A"},
                {"path_fm": "FF", "ancestor_horse_id": "A"},
                {"path_fm": "M", "ancestor_horse_id": "B"},
            ]
        )
        out = add_pedigree_cross_columns(df)
        self.assertEqual(int(out.loc[0, "ancestor_occurrence_global"]), 2)
        self.assertEqual(int(out.loc[0, "ancestor_occurrence_paternal"]), 2)
        self.assertEqual(int(out.loc[0, "ancestor_occurrence_maternal"]), 0)
        self.assertEqual(int(out.loc[0, "ancestor_cross_both_roots"]), 0)
        self.assertEqual(int(out.loc[2, "ancestor_occurrence_maternal"]), 1)
        self.assertEqual(int(out["subject_duplicate_ancestor_id_count"].iloc[0]), 1)
        self.assertEqual(int(out["subject_cross_both_roots_ancestor_id_count"].iloc[0]), 0)
        self.assertEqual(int(out["subject_cross_duplicate_excess_slots"].iloc[0]), 1)
        self.assertEqual(int(out["subject_distinct_ancestor_id_count"].iloc[0]), 2)

    def test_cross_both_roots(self) -> None:
        df = self._minimal_df(
            [
                {"path_fm": "F", "ancestor_horse_id": "X"},
                {"path_fm": "M", "ancestor_horse_id": "X"},
            ]
        )
        out = add_pedigree_cross_columns(df)
        self.assertEqual(int(out.loc[0, "ancestor_occurrence_global"]), 2)
        self.assertEqual(int(out.loc[0, "ancestor_occurrence_paternal"]), 1)
        self.assertEqual(int(out.loc[0, "ancestor_occurrence_maternal"]), 1)
        self.assertEqual(int(out.loc[0, "ancestor_cross_both_roots"]), 1)
        self.assertEqual(int(out["subject_cross_both_roots_ancestor_id_count"].iloc[0]), 1)

    def test_pct_global(self) -> None:
        df = self._minimal_df(
            [
                {"path_fm": "F", "ancestor_horse_id": "A"},
                {"path_fm": "M", "ancestor_horse_id": "B"},
            ]
        )
        out = add_pedigree_cross_columns(df)
        self.assertAlmostEqual(float(out.loc[0, "ancestor_pct_tree_global"]), 50.0)
        self.assertAlmostEqual(float(out.loc[1, "ancestor_pct_tree_global"]), 50.0)

    def test_pct_paternal_subtree(self) -> None:
        df = self._minimal_df(
            [
                {"path_fm": "F", "ancestor_horse_id": "A"},
                {"path_fm": "FF", "ancestor_horse_id": "A"},
                {"path_fm": "FM", "ancestor_horse_id": "C"},
            ]
        )
        out = add_pedigree_cross_columns(df)
        self.assertAlmostEqual(float(out.loc[0, "ancestor_pct_paternal_subtree"]), 200.0 / 3.0)

    def test_empty_id_rows(self) -> None:
        df = self._minimal_df(
            [
                {"path_fm": "F", "ancestor_horse_id": pd.NA},
                {"path_fm": "M", "ancestor_horse_id": "Z"},
            ]
        )
        out = add_pedigree_cross_columns(df)
        self.assertEqual(int(out.loc[0, "ancestor_occurrence_global"]), 0)
        self.assertTrue(pd.isna(out.loc[0, "ancestor_pct_tree_global"]))


if __name__ == "__main__":
    unittest.main()
