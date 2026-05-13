"""pipeline.horse_pedigree_expand の unittest。"""

from __future__ import annotations

import unittest

from research.pedigree_similarity import is_paternal_side

from pipeline.horse_pedigree_expand import (
    PRIMARY_MAX_GENERATION,
    fm_path_from_gp,
    global_depth_after_merge,
    iter_gen5_male_anchor_horse_ids,
    merge_primary_and_branches,
)


class TestHorsePedigreeExpand(unittest.TestCase):
    def test_fm_path_length_and_charset(self) -> None:
        for g in range(1, PRIMARY_MAX_GENERATION + 1):
            for p in range(2**g):
                s = fm_path_from_gp(g, p)
                self.assertEqual(len(s), g, msg=f"g={g} p={p}")
                self.assertTrue(all(c in "FM" for c in s), msg=s)

    def test_fm_path_matches_paternal_walk(self) -> None:
        """各文字がその段の (generation, position) での is_paternal_side と一致すること。"""
        for g in range(2, PRIMARY_MAX_GENERATION + 1):
            for p in range(2**g):
                path = fm_path_from_gp(g, p)
                cg, cp = g, p
                for i in range(g):
                    self.assertEqual(path[i] == "F", is_paternal_side(cg, cp), msg=f"g={g} p={p} step={i}")
                    if cg <= 1:
                        break
                    half = 2 ** (cg - 1)
                    if cp < half:
                        cp = cp
                    else:
                        cp = cp - half
                    cg -= 1

    def test_fm_path_known_corners(self) -> None:
        self.assertEqual(fm_path_from_gp(1, 0), "F")
        self.assertEqual(fm_path_from_gp(1, 1), "M")
        self.assertEqual(fm_path_from_gp(5, 0), "FFFFF")
        self.assertEqual(fm_path_from_gp(5, 31), "MMMMM")

    def test_global_depth_after_merge(self) -> None:
        self.assertEqual(global_depth_after_merge(1), 6)
        self.assertEqual(global_depth_after_merge(5), 10)

    def test_iter_gen5_male_anchors(self) -> None:
        rows = [
            {"generation": 5, "position": 0, "horse_id": "2001100001", "name": "A"},
            {"generation": 5, "position": 1, "horse_id": "2001100002", "name": "B"},
            {"generation": 4, "position": 0, "horse_id": "1999100001", "name": "C"},
        ]
        anchors = iter_gen5_male_anchor_horse_ids(rows)
        self.assertEqual(len(anchors), 1)
        self.assertEqual(anchors[0][0], "2001100001")
        self.assertEqual(len(anchors[0][1]), 5)

    def test_merge_primary_and_branches_deep_row(self) -> None:
        primary = [
            {"generation": 5, "position": 0, "horse_id": "2001100001", "name": "AnchorSire"},
        ]
        branch = {
            "2001100001": [
                {"generation": 1, "position": 0, "horse_id": "1990100999", "name": "DeepSire"},
            ]
        }
        out = merge_primary_and_branches("2019100001", primary, branch, subject_horse_name="Subject")
        paths = {r["path_fm"]: r for r in out}
        self.assertIn("FFFFF", paths)
        deep = paths.get("FFFFFF")
        self.assertIsNotNone(deep)
        assert deep is not None
        self.assertEqual(deep["source"], "merged_gen5_sire")
        self.assertEqual(deep["anchor_horse_id"], "2001100001")
        self.assertEqual(int(deep["generation"]), 6)
        self.assertEqual(deep["horse_id"], "1990100999")

    def test_merge_branch_skips_female_slot(self) -> None:
        primary = [
            {"generation": 5, "position": 0, "horse_id": "2001100001", "name": "AnchorSire"},
        ]
        branch = {
            "2001100001": [
                {"generation": 1, "position": 1, "horse_id": "1990100888", "name": "DamOnly"},
            ]
        }
        out = merge_primary_and_branches("2019100001", primary, branch)
        paths = [r["path_fm"] for r in out]
        self.assertIn("FFFFF", paths)
        self.assertNotIn("FFFFFM", paths)

if __name__ == "__main__":
    unittest.main()
