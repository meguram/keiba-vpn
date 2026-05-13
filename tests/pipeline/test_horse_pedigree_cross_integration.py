"""ped_tbl 実ファイルがあればクロス列のサンプル検証（任意）。"""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from pipeline.build_horse_entity_store import _flatten_pedigree_json_with_gen5_merge
class TestPedCrossIntegration(unittest.TestCase):
    def test_merge_fixture_cross_columns(self) -> None:
        """test_build_horse_entity_store と同じ最小フィクスチャで merge 後のクロスが一貫すること。"""
        from tempfile import TemporaryDirectory

        import json

        with TemporaryDirectory() as td:
            tdp = Path(td)
            ped_root = tdp / "horse_pedigree_5gen"
            (ped_root / "2019").mkdir(parents=True)
            (ped_root / "2001").mkdir(parents=True)
            rec_s = {
                "ancestors": [
                    {"generation": 5, "position": 0, "name": "Anchor", "horse_id": "2001100001"},
                ]
            }
            rec_h = {
                "ancestors": [
                    {"generation": 1, "position": 0, "name": "Deep", "horse_id": "1990100999"},
                ]
            }
            (ped_root / "2019" / "2019100001.json").write_text(json.dumps(rec_s), encoding="utf-8")
            (ped_root / "2001" / "2001100001.json").write_text(json.dumps(rec_h), encoding="utf-8")

            df = _flatten_pedigree_json_with_gen5_merge(
                rec_s,
                "2019100001",
                ped_root,
            )
            self.assertGreater(len(df), 0)
            self.assertIn("ancestor_occurrence_global", df.columns)
            self.assertIn("ancestor_cross_both_roots", df.columns)
            deep = df[df["path_fm"] == "FFFFFF"]
            self.assertEqual(len(deep), 1)
            self.assertGreaterEqual(int(deep.iloc[0]["ancestor_occurrence_global"]), 1)

    def test_real_parquet_if_present(self) -> None:
        p = Path("data/features/horse/ped_tbl/2009/2009100502.parquet")
        if not p.is_file():
            self.skipTest(f"no sample file: {p}")
        df = pd.read_parquet(p)
        if "ancestor_occurrence_global" not in df.columns:
            self.skipTest("ped_tbl not rebuilt with cross columns")
        self.assertTrue((df["subject_distinct_ancestor_id_count"] == df["subject_distinct_ancestor_id_count"].iloc[0]).all())
        dup = df["ancestor_occurrence_global"] >= 2
        if dup.any():
            row = df.loc[dup].iloc[0]
            hid = str(row["ancestor_horse_id"])
            self.assertEqual(int(row["ancestor_occurrence_global"]), int((df["ancestor_horse_id"] == hid).sum()))


if __name__ == "__main__":
    unittest.main()
