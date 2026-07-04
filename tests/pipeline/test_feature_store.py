"""FeatureStore の年別 Parquet 分割の unittest。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.pipeline.features.feature_store import FeatureStore


class TestFeatureStoreYearPartition(unittest.TestCase):
    def test_save_partitions_by_race_id_year(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            store = FeatureStore(base_dir=tdp, features_dir=tdp / "features")
            df = pd.DataFrame(
                {
                    "race_id": ["2024010101", "2023010101"],
                    "horse_id": ["a", "b"],
                    "xfeat": [1.0, 2.0],
                }
            )
            p = store.save_feature_column(
                "xfeat",
                df,
                table_block="race_horse_tbl",
                merge_keys=["race_id", "horse_id"],
                overwrite=True,
            )
            self.assertTrue(p.is_file())
            self.assertIn("2023", p.parts)
            meta = store.column_info("xfeat")
            self.assertTrue(meta.get("year_partitioned"))
            self.assertIn("2023", meta.get("year_paths", {}))
            self.assertIn("2024", meta.get("year_paths", {}))
            loaded = store.load_column("xfeat")
            self.assertEqual(len(loaded), 2)
            sub = store.load_column("xfeat", years=["2023"])
            self.assertEqual(len(sub), 1)
            self.assertEqual(sub["xfeat"].iloc[0], 2.0)

    def test_partition_partial_merges_year_paths(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            store = FeatureStore(base_dir=tdp, features_dir=tdp / "features")
            df23 = pd.DataFrame(
                {"race_id": ["2023010101"], "horse_id": ["a"], "xfeat": [2.0]},
            )
            store.save_feature_column(
                "xfeat",
                df23,
                table_block="race_horse_tbl",
                merge_keys=["race_id", "horse_id"],
                overwrite=True,
                partition_partial=False,
            )
            df24 = pd.DataFrame(
                {"race_id": ["2024010101"], "horse_id": ["b"], "xfeat": [1.0]},
            )
            store.save_feature_column(
                "xfeat",
                df24,
                table_block="race_horse_tbl",
                merge_keys=["race_id", "horse_id"],
                overwrite=True,
                partition_partial=True,
            )
            p23 = store.block_dir("race_horse_tbl") / "2023" / "xfeat.parquet"
            p24 = store.block_dir("race_horse_tbl") / "2024" / "xfeat.parquet"
            self.assertTrue(p23.is_file())
            self.assertTrue(p24.is_file())
            full = store.load_column("xfeat")
            self.assertEqual(len(full), 2)

    def test_save_single_file_without_race_id(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            store = FeatureStore(base_dir=tdp, features_dir=tdp / "features")
            df = pd.DataFrame({"jockey_id": ["1"], "score": [0.5]})
            store.save_feature_column(
                "score",
                df,
                table_block="jockey_tbl",
                merge_keys=["jockey_id"],
                overwrite=True,
            )
            flat = store.block_dir("jockey_tbl") / "score.parquet"
            self.assertTrue(flat.is_file())
            meta = store.column_info("score")
            self.assertFalse(meta.get("year_partitioned"))


if __name__ == "__main__":
    unittest.main()
