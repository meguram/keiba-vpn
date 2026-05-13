"""pipeline.build_rank_target の unittest。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.build_rank_target import build_rank_tbl_for_year, write_rank_target_parquets


class TestBuildRankTarget(unittest.TestCase):
    def test_build_rank_tbl_for_year(self):
        df = pd.DataFrame(
            {
                "race_id": ["202401010101", "202401010101"],
                "horse_id": ["2019100001", "2019100002"],
                "finish_position": [1, "2"],
            }
        )
        out = build_rank_tbl_for_year(df)
        self.assertEqual(len(out), 2)
        self.assertListEqual(list(out.columns), ["race_id", "horse_id", "rank"])
        self.assertEqual(int(out["rank"].iloc[0]), 1)
        self.assertEqual(int(out["rank"].iloc[1]), 2)

    def test_write_rank_target_parquets(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            y = "2024"
            src = tdp / "data" / "local" / "tables" / y
            src.mkdir(parents=True)
            pq.write_table(
                pa.table(
                    {
                        "race_id": ["202401010101"],
                        "horse_id": ["2019100001"],
                        "finish_position": [3],
                    }
                ),
                src / "race_result_flat.parquet",
            )
            man = write_rank_target_parquets(base_dir=tdp, years=[y], overwrite=True)
            self.assertIn(y, man["files_written"])
            p = tdp / "data" / "features" / "target" / "rank_tbl" / y / "rank.parquet"
            self.assertTrue(p.is_file())
            r = pd.read_parquet(p)
            self.assertListEqual(list(r.columns), ["race_id", "horse_id", "rank"])


if __name__ == "__main__":
    unittest.main()
