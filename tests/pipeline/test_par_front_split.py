"""par_front_split ユーティリティのユニットテスト。"""

import unittest

import pandas as pd

from src.pipeline.megu_index.par_front_split import (
    attach_par_front_split_sec,
    fit_par_front_split_coefficients,
)


class TestParFrontSplit(unittest.TestCase):
    def test_fit_and_attach(self):
        rows = []
        for i in range(40):
            rid = f"r{i}"
            t2 = 96.0 + (i % 3) * 0.1
            fs = 48.0 + (i % 3) * 0.05
            rows.append(
                {
                    "race_id": rid,
                    "distance": 1600,
                    "surface": "芝",
                    "finish_pos": 2,
                    "front_split_sec": fs,
                    "race_t2nd_sec": t2,
                    "year": 2023,
                }
            )
        df_2nd = pd.DataFrame(rows)
        par = fit_par_front_split_coefficients(df_2nd, min_cell_n=30)
        self.assertEqual(len(par), 1)
        self.assertIn("par_intercept", par.columns)

        df = pd.DataFrame(
            [
                {
                    "race_id": "r0",
                    "distance": 1600,
                    "surface": "芝",
                    "race_t2nd_sec": 96.0,
                    "front_split_sec": 48.0,
                }
            ]
        )
        out = attach_par_front_split_sec(df, par, df_2nd)
        self.assertTrue(out["par_front_split_sec"].notna().all())


if __name__ == "__main__":
    unittest.main()
