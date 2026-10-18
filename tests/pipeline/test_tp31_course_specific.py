"""tp-31 のコース特注 lift が、日付が厳密に前の結果だけで決まること（先読みが無いこと）。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pipeline.features import tp31_course_specific as cs


def _toy(n_days: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        date = pd.Timestamp("2022-01-01") + pd.Timedelta(days=7 * d)
        for r in range(1, 4):
            ven = ["05", "06"][r % 2]
            dist = [1200, 1600, 2000][r - 1]
            n = 10
            fin = rng.permutation(n) + 1
            for h in range(n):
                rows.append({"race_id": f"{date:%Y%m%d}{ven}{r:02d}", "date": date, "venue_code": ven, "surface": "芝" if r != 2 else "ダ", "distance": dist,
                             "course_type": ["", "外"][r % 2], "sire": f"s{rng.integers(0, 6)}", "dam_sire": f"m{rng.integers(0, 6)}", "jockey_id": f"j{rng.integers(0, 5)}",
                             "trainer_id": f"t{rng.integers(0, 5)}", "a_style": float(rng.random()), "field_size": n, "horse_number": h + 1,
                             "sex_age": f"{['牡', '牝'][h % 2]}{3 + h % 4}", "jockey_weight": 54 + rng.integers(0, 4), "weight": 440 + rng.integers(0, 80),
                             "d_rest": float(rng.integers(7, 120)), "c_cls_step": float(rng.integers(-1, 2)), "finish_position": int(fin[h]), "horse_id": f"h{h}"})
    return pd.DataFrame(rows)


class NoLeakTest(unittest.TestCase):
    def test_future_results_do_not_change_past_rows(self):
        df = _toy()
        a = cs.build_all(df)
        df2 = df.copy()
        late = df2["date"] >= pd.Timestamp("2022-09-01")
        df2.loc[late, "finish_position"] = (df2.loc[late, "finish_position"] % 10) + 1
        b = cs.build_all(df2)
        early = (df["date"] < pd.Timestamp("2022-09-01")).to_numpy()
        for g in a:
            pd.testing.assert_frame_equal(a[g][early].reset_index(drop=True), b[g][early].reset_index(drop=True), check_exact=False, atol=1e-6)

    def test_same_day_results_are_not_used(self):
        df = _toy()
        a = cs.build_group(df, "T3_style_post")
        day = df["date"] == df["date"].iloc[300]
        df2 = df.copy()
        df2.loc[day, "finish_position"] = (df2.loc[day, "finish_position"] % 10) + 1
        b = cs.build_group(df2, "T3_style_post")
        pd.testing.assert_frame_equal(a[day.to_numpy()].reset_index(drop=True), b[day.to_numpy()].reset_index(drop=True), check_exact=False, atol=1e-6)

    def test_first_day_has_zero_lift(self):
        df = _toy()
        a = cs.build_group(df, "T1_sire")
        first = (df["date"] == df["date"].min()).to_numpy()
        self.assertTrue(np.allclose(a.loc[first, [c for c in a.columns if c.endswith("_lift0")]].to_numpy(), 0.0, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
