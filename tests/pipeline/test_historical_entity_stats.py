"""騎手・調教師・馬の通算成績が『日付が厳密に前』の行だけを数えること（同日・同一レースの結果は数えない）。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pipeline.features.historical_entity_stats import build_history_feature_table

COLS = ["jockey_prior_races", "jockey_prior_wins", "jockey_prior_top3", "trainer_prior_races", "trainer_prior_wins", "horse_prior_races", "horse_prior_wins",
        "horse_prior_top3", "horse_prior_avg_finish"]


def _synthetic(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(30):
        date = (pd.Timestamp("2024-01-06") + pd.Timedelta(days=7 * d)).strftime("%Y-%m-%d")
        for ven in ("05", "08"):
            for r in range(1, 5):
                rid = f"2024{ven}{d + 1:02d}{r:02d}"[:12].ljust(12, "0")
                fin = rng.permutation(6) + 1
                for h in range(6):
                    rows.append({"race_id": rid, "horse_number": h + 1, "horse_id": f"h{rng.integers(0, 14)}", "jockey_id": f"j{rng.integers(0, 4)}",
                                 "trainer_id": f"t{rng.integers(0, 3)}", "date": date, "finish_position": int(fin[h])})
    return pd.DataFrame(rows).drop_duplicates(["race_id", "horse_number"])


def _brute(df: pd.DataFrame, idc: str, rid: str, hn: int) -> tuple[int, float, float]:
    row = df[(df["race_id"] == rid) & (df["horse_number"] == hn)].iloc[0]
    prev = df[(df[idc] == row[idc]) & (pd.to_datetime(df["date"]) < pd.to_datetime(row["date"]))]
    return len(prev), float((prev["finish_position"] == 1).sum()), float((prev["finish_position"] <= 3).sum())


class StrictPriorTest(unittest.TestCase):
    def test_matches_bruteforce_strictly_before_date(self):
        df = _synthetic()
        f = build_history_feature_table(df).set_index(["race_id", "horse_number"])
        for _, r in df.sample(60, random_state=1).iterrows():
            for idc, pre in (("jockey_id", "jockey"), ("trainer_id", "trainer"), ("horse_id", "horse")):
                n, w, t3 = _brute(df, idc, r["race_id"], int(r["horse_number"]))
                got = f.loc[(r["race_id"], int(r["horse_number"]))]
                self.assertEqual((got[f"{pre}_prior_races"], got[f"{pre}_prior_wins"], got[f"{pre}_prior_top3"]), (n, w, t3), (idc, r["race_id"]))

    def test_same_race_and_same_day_results_do_not_change_values(self):
        df = _synthetic()
        a = build_history_feature_table(df).set_index(["race_id", "horse_number"])
        b_df = df.copy()
        day = b_df["date"].iloc[len(b_df) // 2]
        same = b_df["date"] == day
        b_df.loc[same, "finish_position"] = (b_df.loc[same, "finish_position"] % 6) + 1       # 同じ日の結果をすべて変える
        b = build_history_feature_table(b_df).set_index(["race_id", "horse_number"])
        idx = df.index[same]
        keys = list(zip(df.loc[idx, "race_id"], df.loc[idx, "horse_number"]))
        pd.testing.assert_frame_equal(a.loc[keys, COLS], b.loc[keys, COLS])                      # その日の行の特徴量は変わらない

    def test_future_results_do_not_change_past_rows(self):
        df = _synthetic()
        a = build_history_feature_table(df).set_index(["race_id", "horse_number"])
        b_df = df.copy()
        late = pd.to_datetime(b_df["date"]) >= pd.Timestamp("2024-06-01")
        b_df.loc[late, "finish_position"] = (b_df.loc[late, "finish_position"] % 6) + 1
        b = build_history_feature_table(b_df).set_index(["race_id", "horse_number"])
        early = df[~late]
        keys = list(zip(early["race_id"], early["horse_number"]))
        pd.testing.assert_frame_equal(a.loc[keys], b.loc[keys])


if __name__ == "__main__":
    unittest.main()
