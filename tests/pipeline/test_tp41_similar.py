"""tp-41 同じレースの似た馬の特徴量: 先読みが無いこと（将来・同じ日の結果を変えても特徴量が変わらない）と、定義の確認。"""

from __future__ import annotations

import unittest
import warnings

import numpy as np
import pandas as pd

from src.pipeline.features import tp41_similar as ts
from tests.pipeline.test_factory_self import _late_modified, _toy

warnings.filterwarnings("ignore")


def _megu(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    m = df[ts.KEY].copy()
    m["megu"] = 40 + 8 * rng.standard_normal(len(m)) - 0.3 * df["finish_position"].to_numpy()      # 結果に連動（先読みしていれば検出できる）
    return m


def _features(df: pd.DataFrame, megu: pd.DataFrame | None) -> pd.DataFrame:
    T = ts.runner_table(df, megu)
    T["sire_id"] = "S" + (T["horse_id"].str[-1].astype(int) % 3).astype(str)
    T["dsire_id"] = "D" + (T["horse_id"].str[-2].astype(int) % 2).astype(str)
    T = ts.sort_for_races(ts.standardize_strength(T))
    F = ts.runner_features(T)
    return F.set_index(ts.KEY)


class NoLeakTest(unittest.TestCase):
    def test_future_results_do_not_change_past_rows(self):
        df = _toy()
        cut = "2021-06-01"
        a = _features(df, _megu(df))
        d2 = _late_modified(df, cut)
        b = _features(d2, _megu(d2))
        early = df[df["date"] < pd.Timestamp(cut)][ts.KEY].set_index(ts.KEY).index
        pd.testing.assert_frame_equal(a.loc[early].sort_index(), b.loc[early].sort_index(), check_exact=False, atol=1e-9)

    def test_same_day_results_are_not_used(self):
        df = _toy()
        day = df["date"].sort_values().iloc[len(df) // 2]
        d2 = df.copy()
        same = d2["date"] == day
        d2.loc[same, "finish_position"] = (d2.loc[same, "finish_position"] % 8) + 1
        d2.loc[same, "passing_order"] = "1-1-1"
        d2.loc[same, "margin"] = "大"
        a, b = _features(df, _megu(df)), _features(d2, _megu(d2))
        idx = df[same][ts.KEY].set_index(ts.KEY).index
        pd.testing.assert_frame_equal(a.loc[idx].sort_index(), b.loc[idx].sort_index(), check_exact=True)       # 厳密一致


class DefinitionTest(unittest.TestCase):
    def _tiny(self):
        return pd.DataFrame({
            "race_id": ["R"] * 4, "horse_id": list("abcd"), "date": pd.Timestamp("2022-01-01"), "horse_number": [1, 2, 3, 4], "nfield": 4.0,
            "sty": [0.1, 0.2, 0.8, np.nan], "apt_logd": [7.4, 7.4, 7.7, 7.4], "apt_turf": [1.0, 1.0, 0.0, 1.0], "s": [1.0, 0.5, -0.5, 0.0],
            "trainer_id": ["t1", "t1", "t2", "t3"], "sire_id": ["x", "x", "y", "nan"], "dsire_id": ["m", "n", "m", "nan"],
        })

    def test_front_counts_and_stable_mates(self):
        T = self._tiny()
        F = ts.runner_features(T)
        self.assertEqual(F["sp_front_n"].tolist(), [1.0, 1.0, 2.0, 2.0])             # 先行（<0.25）の他馬の数
        self.assertEqual(F["sp_own_front"].tolist(), [1.0, 1.0, 0.0, 0.0])
        self.assertEqual(F["st_trainer_n"].tolist(), [1.0, 1.0, 0.0, 0.0])
        self.assertEqual(F["st_sire_n"].tolist(), [1.0, 1.0, 0.0, 0.0])
        self.assertEqual(F["st_dsire_n"].tolist(), [1.0, 0.0, 1.0, 0.0])
        self.assertEqual(F["st_trainer_rank"].iloc[:2].tolist(), [0.0, 1.0])         # 同じ調教師の中で a が最も強い
        self.assertTrue(np.isnan(F["st_trainer_rank"].iloc[2]))
        self.assertEqual(F["ss_pct"].tolist()[:3], [1.0, 2 / 3, 0.0])

    def test_pairs_count_and_flags(self):
        T = self._tiny()
        T["sire_id"], T["dsire_id"] = T["sire_id"].astype(str), T["dsire_id"].astype(str)
        Pp = ts.race_pairs(T)
        self.assertEqual(len(Pp), 6)
        row = Pp[(Pp["i"] == 0) & (Pp["j"] == 1)].iloc[0]
        self.assertEqual((row["same_trainer"], row["same_sire"], row["both_front"]), (1.0, 1.0, 1.0))
        self.assertAlmostEqual(row["d_str"], 0.5)
        row = Pp[(Pp["i"] == 0) & (Pp["j"] == 3)].iloc[0]
        self.assertTrue(np.isnan(row["both_front"]))                                  # 位置が不明なペアは欠損


if __name__ == "__main__":
    unittest.main()
