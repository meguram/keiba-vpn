"""推論時の ped_* / extras が、学習側の関数と同じ値になること。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pipeline.features import pedigree_context as pc
from src.pipeline.features import serving_ped_extras as sv


def _runs(n_days: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        date = pd.Timestamp("2024-01-06") + pd.Timedelta(days=7 * d)
        for r in range(1, 5):
            rid = f"2024{d:02d}{r:02d}"
            surf = ["芝", "ダート", "障"][int(rng.integers(0, 3))]
            ven = ["05", "06"][int(rng.integers(0, 2))]
            dist = int(rng.choice([1200, 1600, 2000, 2400]))
            tc = str(rng.choice(["良", "稍重", "重"]))
            fin = rng.permutation(8) + 1
            for h in range(8):
                rows.append({"date": date, "race_id": rid, "horse_id": f"h{h}", "surface": surf, "distance": dist, "track_condition": tc, "venue_code": ven,
                             "finish_position": int(fin[h]), "pos_early": float(rng.random()), "l3f": float(34 + rng.random() * 3) if rng.random() > 0.05 else np.nan})
    return pd.DataFrame(rows)


class PedParityTest(unittest.TestCase):
    def test_matches_training_function(self):
        h = _runs()
        day = pd.Timestamp("2024-06-01")
        hist, tgt = h[h["date"] < day].reset_index(drop=True), h[h["date"] >= day].reset_index(drop=True)
        full = pd.concat([hist, tgt], ignore_index=True)
        df = full[["date", "race_id", "horse_id", "surface", "distance", "track_condition", "venue_code", "finish_position"]].copy()
        df["sire"] = ""
        df["dam_sire"] = ""
        labels = full[["race_id", "horse_id", "pos_early", "l3f"]]
        ref = pc.build(df, labels)
        st = sv.PedState.from_history(hist)
        first = tgt["date"].min()
        t1 = tgt[tgt["date"] == first]
        got = st.features(t1)
        self.assertEqual(list(got.columns), list(ref.columns))
        np.testing.assert_allclose(got.to_numpy(), ref.loc[len(hist) + t1.index].to_numpy(), rtol=0, atol=0, equal_nan=True)

    def test_same_day_and_future_history_not_used(self):
        h = _runs()
        day = pd.Timestamp("2024-03-02")
        t = h[h["date"] == day].reset_index(drop=True)
        a = sv.PedState.from_history(h[h["date"] < day]).features(t)
        b = sv.PedState.from_history(h).features(t)      # 同日・未来の結果が履歴に入っていても同じ
        np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)

    def test_unknown_going_is_good_and_save_load(self):
        h = _runs()
        day = pd.Timestamp("2024-06-01")
        st = sv.PedState.from_history(h[h["date"] < day])
        t = h[h["date"] == day + pd.Timedelta(days=0)].reset_index(drop=True)
        if t.empty:
            t = h[h["date"] > day].head(8).reset_index(drop=True)
        t = t.assign(date=day + pd.Timedelta(days=1))
        t1 = t.assign(track_condition="良")
        t2 = t.assign(track_condition=None)
        np.testing.assert_allclose(st.features(t1).to_numpy(), st.features(t2).to_numpy(), equal_nan=True)
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            st.save(d)
            st2 = sv.PedState.load(d)
        np.testing.assert_allclose(st.features(t1).to_numpy(), st2.features(t1).to_numpy(), equal_nan=True)

    def test_column_count(self):
        self.assertEqual(len(sv.PED_COLUMNS), 24)


class ExtrasTest(unittest.TestCase):
    def test_rank_gap_and_missing(self):
        keys = pd.DataFrame({"race_id": ["r1"] * 4, "horse_id": ["a", "b", "c", "d"]}, index=[10, 11, 12, 13])
        rows = pd.DataFrame([
            {"race_id": "r1", "horse_id": "a", "oikiri_eval": 4.0, "has_comment": 1.0, "comment_len": 10.0, "comment_pos": 1.0, "comment_neg": 0.0, "comment_result_words": 0.0},
            {"race_id": "r1", "horse_id": "b", "oikiri_eval": 3.0},
            {"race_id": "r1", "horse_id": "c", "oikiri_eval": 3.0, "has_comment": 0.0, "comment_len": 0.0, "comment_pos": 0.0, "comment_neg": 0.0},
        ])
        e = sv.extras_features(keys, rows)
        self.assertEqual(list(e.columns), sv.EXTRAS_COLUMNS)
        self.assertEqual(list(e.index), [10, 11, 12, 13])
        self.assertEqual(float(e.loc[10, "oikiri_eval_rank"]), 1.0)
        self.assertEqual(float(e.loc[11, "oikiri_eval_rank"]), 2.5)
        self.assertEqual(float(e.loc[11, "oikiri_eval_gap"]), -1.0)
        self.assertTrue(np.isnan(e.loc[13, "oikiri_eval"]))
        self.assertEqual(float(e.loc[13, "has_comment"]), 0.0)
        self.assertEqual(float(e.loc[11, "has_comment"]), 0.0)

    def test_no_rows(self):
        keys = pd.DataFrame({"race_id": ["r1", "r1"], "horse_id": ["a", "b"]})
        e = sv.extras_features(keys, pd.DataFrame())
        self.assertTrue(e["oikiri_eval"].isna().all())
        self.assertTrue((e["has_comment"] == 0).all())


if __name__ == "__main__":
    unittest.main()
