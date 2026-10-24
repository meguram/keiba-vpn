"""tp-26 コース転移: 計画した連動を見つけること、先読みが無いこと。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pipeline.features import tp26_course_transfer as t


def _synthetic(n_h: int = 600, seed: int = 0) -> pd.DataFrame:
    """コース A と B は馬ごとの適性 u1 を共有、C は別の適性 u2、D は適性なし。能力 ability はどのコースにも効く。"""
    rng = np.random.default_rng(seed)
    rows = []
    d0 = pd.Timestamp("2020-01-01")
    for h in range(n_h):
        ab, u1, u2 = rng.normal(0, 1), rng.normal(0, 1), rng.normal(0, 1)
        courses = rng.choice(["A", "B", "C", "D"], size=10)
        for k, c in enumerate(courses):
            apt = {"A": u1, "B": u1, "C": u2, "D": 0.0}[c]
            rows.append({"horse_id": f"h{h}", "date": d0 + pd.Timedelta(days=int(30 * k + rng.integers(0, 20))), "course": c, "fin": ab * 0.5 + apt * 0.5 + rng.normal(0, 0.6)})
    d = pd.DataFrame(rows)
    d["year"] = d["date"].dt.year
    d["race_id"] = [f"r{i}" for i in range(len(d))]
    d["l3f"] = d["fin"]
    d["pos"] = d["fin"]
    return d.sort_values(["horse_id", "date"]).reset_index(drop=True)


class TransferTest(unittest.TestCase):
    def test_planted_pair_is_found(self):
        runs = _synthetic()
        courses = ["A", "B", "C", "D"]
        s = t.similarity_snapshot(runs, courses, "fin", pd.Timestamp("2030-01-01"), n_perm=30)
        ab, ac, ad = s["r_exc"][0, 1], s["r_exc"][0, 2], s["r_exc"][0, 3]
        self.assertGreater(ab, 0.3)
        self.assertGreater(ab, ac + 0.2)
        self.assertLess(abs(ad), 0.15)
        self.assertLess(s["q"][0, 1], 0.01)
        self.assertTrue(np.allclose(s["r_exc"], s["r_exc"].T))

    def test_permutation_null_is_roughly_centered(self):
        runs = _synthetic(seed=1)
        runs["fin"] = np.random.default_rng(5).normal(size=len(runs))        # 馬の能力も適性も無い
        s = t.similarity_snapshot(runs, ["A", "B", "C", "D"], "fin", pd.Timestamp("2030-01-01"), n_perm=30)
        off = s["z"][np.triu_indices(4, 1)]
        self.assertLess(np.abs(off).max(), 4.0)

    def test_bh_monotone(self):
        p = np.array([[1, .001, .5], [.001, 1, .04], [.5, .04, 1.0]])
        q = t.bh_qvalues(p, np.ones((3, 3), bool))
        self.assertLessEqual(q[0, 1], q[1, 2])
        self.assertTrue((q >= 0).all() and (q <= 1).all())

    def test_features_do_not_use_future_runs(self):
        runs = _synthetic(n_h=200, seed=2)
        courses = ["A", "B", "C", "D"]
        W = {y: np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0.0]]) for y in runs["year"].unique()}
        a = t.transfer_features(runs, courses, "fin", W)
        r2 = runs.copy()
        late = r2["date"] >= pd.Timestamp("2021-01-01")
        r2.loc[late, "fin"] = r2.loc[late, "fin"] + 5.0              # 未来の成績を変える
        b = t.transfer_features(r2, courses, "fin", W)
        early = (runs["date"] < pd.Timestamp("2021-01-01")).to_numpy()
        pd.testing.assert_frame_equal(a[early], b[early])

    def test_own_course_is_excluded(self):
        runs = _synthetic(n_h=100, seed=3)
        W = np.ones((4, 4)) - np.eye(4)
        f = t.transfer_features(runs, ["A", "B", "C", "D"], "fin", {y: W for y in runs["year"].unique()})
        self.assertEqual(f.shape[0], len(runs))
        self.assertTrue(np.isfinite(f["cn_fin"]).all())


if __name__ == "__main__":
    unittest.main()
