"""最終モデルの保存・読み込み・推論のテスト（保存したモデルが、学習時と同じ予測を出す）。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.models import ensemble_bundle as eb
from src.pipeline.models import win_models as wm
from src.pipeline.models import zoo


def _data(n_races=80, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        n = int(rng.integers(6, 12)); w = rng.permutation(n)
        for h in range(n):
            x1 = rng.normal()
            rows.append({"race_id": f"r{r}", "x1": x1, "x2": rng.normal(), "x3": rng.normal() if rng.random() > 0.2 else np.nan, "c": rng.choice(["a", "b", "c"]), "fin": int(w[h]) + 1})
    df = pd.DataFrame(rows)
    df["c"] = df["c"].astype("category")
    y = {"win": (df.fin == 1).astype(int).to_numpy(), "top2": (df.fin <= 2).astype(int).to_numpy(), "top3": (df.fin <= 3).astype(int).to_numpy()}
    return df[["x1", "x2", "x3", "c"]], y, df["race_id"].to_numpy()


def _spec(n):
    w = [1.0 / n] * n
    return {"targets": {"win": {"method": "weighted_logit", "weights": w}, "top2": {"method": "greedy", "weights": w}, "top3": {"method": "greedy", "weights": w}}}


class BundleTest(unittest.TestCase):
    def setUp(self):
        self.X, self.y, self.race = _data()
        cols = list(self.X.columns)
        lgb_p = {"lgb": {"num_leaves": 7, "min_data_in_leaf": 10}, "n_rounds": 20}
        self.members = [
            eb.MemberSpec("lgb_b", "lgb_binary", lgb_p, cols),
            eb.MemberSpec("lgb_s", "lgb_softmax", lgb_p, cols),
            eb.MemberSpec("xgb_b", "xgb_binary", {"device": "cpu", "n_estimators": 20, "max_depth": 3}, cols),
            eb.MemberSpec("nn", "nn", {"hidden": (8,), "epochs": 2}, cols),
            eb.MemberSpec("linear", "linear", {"epochs": 2}, cols),
        ]
        self.b = eb.EnsembleBundle(self.members, _spec(len(self.members)), {"lgb_s": (0.9, 0.8)}).fit(self.X, self.y, self.race, threads=2, seeds=[0, 1])

    def test_member_matches_training_time_prediction(self):
        m = self.members[0]
        got = self.b.predict_member(m, self.X, self.race)
        outs = []
        for s in (0, 1):
            o = zoo.fit_predict("lgb_binary", self.X[m.feature_cols], self.y, self.race, self.X[m.feature_cols], self.race, 2, seed=s, params=dict(m.params))
            outs.append(zoo.finalize(self.race, o))
        for t in ("win", "top2", "top3"):
            np.testing.assert_allclose(got[t], (outs[0][t] + outs[1][t]) / 2, atol=1e-9)

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            self.b.save(Path(d))
            b2 = eb.EnsembleBundle.load(Path(d))
            a = self.b.predict(self.X, self.race)
            c = b2.predict(self.X, self.race)
            np.testing.assert_allclose(a.to_numpy(), c.to_numpy(), atol=1e-5)

    def test_totals_per_race(self):
        p = self.b.predict(self.X, self.race)
        s = p.groupby(self.race).sum()
        np.testing.assert_allclose(s["p_win"], 1.0, atol=1e-6)
        np.testing.assert_allclose(s["p_top2"], 2.0, atol=1e-5)
        np.testing.assert_allclose(s["p_top3"], 3.0, atol=1e-5)

    def test_unknown_category_is_handled(self):
        X2 = self.X.copy()
        X2["c"] = X2["c"].astype(str).replace({"a": "zzz"})
        p = self.b.predict(X2, self.race)
        self.assertFalse(p.isna().any().any())



class MtlMemberTest(unittest.TestCase):
    def test_mtl_member_roundtrip_matches_direct_prediction(self):
        from src.pipeline.models.tp25c_mtl import mtl_predict

        X, y, race = _data(n_races=60, seed=3)
        rng = np.random.default_rng(0)
        aux = rng.normal(size=(len(X), 2))
        aux[::7, 0] = np.nan
        cols = list(X.columns)
        m = eb.MemberSpec("mtl", "mtl_nn", {"hidden": (8,), "epochs": 2, "aux_w": 0.5}, cols)
        b = eb.EnsembleBundle([m], _spec(1)).fit(X, y, race, threads=1, seeds=[0, 1], aux=aux)
        with tempfile.TemporaryDirectory() as d:
            b.save(Path(d))
            b2 = eb.EnsembleBundle.load(Path(d))
        p1, p2 = b.predict_member(m, X, race), b2.predict_member(m, X, race)
        for t_ in ("win", "top2", "top3"):
            np.testing.assert_allclose(p1[t_], p2[t_], atol=1e-6)
        o = mtl_predict(b2.models["mtl"][0], b2._cast(X, cols), race)
        self.assertEqual(o["aux"].shape, (len(X), 2))
        self.assertTrue(np.isfinite(p1["win"]).all())


if __name__ == "__main__":
    unittest.main()
