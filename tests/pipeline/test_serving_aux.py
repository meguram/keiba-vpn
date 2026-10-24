"""補助タスクつき NN の保存・推論（mtl_infer / AuxModels）が、学習時の検証予測と同じ値になること。"""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features import serving_aux as sa
from src.pipeline.models.tp25c_mtl import mtl_fit_predict, mtl_infer


def _data(n_races: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        n = int(rng.integers(8, 14))
        fin = rng.permutation(n) + 1
        for h in range(n):
            rows.append({"race_id": f"r{r:03d}", "x1": rng.normal(), "x2": rng.normal() if rng.random() > 0.1 else np.nan, "surface": str(rng.choice(["芝", "ダート"])), "fin": int(fin[h])})
    d = pd.DataFrame(rows)
    d["surface"] = d["surface"].astype("category")
    return d


class MtlInferTest(unittest.TestCase):
    def test_saved_model_reproduces_validation_aux(self):
        d = _data()
        X, race, fin = d[["x1", "x2", "surface"]], d["race_id"].to_numpy(), d["fin"].to_numpy()
        Y = {"win": (fin == 1).astype(int), "top2": (fin <= 2).astype(int), "top3": (fin <= 3).astype(int)}
        A = np.stack([d["x1"].to_numpy() * 2 + 1, np.where(fin <= 3, 1.0, np.nan)], axis=1)
        o = mtl_fit_predict(X, Y, A, race, X, race, seed=0, epochs=2, device="cpu", want_model=True)
        got = mtl_infer(pickle.loads(pickle.dumps(o["model"])), X, race, device="cpu")
        np.testing.assert_allclose(got, o["aux"], rtol=1e-5, atol=1e-5)

    def test_default_has_no_model(self):
        d = _data(20)
        X, race, fin = d[["x1", "x2", "surface"]], d["race_id"].to_numpy(), d["fin"].to_numpy()
        Y = {"win": (fin == 1).astype(int), "top2": (fin <= 2).astype(int), "top3": (fin <= 3).astype(int)}
        o = mtl_fit_predict(X, Y, None, race, X, race, seed=0, epochs=1, device="cpu")
        self.assertNotIn("model", o)


class RelativeTest(unittest.TestCase):
    def test_relative_columns(self):
        names = ["mtl_" + x for x in sa.AUX_NAMES]
        df = pd.DataFrame(np.random.default_rng(1).normal(size=(6, len(names))), columns=names)
        rel = sa.relative(df, ["a", "a", "a", "b", "b", "b"])
        self.assertEqual(len(rel.columns), 2 * len(names))
        self.assertAlmostEqual(float(rel["mtl_t_pos_early_gap"].iloc[:3].sum()), 0.0, places=6)
        self.assertEqual(sa.aux_columns()[:9], names)


if __name__ == "__main__":
    unittest.main()
