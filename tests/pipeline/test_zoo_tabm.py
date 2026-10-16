"""NN の TabM-mini とノイズ注入のオプションのテスト。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd


def _data(races=600, n=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(races):
        x = rng.normal(size=(n, 3))
        sc = x[:, 0] * 1.5 + rng.normal(0, 0.5, n)
        order = np.argsort(-sc)
        fin = np.empty(n, int); fin[order] = np.arange(1, n + 1)
        for i in range(n):
            rows.append({"race_id": f"r{r:04d}", "a": x[i, 0], "b": x[i, 1], "c": x[i, 2], "fin": fin[i]})
    d = pd.DataFrame(rows)
    return d


class TabmTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch がありません")

    def _fit(self, epochs=6, **kw):
        from src.pipeline.models import zoo

        d = _data()
        tr = d["race_id"] < "r0500"
        X = d[["a", "b", "c"]]
        y = {"win": (d["fin"] == 1).astype(int).to_numpy(), "top2": (d["fin"] <= 2).astype(int).to_numpy(), "top3": (d["fin"] <= 3).astype(int).to_numpy()}
        o = zoo.nn_fit_predict(X[tr].reset_index(drop=True), {k: v[tr.to_numpy()] for k, v in y.items()}, d.loc[tr, "race_id"].to_numpy(),
                               X[~tr].reset_index(drop=True), d.loc[~tr, "race_id"].to_numpy(), seed=0, hidden=(16,), epochs=epochs, lr=1e-2, device="cpu", **kw)
        return d[~tr], o

    def test_tabm_outputs_and_learns(self):
        dv, o = self._fit(epochs=25, tabm_k=4)
        for k in ("win_score", "top2_logit", "top3_logit"):
            self.assertEqual(len(o[k]), len(dv)); self.assertTrue(np.isfinite(o[k]).all())
        # 学習できている: 勝ち馬のスコアの平均が他より高い
        win = (dv["fin"] == 1).to_numpy()
        self.assertGreater(o["win_score"][win].mean(), o["win_score"][~win].mean())

    def test_noise_runs(self):
        dv, o = self._fit(noise=0.1)
        self.assertTrue(np.isfinite(o["win_score"]).all())

    def test_tabm_via_fit_predict(self):
        from src.pipeline.models import zoo

        d = _data()
        tr = (d["race_id"] < "r0500").to_numpy()
        X = d[["a", "b", "c"]]
        y = {"win": (d["fin"] == 1).astype(int).to_numpy(), "top2": (d["fin"] <= 2).astype(int).to_numpy(), "top3": (d["fin"] <= 3).astype(int).to_numpy(), "finish": d["fin"].to_numpy()}
        out = zoo.fit_predict("nn", X[tr].reset_index(drop=True), {k: v[tr] for k, v in y.items()}, d["race_id"].to_numpy()[tr], X[~tr].reset_index(drop=True), d["race_id"].to_numpy()[~tr], 2,
                              params={"hidden": (16,), "epochs": 4, "tabm_k": 3})
        fp = zoo.finalize(d["race_id"].to_numpy()[~tr], out)
        np.testing.assert_allclose(pd.Series(fp["win"]).groupby(d["race_id"].to_numpy()[~tr]).sum().to_numpy(), 1.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
