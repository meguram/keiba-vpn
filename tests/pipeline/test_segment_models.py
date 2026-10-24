"""区分別モデル（クラス・コース・競馬場）の部品のテスト。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.pipeline.models import segment_models as sm


class SegmentTest(unittest.TestCase):
    def test_grade_bucket(self):
        rc = pd.Series(["サラ系３歳 未勝利", "サラ系２歳 新馬", "サラ系３歳以上 １勝クラス", "サラ系４歳以上 ２勝クラス", "サラ系３歳以上 ３勝クラス", "サラ系３歳以上 オープン", "", "サラ系４歳以上 オープン"])
        sf = pd.Series(["ダート", "芝", "芝", "ダート", "芝", "芝", "障", "芝"])
        g = sm.grade_bucket(rc, sf).tolist()
        self.assertEqual(g, ["未勝利", "新馬", "1勝", "2勝", "3勝", "オープン・重賞", "障害", "オープン・重賞"])

    def test_course_key_and_venue_surface(self):
        k = sm.course_key(pd.Series(["芝", "ダート", "障", "芝"]), pd.Series([1200, 1800, 3000, 2400]))
        self.assertEqual(list(k), ["芝/〜1300", "ダ/1800〜2100", "障害", "芝/2200〜"])
        v = sm.venue_surface(pd.Series(["5", "06"]), pd.Series(["芝", "ダート"]))
        self.assertEqual(list(v), ["05/芝", "06/ダ"])

    def _data(self, races=400, n=8, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        for r in range(races):
            seg = "A" if r % 2 == 0 else "B"
            x = rng.normal(size=n)
            score = x * (1.0 if seg == "A" else -1.0)         # 区分によって特徴量の効き方が逆
            order = np.argsort(-(score + rng.normal(0, 0.5, n)))
            fin = np.empty(n, int); fin[order] = np.arange(1, n + 1)
            for i in range(n):
                rows.append({"race_id": f"r{r:04d}", "horse_id": f"h{i}", "year": 2020 + r * 4 // races, "x": x[i], "finish_position": fin[i], "seg": seg})
        d = pd.DataFrame(rows)
        return d

    def test_separate_beats_global_when_effect_differs(self):
        d = self._data()
        X = d[["x"]]
        meta = d[["race_id", "horse_id", "year", "finish_position"]]
        yrs = d["year"].to_numpy()
        folds = [SimpleNamespace(train_index=np.flatnonzero(yrs < 2023), valid_index=np.flatnonzero(yrs == 2023))]
        prm = {"learning_rate": 0.1, "num_leaves": 4, "min_data_in_leaf": 20, "lambda_l2": 1.0}
        sep, info = sm.oof_segmented(X, meta, folds, d["seg"].to_numpy(), prm, 60, 2, mode="separate", min_rows=100)
        glob, _ = sm.oof_segmented(X, meta, folds, np.zeros(len(d), int).astype(str), prm, 60, 2, mode="separate", min_rows=100)
        m = sep.merge(d[["race_id", "horse_id", "finish_position"]], on=["race_id", "horse_id"])
        mg = glob.merge(d[["race_id", "horse_id", "finish_position"]], on=["race_id", "horse_id"])
        ll = lambda df: wm.binary_logloss(df["p_win"].to_numpy(), (df["finish_position"] == 1).astype(float).to_numpy())
        self.assertLess(ll(m), ll(mg))
        np.testing.assert_allclose(sep.groupby("race_id")["p_win"].sum(), 1.0, atol=1e-6)

    def test_fallback_uses_base(self):
        d = self._data(races=200)
        X, meta = d[["x"]], d[["race_id", "horse_id", "year", "finish_position"]]
        yrs = d["year"].to_numpy()
        folds = [SimpleNamespace(train_index=np.flatnonzero(yrs < 2023), valid_index=np.flatnonzero(yrs == 2023))]
        base = meta.loc[yrs == 2023, ["race_id", "horse_id"]].assign(p_win=1 / 8, p_top2=2 / 8, p_top3=3 / 8)
        out, _ = sm.oof_segmented(X, meta, folds, d["seg"].to_numpy(), {"learning_rate": 0.1, "num_leaves": 4}, 10, 2, min_rows=10 ** 9, base_oof=base)
        np.testing.assert_allclose(out["p_win"], 1 / 8, atol=1e-6)


from src.pipeline.models import win_models as wm  # noqa: E402

if __name__ == "__main__":
    unittest.main()
