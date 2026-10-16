"""v3 推論用の相対化・前段モデル・tl_* 列（serving_stage_rel）の単体テスト。"""

from __future__ import annotations

import tempfile
import unittest

import numpy as np
import pandas as pd

from src.pipeline.features import serving_stage_rel as sr
from src.pipeline.models import win_models as wm


def _toy(n_races: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        n = int(rng.integers(1, 12))
        for k in range(n):
            rows.append({"race_id": f"R{r:03d}", "horse_id": f"H{r}_{k}", "a": rng.normal(), "b": float(rng.integers(0, 3)), "c": np.nan if rng.random() < 0.2 else rng.normal()})
    return pd.DataFrame(rows).sample(frac=1.0, random_state=1).reset_index(drop=True)


def _old_formula(df: pd.DataFrame, cols, suffix=("_rz", "_rrank", "_rgap")) -> pd.DataFrame:
    """変更前の win_models.add_race_relative と同じ式（そのまま写したもの）。"""
    g = df.groupby("race_id", sort=False)
    out = {}
    for c in cols:
        x = df[c].astype("float64")
        mu, sd = g[c].transform("mean"), g[c].transform("std")
        out[c + suffix[0]] = ((x - mu) / sd.replace(0, np.nan)).astype("float32")
        out[c + suffix[1]] = g[c].rank(ascending=False, method="average").astype("float32")
        out[c + suffix[2]] = (x - g[c].transform("max")).astype("float32")
    return pd.DataFrame(out, index=df.index)


class RaceRelativeTest(unittest.TestCase):
    def test_same_as_previous_formula(self):
        df = _toy()
        pd.testing.assert_frame_equal(wm.add_race_relative(df, ["a", "b", "c"]), _old_formula(df, ["a", "b", "c"]))

    def test_c2_two_parts_equal_old_c2(self):
        df = _toy()
        new = sr.race_relative(df, ["a", "c"], ("_rz", "_rrank"))
        old = _old_formula(df, ["a", "c"])
        self.assertEqual(list(new.columns), ["a_rz", "a_rrank", "c_rz", "c_rrank"])
        pd.testing.assert_frame_equal(new, old[new.columns])

    def test_by_names_order_and_values(self):
        df = _toy()
        names = ["c_rgap", "a_rz", "a_rrank", "b_rz"]
        got = sr.relative_by_names(df, names)
        self.assertEqual(list(got.columns), names)
        full = _old_formula(df, ["a", "b", "c"])
        pd.testing.assert_frame_equal(got, full[names])

    def test_independent_of_row_order_and_other_races(self):
        df = _toy()
        a = sr.relative_by_names(df, ["a_rz", "a_rrank", "a_rgap"])
        one = df[df["race_id"] == "R003"]
        b = sr.relative_by_names(one, ["a_rz", "a_rrank", "a_rgap"])
        pd.testing.assert_frame_equal(a.loc[b.index], b)             # そのレースの馬だけで同じ値になる（推論は 1 レースずつでよい）

    def test_bad_name(self):
        with self.assertRaises(ValueError):
            sr.relative_by_names(_toy(), ["a_mean"])
        with self.assertRaises(KeyError):
            sr.relative_by_names(_toy(), ["zzz_rz"])

    def test_single_horse_race_is_nan_z(self):
        df = pd.DataFrame({"race_id": ["r", "s", "s"], "a": [1.0, 1.0, 3.0]})
        got = sr.race_relative(df, ["a"])
        self.assertTrue(np.isnan(got.loc[0, "a_rz"]))
        self.assertEqual(float(got.loc[0, "a_rrank"]), 1.0)


class StageModelsTest(unittest.TestCase):
    def _data(self, n=1500, seed=0):
        rng = np.random.default_rng(seed)
        X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n),
                          "surface": pd.Categorical(rng.choice(["芝", "ダート", "(なし)"], size=n))})
        race = np.repeat(np.arange(n // 10), 10)
        y1 = X["x1"].to_numpy() * 2 + rng.normal(size=n) * 0.1
        y2 = 1 / (1 + np.exp(-X["x2"].to_numpy()))
        return X, race, {"s_time_pred": y1, "s_rank_pred": y2}

    def test_roundtrip_and_gap(self):
        X, race, tg = self._data()
        m = sr.StageModels.fit(X, tg, threads=2, n_rounds=30, params={"min_data_in_leaf": 20})
        p = m.predict(X, race)
        self.assertEqual(list(p.columns), sr.STAGE_COLS)
        self.assertGreater(np.corrcoef(p["s_time_pred"], tg["s_time_pred"])[0, 1], 0.9)
        g = p.groupby(race)["s_time_pred"].transform("min")
        np.testing.assert_allclose(p["s_time_gap"], p["s_time_pred"] - g, atol=1e-5)
        self.assertTrue((p["s_time_gap"] >= -1e-6).all())
        with tempfile.TemporaryDirectory() as d:
            m.save(d)
            m2 = sr.StageModels.load(d)
            q = m2.predict(X, race)
            pd.testing.assert_frame_equal(p, q)

    def test_object_dtype_and_missing_category_are_cast(self):
        X, race, tg = self._data()
        m = sr.StageModels.fit(X, tg, threads=2, n_rounds=20, params={"min_data_in_leaf": 20})
        Xo = X.copy()
        Xo["surface"] = Xo["surface"].astype(str)               # 推論の入力は category とは限らない
        pd.testing.assert_frame_equal(m.predict(X, race), m.predict(Xo, race))
        Xn = X.copy()
        Xn["surface"] = pd.Series(Xn["surface"].astype(str)).where(np.arange(len(X)) % 2 == 0, None)   # 欠損 → 「(なし)」
        Xm = X.copy()
        Xm["surface"] = pd.Series(Xm["surface"].astype(str)).where(np.arange(len(X)) % 2 == 0, "(なし)")
        pd.testing.assert_frame_equal(m.predict(Xn, race), m.predict(Xm, race))

    def test_missing_column(self):
        X, race, tg = self._data()
        m = sr.StageModels.fit(X, tg, threads=2, n_rounds=10, params={"min_data_in_leaf": 20})
        with self.assertRaises(KeyError):
            m.predict(X.drop(columns=["x1"]), race)


class TlColumnsTest(unittest.TestCase):
    def test_no_records_gives_nan_rows_with_keys(self):
        races = pd.DataFrame({"race_id": ["202506010101"] * 2, "horse_id": ["h1", "h2"], "date": ["2025-01-05", "2025-01-05"]})
        out = sr.tl_columns_for_races(races, {"h1": None, "h2": None}, pd.DataFrame())
        self.assertEqual(list(out["horse_id"]), ["h1", "h2"])
        self.assertEqual(list(out.columns), ["race_id", "horse_id"])


if __name__ == "__main__":
    unittest.main()
