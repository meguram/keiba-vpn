"""tp-32: 馬具と『初めて・替わり』系の状況特徴。先読みが無いこと（将来の結果を変えても過去行が変わらない）と、定義の確認。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pipeline.features import tp32_equipment as te
from tests.pipeline.test_factory_self import _late_modified, _toy


def _eq_table(df: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    e = df[["horse_id", "race_id", "date", "finish_position", "field_size"]].copy()
    e["rdate"] = e["date"].dt.strftime("%Y%m%d")
    e["blinker"] = [str(int(x)) for x in rng.integers(0, 2, len(e))]
    e["rank"] = e["finish_position"].astype(str)
    e["entry"] = e["field_size"].astype(str)
    return e[["horse_id", "race_id", "rdate", "blinker", "rank", "entry"]]


def _eq_late_modified(e: pd.DataFrame, cut: str) -> pd.DataFrame:
    e2 = e.copy()
    late = pd.to_datetime(e2["rdate"], format="%Y%m%d") >= pd.Timestamp(cut)
    e2.loc[late, "rank"] = ((e2.loc[late, "rank"].astype(int) % 8) + 1).astype(str)
    return e2


class NoLeakTest(unittest.TestCase):
    def test_every_group_is_as_of(self):
        df = _toy()
        cut = "2021-06-01"
        E = _eq_table(df)
        a = te.build_groups(df, E=E)
        b = te.build_groups(_late_modified(df, cut), E=_eq_late_modified(E, cut))
        self.assertEqual(set(a), set(te.GROUPS))
        early = (pd.to_datetime(df["date"]) < pd.Timestamp(cut)).to_numpy()
        for g in a:
            fa, fb = a[g], b[g]
            self.assertEqual(len(fa), len(df))
            cols = [c for c in fa.columns if c not in ("race_id", "horse_id")]
            np.testing.assert_allclose(fa.loc[early, cols].to_numpy(float), fb.loc[early, cols].to_numpy(float), atol=1e-5, equal_nan=True, err_msg=g)

    def test_same_day_results_are_not_used(self):
        # 同じ日の別レースの結果を変えても、その日の行の特徴量は変わらない（日付が厳密に前だけ）
        df = _toy()
        day = df["date"].sort_values().iloc[len(df) // 2]
        d2 = df.copy()
        m = d2["date"] == day
        d2.loc[m, "finish_position"] = (d2.loc[m, "finish_position"] % 8) + 1
        a, b = te.build_groups(df, E=None), te.build_groups(d2, E=None)
        for g in a:
            cols = [c for c in a[g].columns if c not in ("race_id", "horse_id")]
            np.testing.assert_allclose(a[g].loc[m.to_numpy(), cols].to_numpy(float), b[g].loc[m.to_numpy(), cols].to_numpy(float), atol=1e-5, equal_nan=True, err_msg=g)


class DefinitionTest(unittest.TestCase):
    def test_first_blinker_and_off(self):
        rows = []
        for i, (b, r) in enumerate([("0", 3), ("0", 5), ("1", 2), ("1", 4), ("0", 6)]):
            rows.append({"horse_id": "h1", "race_id": f"r{i}", "rdate": f"2022010{i + 1}", "blinker": b, "rank": str(r), "entry": "10"})
        E = pd.DataFrame(rows)
        runs = pd.DataFrame({"race_id": E["race_id"], "horse_id": "h1", "date": pd.to_datetime(E["rdate"]), "start_time": "10:00", "venue_code": "05", "surface": "芝",
                             "distance": 1600.0, "direction": "右", "grade": "1勝", "race_class": "x", "weather": "晴", "track_condition": "良", "field_size": 10, "round": 1,
                             "bracket_number": 1, "horse_number": 1, "horse_name": "x", "sex_age": "牡4", "jockey_id": "j", "trainer_id": "t", "jockey_weight": 56.0,
                             "weight": 480, "weight_change": 0, "finish_position": E["rank"].astype(int), "margin": "", "passing_order": "1-1", "last_3f": 35.0, "time_sec": 96.0})
        f = te.build_groups(runs, E=E, only=["EQ"])["EQ"].reset_index(drop=True)
        self.assertEqual(list(f["eq_first"]), [0.0, 0.0, 1.0, 0.0, 0.0])
        self.assertEqual(list(f["eq_off"]), [0.0, 0.0, 0.0, 0.0, 1.0])
        self.assertEqual(list(f["eq_on_cont"]), [0.0, 0.0, 0.0, 1.0, 0.0])
        self.assertEqual(f["eq_n_on"].tolist(), [0.0, 0.0, 0.0, 1.0, 2.0])

    def test_distance_tiers_and_surface_switch(self):
        df = _toy()
        g = te.build_groups(df, only=["DIST", "SURF"])
        d = g["DIST"]
        self.assertTrue(d["dc_tier"].dropna().between(-3, 3).all())
        s = g["SURF"]
        self.assertTrue(((s["sf_t2d"].fillna(0) + s["sf_d2t"].fillna(0)) <= s["sf_switch"].fillna(0) + 1e-6).all())


if __name__ == "__main__":
    unittest.main()
