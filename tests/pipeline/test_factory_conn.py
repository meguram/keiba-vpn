"""ファクトリー G2（関係者の as-of 特徴量）: 手計算との一致と、将来の結果を変えても過去行が変わらないこと。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pipeline.features import factory_conn as fc


def _toy(n_days: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    horses = [f"h{i}" for i in range(14)]
    jockeys = [f"j{i}" for i in range(5)]
    trainers = [f"t{i}" for i in range(4)]
    owners = [f"o{i}" for i in range(3)]
    for d in range(n_days):
        date = pd.Timestamp("2022-01-01") + pd.Timedelta(days=7 * d)
        for r in range(1, 5):
            field = list(rng.choice(horses, size=8, replace=False))
            fin = rng.permutation(8) + 1
            for k, (hid, f) in enumerate(zip(field, fin)):
                rows.append({"race_id": f"R{d:03d}{r:02d}", "date": date, "venue_code": "05" if d % 2 else "06", "surface": "芝" if r % 2 else "ダート", "distance": 1600,
                             "grade": "", "race_class": "1勝クラス", "track_condition": "良", "start_time": f"{9 + 2 * r:02d}:{r * 5:02d}", "field_size": 8, "round": r,
                             "finish_position": int(f), "bracket_number": k + 1, "horse_number": k + 1, "horse_id": hid,
                             "jockey_id": jockeys[hash(hid + str(d)) % 5], "trainer_id": trainers[int(hid[1:]) % 4], "jockey_weight": 55.0 + (k % 3),
                             "last_3f": 34.0 + rng.random() * 3, "passing_order": f"{k + 1}-{k + 1}", "weight": 480, "sex_age": "牡3",
                             "owner": owners[int(hid[1:]) % 3], "breeder": "B" + str(int(hid[1:]) % 2), "birthplace": "安平町"})
    return pd.DataFrame(rows)


class FactoryConnTest(unittest.TestCase):
    def test_entity_rate_matches_bruteforce(self):
        h = fc.prepare(_toy())
        eng = fc.Engine(h)
        f = fc.entity_features(eng, "j", [])
        i = len(h) - 5
        row = h.iloc[i]
        past = h[(h["jockey_id"] == row["jockey_id"]) & (h["date"] < row["date"])]
        g = h[h["date"] < row["date"]]
        for v in ("win", "res"):
            prior = g[v].mean()
            exp = (past[v].sum() + fc.K_ENT * prior) / (len(past) + fc.K_ENT)
            self.assertAlmostEqual(float(f[f"j_{v}"][i]), exp, places=4)
        self.assertAlmostEqual(float(f["j_n"][i]), np.log1p(len(past)), places=5)

    def test_window_and_streak(self):
        h = fc.prepare(_toy())
        eng = fc.Engine(h)
        w = fc.window_features(eng, "t", windows=(30,))
        st = fc.streak_features(eng, "t")
        i = len(h) - 3
        row = h.iloc[i]
        past = h[(h["trainer_id"] == row["trainer_id"]) & (h["date"] < row["date"])]
        win = past[past["date"] >= row["date"] - pd.Timedelta(days=30)]
        self.assertAlmostEqual(float(w["t_w30_n"][i]), len(win))
        wins = past[past["win"] == 1]
        self.assertEqual(float(st["t_days_since_win"][i]), min((row["date"] - wins["date"].max()).days, 730))
        self.assertEqual(float(st["t_days_since_ride"][i]), min((row["date"] - past["date"].max()).days, 365))

    def test_same_day_uses_only_races_90min_earlier(self):
        h = fc.prepare(_toy())
        f = fc.same_day_features(h)
        i = 200
        row = h.iloc[i]
        t = fc._minutes(h["start_time"])
        same = h[(h["jockey_id"] == row["jockey_id"]) & (h["date"] == row["date"]) & (t <= t[i] - 90) & h["ok"]]
        self.assertEqual(float(f["j_sd_rides_before"][i]), len(same))

    def test_no_future_leak(self):
        df = _toy()
        h1 = fc.prepare(df)
        g1 = fc.build_all(h1, entities=["j", "t", "o", "jt", "jh"], conds=["surf", "lay"])
        df2 = df.copy()
        late = df2["date"] >= pd.Timestamp("2022-06-01")
        df2.loc[late, "finish_position"] = (df2.loc[late, "finish_position"] % 8) + 1
        df2.loc[late, "last_3f"] = 33.0
        g2 = fc.build_all(fc.prepare(df2), entities=["j", "t", "o", "jt", "jh"], conds=["surf", "lay"])
        early = (h1["date"] < pd.Timestamp("2022-06-01")).to_numpy()
        n = 0
        for grp, cols in g1.items():
            for c, a in cols.items():
                b = g2[grp][c]
                np.testing.assert_allclose(np.asarray(a)[early], np.asarray(b)[early], rtol=1e-5, atol=1e-6, equal_nan=True, err_msg=f"{grp}.{c}")
                n += 1
        self.assertGreater(n, 100)


if __name__ == "__main__":
    unittest.main()
