"""tp-16 特徴量 bake-off 候補の単体テスト（リーク・定義の確認）。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pipeline.features import bakeoff as bo


def _toy(n_races: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    horses = [f"h{i}" for i in range(12)]
    for r in range(n_races):
        d = pd.Timestamp("2022-01-01") + pd.Timedelta(days=7 * r)
        field = list(rng.choice(horses, size=8, replace=False))
        fin = rng.permutation(8) + 1
        for h, f in zip(field, fin):
            rows.append({"race_id": f"2022{'06' if r % 2 else '08'}01{(r % 8) + 1:02d}{(r % 12) + 1:02d}", "horse_id": h, "date": d, "venue_code": "06" if r % 2 else "08",
                         "surface": "芝" if r % 3 else "ダート", "distance": 1600, "finish_position": int(f), "field_size": 8})
    df = pd.DataFrame(rows)
    df["race_id"] = [f"{x}{i // 8:04d}"[:12] for i, x in enumerate(df["race_id"])]
    df["race_id"] = [f"R{i // 8:05d}" for i in range(len(df))]
    return df


class CalendarTest(unittest.TestCase):
    def test_cyclic(self):
        df = pd.DataFrame({"race_id": ["202206010101", "202206010102"], "date": ["2022-01-01", "2022-12-31"]})
        c = bo.calendar_features(df)
        self.assertAlmostEqual(float(c["cal_sin1"].iloc[0]), 0.0, places=6)
        self.assertLess(abs(float(c["cal_sin1"].iloc[1])), 0.05)          # 年末は年始の近く
        self.assertEqual(int(c["cal_kai"].iloc[0]), 1)


class NoLeakTest(unittest.TestCase):
    """未来の結果を変えても、過去の行の特徴量は変わらない。"""

    def _check(self, fn):
        df = _toy()
        a = fn(df)
        df2 = df.copy()
        late = df2["date"] >= pd.Timestamp("2022-06-01")
        df2.loc[late, "finish_position"] = (df2.loc[late, "finish_position"] % 8) + 1
        b = fn(df2)
        early = (df["date"] < pd.Timestamp("2022-06-01")).to_numpy()
        pd.testing.assert_frame_equal(a[early].reset_index(drop=True), b[early].reset_index(drop=True), check_exact=False, atol=1e-9)

    def test_horse_season(self):
        self._check(bo.horse_season_features)

    def test_aftermath(self):
        self._check(bo.aftermath_features)

    def test_glicko(self):
        self._check(bo.glicko_rd_features)


class GlickoTest(unittest.TestCase):
    def test_rd_shrinks_then_grows(self):
        d = pd.DataFrame({"race_id": [f"r{i}" for i in range(4)], "horse_id": "h", "date": pd.to_datetime(["2022-01-01", "2022-01-08", "2022-01-15", "2023-01-15"]),
                          "finish_position": 1, "field_size": 5})
        g = bo.glicko_rd_features(d)["g_rd"].to_numpy()
        self.assertEqual(g[0], 1.0)
        self.assertLess(g[1], g[0])
        self.assertLess(g[2], g[1])
        self.assertGreater(g[3], g[2])             # 1 年空くと不確実性が戻る


class AftermathTest(unittest.TestCase):
    def test_opponent_next_results(self):
        # r1: A・B が走る。B は r2（2022-02-01）で 1 着、A は r3（2022-03-01）で勝つ。
        d = pd.DataFrame([
            {"race_id": "r1", "horse_id": "A", "date": "2022-01-01", "finish_position": 1, "field_size": 3},
            {"race_id": "r1", "horse_id": "B", "date": "2022-01-01", "finish_position": 2, "field_size": 3},
            {"race_id": "r1", "horse_id": "C", "date": "2022-01-01", "finish_position": 3, "field_size": 3},
            {"race_id": "r2", "horse_id": "B", "date": "2022-02-01", "finish_position": 1, "field_size": 3},
            {"race_id": "r2", "horse_id": "C", "date": "2022-02-01", "finish_position": 3, "field_size": 3},
            {"race_id": "r2", "horse_id": "D", "date": "2022-02-01", "finish_position": 2, "field_size": 3},
            {"race_id": "r3", "horse_id": "A", "date": "2022-03-01", "finish_position": 1, "field_size": 3},
        ])
        d["date"] = pd.to_datetime(d["date"])
        f = bo.aftermath_features(d, lags=1)
        a3 = f.iloc[6]                        # A の r3 時点: 直近走 r1 の相手 B・C の次走 (r2) は結果が分かっている（自分 A は除く）
        self.assertEqual(a3["af_cnt_3"], 2.0)
        r_b, r_c = 0.5, -0.5                  # B は 1 着 → +0.5、C は 3 着 → −0.5
        self.assertAlmostEqual(float(f.iloc[6]["af_cnt_3"]), 2.0)
        self.assertTrue(np.isnan(f.iloc[0]["af_res_3"]))        # 初出走には後日談なし
        b2 = f.iloc[3]                        # B の r2 時点: 相手 A・C の次走は r3 / r2。r1 の相手の次走で、今回(2022-02-01)より前は無い
        self.assertEqual(b2["af_cnt_3"], 0.0)
        self.assertAlmostEqual(r_b + r_c, 0.0)


class CushionTest(unittest.TestCase):
    def test_join_and_surface(self):
        cu = pd.DataFrame({"date": ["2022-04-02", "2022-04-02"], "venue_code": ["06", "09"], "cushion_value": [9.5, 8.0], "turf_moisture_goal": [12.0, 10.0],
                           "turf_moisture_4corner": [11.0, 9.0], "dirt_moisture_goal": [3.0, 4.0], "dirt_moisture_4corner": [2.0, 3.0]})
        df = pd.DataFrame({"race_id": ["a", "b"], "horse_id": ["h1", "h1"], "date": pd.to_datetime(["2022-04-02", "2022-04-02"]), "venue_code": ["06", "06"],
                           "surface": ["芝", "ダート"], "finish_position": [1, 2], "field_size": [5, 5]})
        f = bo.cushion_features(df, cu)
        self.assertEqual(float(f["cu_cushion"].iloc[0]), 9.5)
        self.assertTrue(np.isnan(f["cu_cushion"].iloc[1]))          # ダートにクッション値は無い
        self.assertEqual(float(f["cu_moist_goal"].iloc[1]), 3.0)    # ダートはダートの含水率


class InbreedingTest(unittest.TestCase):
    def _full(self, gens: dict) -> list[dict]:
        out = []
        for (g, p), name in gens.items():
            out.append({"generation": g, "position": p, "horse_id": name, "name": name})
        return out

    def test_full_sibling_mating(self):
        # 父母が全きょうだい（祖父母が共通）→ F = 0.25
        anc = self._full({(1, 0): "S", (1, 1): "D", (2, 0): "GS", (2, 1): "GD", (2, 2): "GS", (2, 3): "GD"})
        self.assertAlmostEqual(bo.inbreeding_from_ancestors(anc)["F"], 0.25)

    def test_half_cousin(self):
        # 父方 3 世代目・母方 3 世代目に同じ祖先 X: F = 0.5^(3+3-1) = 1/32
        anc = self._full({(1, 0): "S", (1, 1): "D", (2, 0): "a", (2, 1): "b", (2, 2): "c", (2, 3): "d", (3, 0): "X", (3, 1): "e", (3, 4): "X", (3, 5): "f"})
        self.assertAlmostEqual(bo.inbreeding_from_ancestors(anc)["F"], 2 ** -5)

    def test_no_common(self):
        anc = self._full({(1, 0): "S", (1, 1): "D", (2, 0): "a", (2, 1): "b", (2, 2): "c", (2, 3): "d"})
        self.assertEqual(bo.inbreeding_from_ancestors(anc)["F"], 0.0)


class OikiriTest(unittest.TestCase):
    def test_days_before(self):
        df = pd.DataFrame({"race_id": ["r1"], "horse_id": ["h"], "date": pd.to_datetime(["2024-07-21"])})
        ok = pd.DataFrame({"race_id": ["r1"], "horse_id": ["h"], "training_date": ["2024/07/18(木)"]})
        self.assertEqual(float(bo.oikiri_timing_features(df, ok)["ok_days_before"].iloc[0]), 3.0)


class VenueEraTest(unittest.TestCase):
    def test_kyoto(self):
        df = pd.DataFrame({"race_id": ["a", "b", "c"], "horse_id": ["h", "h", "h"], "date": pd.to_datetime(["2020-10-01", "2023-05-01", "2023-06-01"]), "venue_code": ["08", "08", "08"]})
        f = bo.venue_era_features(df)
        self.assertEqual(list(f["ve_kyoto_era"]), [1, 2, 2])
        self.assertEqual(list(f["ve_horse_n_kyoto_post"]), [0, 0, 1])


if __name__ == "__main__":
    unittest.main()
