"""特徴量ファクトリー G4 の候補群が、未来の結果を変えても過去の行の値を変えないこと（先読みなし）。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pipeline.features import factory_embed as fe
from src.pipeline.features import factory_ped as fp_
from src.pipeline.features import factory_src as fs


def _toy(n_races: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    horses = [f"20180{i:05d}" for i in range(30)]
    rows = []
    for r in range(n_races):
        d = pd.Timestamp("2020-01-05") + pd.Timedelta(days=9 * r)
        field = list(rng.choice(horses, size=8, replace=False))
        fin = rng.permutation(8) + 1
        for k, (h, f) in enumerate(zip(field, fin)):
            rows.append({"race_id": f"R{r:05d}", "horse_id": h, "date": d.strftime("%Y-%m-%d"), "venue_code": "05" if r % 2 else "06", "surface": "芝" if r % 3 else "ダート",
                         "distance": int(rng.choice([1200, 1600, 2000])), "track_condition": "良", "finish_position": int(f), "field_size": 8, "horse_number": k + 1,
                         "jockey_id": f"j{k % 4}", "trainer_id": f"t{k % 3}"})
    B = pd.DataFrame(rows)
    return B.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)


def _shuffle_late(B: pd.DataFrame, cut: str) -> pd.DataFrame:
    B2 = B.copy()
    late = (B2["date"] >= cut).to_numpy()
    rng = np.random.default_rng(1)
    fp = B2["finish_position"].to_numpy().copy()
    fp[late] = rng.permutation(fp[late])
    B2["finish_position"] = fp
    return B2


class NoLeakTest(unittest.TestCase):
    CUT = "2021-06-01"

    def _early_equal(self, fn):
        B = _toy()
        a, b = fn(B), fn(_shuffle_late(B, self.CUT))
        early = (B["date"] < self.CUT).to_numpy()
        np.testing.assert_allclose(a[early].to_numpy(dtype=float), b[early].to_numpy(dtype=float), atol=1e-9, equal_nan=True)

    def test_asof_rates(self):
        self._early_equal(lambda B: fs.asof_rates(B, B["jockey_id"], "jk"))

    def test_history_aggs_external_values_shift(self):
        B = _toy()
        v = B[["race_id", "horse_id"]].assign(x=np.arange(len(B), dtype=float))
        f = fs.history_aggs(B, v, ["x"], "h_")
        # 各馬の最初の出走は履歴なし。2 走目の直近値は 1 走目の値
        first = B.groupby("horse_id").head(1).index
        self.assertTrue(f.loc[first, "h_x_last"].isna().all())
        h = B["horse_id"].iloc[0]
        idx = B.index[B["horse_id"] == h]
        self.assertEqual(float(f.loc[idx[1], "h_x_last"]), float(idx[0]))
        self.assertEqual(float(f.loc[idx[2], "h_x_best"]), float(idx[1]))

    def test_dam_sib_excludes_self_and_future(self):
        B = _toy()
        A = pd.DataFrame([(h, g, p, f"a_{g}_{p}_{int(h[-2:]) % 3}") for h in B["horse_id"].unique() for g, p in ((1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (2, 3))],
                         columns=["horse_id", "gen", "pos", "anc_id"])
        f1 = fp_.ped_dam_sib(B, A, None)
        f2 = fp_.ped_dam_sib(_shuffle_late(B, self.CUT), A, None)
        early = (B["date"] < self.CUT).to_numpy()
        np.testing.assert_allclose(f1[early].to_numpy(dtype=float), f2[early].to_numpy(dtype=float), atol=1e-9, equal_nan=True)

    def test_anc_apt_no_leak(self):
        B = _toy()
        A = pd.DataFrame([(h, g, p, f"a_{g}_{p}_{int(h[-2:]) % 4}") for h in B["horse_id"].unique() for g, p in ((1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (2, 3))],
                         columns=["horse_id", "gen", "pos", "anc_id"])
        f1 = fp_.ped_anc_apt(B, A, None, max_gen=2)
        f2 = fp_.ped_anc_apt(_shuffle_late(B, self.CUT), A, None, max_gen=2)
        early = (B["date"] < self.CUT).to_numpy()
        np.testing.assert_allclose(f1[early].to_numpy(dtype=float), f2[early].to_numpy(dtype=float), atol=1e-9, equal_nan=True)


class EmbeddingYearTest(unittest.TestCase):
    def test_first_year_nan_and_no_future(self):
        B = _toy(400)
        Z = fe.entity_context_svd(B, B["jockey_id"], 2, "e")
        y = pd.to_datetime(B["date"]).dt.year
        self.assertTrue(Z[y == y.min()].isna().all().all())
        B2 = _shuffle_late(B, "2022-01-01")
        Z2 = fe.entity_context_svd(B2, B2["jockey_id"], 2, "e")
        early = (y <= 2021).to_numpy()
        np.testing.assert_allclose(np.abs(Z[early].to_numpy(dtype=float)), np.abs(Z2[early].to_numpy(dtype=float)), atol=1e-6, equal_nan=True)


class MarketColumnsTest(unittest.TestCase):
    def test_market_pattern(self):
        for c in ("est_pop", "h1_pop", "pop_tan", "odds_tan", "odds_fuku_low", "popularity"):
            self.assertTrue(fs.MARKET_PAT.search(c) or c == "popularity", c)
        for c in ("owner", "blinker", "rota_type", "h1_stime"):
            self.assertFalse(fs.MARKET_PAT.search(c), c)

    def test_src_cols_have_no_market(self):
        self.assertFalse([c for c in fs.SRC_COLS if fs.MARKET_PAT.search(c)])


if __name__ == "__main__":
    unittest.main()
