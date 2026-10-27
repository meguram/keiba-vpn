"""想定めぐ指数の階層縮約（HEM）: 先読みが無いこと・走数が少ないほど縮約されること。"""

import unittest

import numpy as np
import pandas as pd

from src.scripts.data import hem_megu as hm


def _synthetic(seed: int = 0, n_horse: int = 120, runs: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for hid in range(n_horse):
        ability = rng.normal(0, 8)
        jk = f"J{hid % 7}"
        for k in range(runs):
            d = 20210101 + k * 70 + (hid % 5)          # 馬ごとに日付が少しずれる
            rows.append({"race_id": f"R{hid}_{k}", "horse_id": f"H{hid}", "jockey_id": jk, "jw": 55.0, "dnum": d, "surf": int(k % 2), "dist": 1200.0 + 200 * (k % 3),
                         "lvl": 2, "age": 3, "y": 40 + ability + rng.normal(0, 10), "low": False})
    h = pd.DataFrame(rows)
    h["year"] = h["dnum"] // 10000
    return h


class HemTests(unittest.TestCase):
    def setUp(self):
        self.h = _synthetic()
        self.cells = {"glob": 40.0, "fine": {}, "mid": {}}
        self.p = dict(hm.DEFAULT)

    def _feat(self, h: pd.DataFrame, T: pd.DataFrame) -> pd.DataFrame:
        return hm.features(hm.Hist(h, self.cells), T, self.p)

    def test_no_lookahead_future_values_do_not_change_earlier_rows(self):
        T = hm.targets_of(self.h)
        base = self._feat(self.h, T)
        h2 = self.h.copy()
        cut = 20210301
        h2.loc[h2["dnum"] >= cut, "y"] += 500.0                  # 日付が cut 以降の指数を壊す
        shifted = self._feat(h2, T)
        early = (T["dnum"] <= cut).to_numpy()                     # cut 以前（cut 当日の行は cut 当日の走を使わない）
        for c in ("dhat", "us", "ud", "jock", "wmean", *hm.CHG_COLS):
            a, b = base.loc[early, c].to_numpy(), shifted.loc[early, c].to_numpy()
            self.assertTrue(np.allclose(a, b, equal_nan=True), c)

    def test_first_run_has_no_history(self):
        T = hm.targets_of(self.h)
        F = self._feat(self.h, T)
        first = self.h.groupby("horse_id")["dnum"].transform("min") == self.h["dnum"]
        self.assertTrue((F.loc[first, "n"] == 0).all())
        self.assertTrue((F.loc[first, "dhat"] == 0).all())

    def test_fewer_runs_shrink_more(self):
        # 全走が同じ偏差(+20)の馬: 走数が多いほど d̂ は +20 に近い
        rows = []
        for n in (1, 2, 5):
            for k in range(n):
                rows.append({"race_id": f"A{n}_{k}", "horse_id": f"A{n}", "jockey_id": "J", "jw": 55.0, "dnum": 20210101 + k * 30, "surf": 1, "dist": 1600.0, "lvl": 2, "age": 3, "y": 60.0, "low": False})
            rows.append({"race_id": f"A{n}_t", "horse_id": f"A{n}", "jockey_id": "J", "jw": 55.0, "dnum": 20211201, "surf": 1, "dist": 1600.0, "lvl": 2, "age": 3, "y": 60.0, "low": False})
        h = pd.DataFrame(rows)
        h["year"] = h["dnum"] // 10000
        F = hm.features(hm.Hist(h, self.cells), hm.targets_of(h), self.p)
        last = h["race_id"].str.endswith("_t").to_numpy()
        d = dict(zip(h.loc[last, "horse_id"], F.loc[last, "dhat"]))
        self.assertLess(d["A1"], d["A2"])
        self.assertLess(d["A2"], d["A5"])
        self.assertLess(d["A5"], 20.0)                            # 縮約されているので偏差そのもの(+20)にはならない

    def test_jockey_effect_uses_strictly_earlier_dates(self):
        h = self.h
        T = hm.targets_of(h)
        F = self._feat(h, T)
        jt = hm.jockey_table(hm.Hist(h, self.cells), self.p)
        # 全体で最も早い日付の行は、騎手の過去騎乗が無いので効果 0
        first_day = T["dnum"] == T["dnum"].min()
        self.assertTrue((F.loc[first_day, "jock"] == 0).all())
        self.assertGreater(len(jt), 0)


if __name__ == "__main__":
    unittest.main()
