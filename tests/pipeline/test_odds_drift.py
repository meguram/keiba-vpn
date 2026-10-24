"""オッズのずれ（late money）の監視部品のテスト。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pipeline.betting import odds_drift as od


def _doc(rid, final, early, at_final="2026-10-17T15:40:00+09:00", at_early="2026-10-17T15:25:00+09:00"):
    return {"race_id": rid, "phase": "final", "snapshot_at": at_final, "entries": [{"horse_number": i + 1, "win_odds": o, "place_odds_min": max(o / 4, 1.1), "popularity": i + 1} for i, o in enumerate(final)],
            "snapshots": [{"phase": "t15", "snapshot_at": at_early, "entries": [{"horse_number": i + 1, "win_odds": o, "place_odds_min": max(o / 4, 1.1), "popularity": i + 1} for i, o in enumerate(early)]}]}


class OddsDriftTest(unittest.TestCase):
    def test_flatten_and_pair(self):
        d = _doc("r1", [2.0, 8.0, -3.0], [2.5, 10.0, -3.0])
        flat = od.flatten_docs([d])
        self.assertFalse((flat["win_odds"] <= 0).any())              # 取消・除外は除く
        m = od.pair_phases(flat, early="t15")
        self.assertEqual(len(m), 2)
        np.testing.assert_allclose(m.sort_values("horse_number")["log_ratio"], np.log([2.0 / 2.5, 8.0 / 10.0]))

    def _many(self, shrink_longshot: float, n=400, seed=0):
        rng = np.random.default_rng(seed)
        docs = []
        for r in range(n):
            early = np.exp(rng.uniform(np.log(1.5), np.log(120), 10))
            final = early * np.where(early > 20, shrink_longshot, 1.0) * np.exp(rng.normal(0, 0.05, 10))
            docs.append(_doc(f"r{r}", final.round(1), early.round(1)))
        return docs

    def test_haircut_reflects_longshot_compression(self):
        m = od.pair_phases(od.flatten_docs(self._many(0.7)), early="t15")
        tab = od.haircut_table(m, min_n=50, shrink=1.0)
        self.assertLess(tab["50+"], 0.8)
        self.assertGreater(tab["<3"], 0.9)
        adj = od.apply_haircut(np.array([2.0, 60.0]), tab)
        self.assertAlmostEqual(adj[0], 2.0 * tab["<3"])
        self.assertLess(adj[1], 60.0)

    def test_drift_alert_detects_shift(self):
        base = od.pair_phases(od.flatten_docs(self._many(0.95, seed=1)), early="t15")
        recent = od.pair_phases(od.flatten_docs(self._many(0.7, seed=2)), early="t15")
        a = od.drift_alert(recent, base, min_n=50)
        self.assertTrue(bool(a[a["band"] == "50+"]["警報"].iloc[0]))
        same = od.drift_alert(base, od.pair_phases(od.flatten_docs(self._many(0.95, seed=3)), early="t15"), min_n=50)
        self.assertFalse(bool(same["警報"].any()))


if __name__ == "__main__":
    unittest.main()
