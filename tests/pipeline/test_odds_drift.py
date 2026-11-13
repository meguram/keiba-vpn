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


def test_race_ids_between_uses_local_records_without_listing(tmp_path):
    """週次の集計の対象は、ローカルの flat と race_lists から決める（GCS の一覧と全文書の読み出しをやめた。2026-11-13）。"""
    import json

    import pandas as pd

    from src.pipeline.betting.odds_drift import race_ids_between

    d = tmp_path / "tables" / "2026"
    d.mkdir(parents=True)
    pd.DataFrame({"race_id": ["202605010101", "202605010201"], "date": ["2026-10-10", "2026-10-20"]}).to_parquet(d / "race_result_flat.parquet")
    (tmp_path / "race_lists").mkdir()
    (tmp_path / "race_lists" / "20261025.json").write_text(json.dumps({"races": [{"race_id": "202605020101"}]}))
    assert race_ids_between("20261011", "20261031", tmp_path / "tables") == ["202605010201", "202605020101"]
