"""tp-30: 荒れ度の事後指標・事前入力（先読み検査）・温度調整・買い目の組み方。"""

from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from src.pipeline.features import tp30_volatility as tv


def _toy_table(n_races: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        n = 10
        for h in range(n):
            rows.append({"race_id": f"R{r:02d}", "horse_id": f"h{r}_{h}", "date": pd.Timestamp("2024-01-06") + pd.Timedelta(days=7 * r), "field_size": n, "distance": 1600,
                         "round": r + 1, "horse_prior_races": int(rng.integers(0, 5)), "surface": "芝", "venue_code": "05", "grade": "未勝利", "race_class": "x",
                         "weather": "晴", "track_condition": "良", "a_style": float(rng.random()),
                         **{c: float(rng.normal()) for c in tv.AGG_COLS}, "finish_position": h + 1, "popularity": h + 1, "odds": 2.0 + h})
    return pd.DataFrame(rows)


class ParseTest(unittest.TestCase):
    def test_list_and_concatenated_forms(self):
        ok = json.dumps({"単勝": {"numbers": "5", "payout": "120"}, "複勝": [{"numbers": "5", "payout": "100"}, {"numbers": "2", "payout": "240"}],
                         "ワイド": [{"numbers": "2 - 5", "payout": "190"}, {"numbers": "3 - 5", "payout": "150"}], "三連複": {"numbers": "2 - 3 - 5", "payout": "1,620"}})
        p = tv.parse_payoff(ok)
        self.assertEqual(p["tansho"], ((5,), 120))
        self.assertEqual(len(p["fukusho"]), 2)
        self.assertEqual(p["sanrenpuku"], ((2, 3, 5), 1620))
        bad = json.dumps({"複勝": {"numbers": "3168", "payout": "330120540"}, "ワイド": {"numbers": "3 - 163 - 88 - 16", "payout": "6503,460840"}})
        q = tv.parse_payoff(bad)
        self.assertEqual(q["fukusho"], [])          # 連結されて分解できないセルは空にする（桁あふれの払戻を使わない）
        self.assertEqual(q["wide"], [])


class PosthocTest(unittest.TestCase):
    def test_favourites_win_is_calm(self):
        rows = []
        for h in range(10):
            rows.append({"race_id": "A", "horse_id": f"a{h}", "finish_position": h + 1, "popularity": h + 1, "payoff": json.dumps({"単勝": {"numbers": "1", "payout": "150"}, "三連複": {"numbers": "1 - 2 - 3", "payout": "300"}})})
        for h in range(10):
            rows.append({"race_id": "B", "horse_id": f"b{h}", "finish_position": 10 - h, "popularity": h + 1, "payoff": json.dumps({"単勝": {"numbers": "10", "payout": "9000"}, "三連複": {"numbers": "8 - 9 - 10", "payout": "90000"}})})
        ph = tv.posthoc(pd.DataFrame(rows)).set_index("race_id")
        self.assertEqual(ph.loc["A", "win_pop"], 1)
        self.assertEqual(ph.loc["B", "win_pop"], 10)
        self.assertEqual(ph.loc["A", "fav3_all_top3"], 1)
        self.assertGreater(ph.loc["A", "rho"], 0.9)
        self.assertLess(ph.loc["B", "rho"], -0.9)
        self.assertEqual(ph.loc["B", "n_long_top3"], 3)


class NoLeakTest(unittest.TestCase):
    def test_context_ignores_results_and_other_races(self):
        a = _toy_table()
        ctx = tv.race_context(a)
        bad = [c for c in ctx.columns if any(k in c for k in ("pop", "odds", "finish", "mk_"))]
        self.assertEqual(bad, [])                    # 事前入力に人気・オッズ・着順の列が無い
        b = a.copy()
        b["finish_position"] = (b["finish_position"] % 10) + 1
        b["popularity"] = 99
        b["odds"] = 1.0
        b.loc[b["race_id"] == "R03", AGG := tv.AGG_COLS[0]] = 123.0          # 別のレースの値を変える
        ctx2 = tv.race_context(b)
        keep = ctx["race_id"] != "R03"
        pd.testing.assert_frame_equal(ctx[keep].reset_index(drop=True), ctx2[keep].reset_index(drop=True))


class TemperTest(unittest.TestCase):
    def test_sum_and_identity(self):
        p = np.array([0.4, 0.3, 0.2, 0.1])
        self.assertTrue(np.allclose(tv.temper(p, 1.0, 1.0), p, atol=1e-6))
        for k, base in ((1.0, p), (2.0, np.array([0.8, 0.6, 0.4, 0.2]))):
            q = tv.temper(base, 1.7, k)
            self.assertAlmostEqual(q.sum(), k, places=5)
        flat = tv.temper(p, 3.0, 1.0)
        self.assertLess(flat.max(), p.max())          # tau が大きいと平らになる

    def test_batch_matches_single(self):
        rid = np.array(["a"] * 4 + ["b"] * 3)
        pw = np.array([0.4, 0.3, 0.2, 0.1, 0.5, 0.3, 0.2])
        r, mask, (P,), pos = tv.pad_races(rid, pw)
        Q = tv.temper_batch(P, mask, np.array([1.5, 0.8]), 1.0)
        out = np.zeros(len(pw))
        out[pos[mask]] = Q[mask]
        self.assertTrue(np.allclose(out[:4], tv.temper(pw[:4], 1.5, 1.0), atol=1e-5))
        self.assertTrue(np.allclose(out[4:], tv.temper(pw[4:], 0.8, 1.0), atol=1e-5))


class CombosTest(unittest.TestCase):
    def test_sizes(self):
        self.assertEqual(len(tv.box_combos(list(range(5)))), 10)
        self.assertEqual(len(tv.nagashi_combos([0], [0, 1, 2, 3, 4])), 6)       # 軸 + 相手 4 頭から 2 頭
        f = tv.formation_combos([[0, 1], [0, 1, 2, 3], [0, 1, 2, 3, 4]])
        self.assertTrue(all(len(set(t)) == 3 for t in f))
        self.assertIn((0, 1, 4), f)
        self.assertNotIn((2, 3, 4), f)               # 1 列目の馬を含まない組は買えない


if __name__ == "__main__":
    unittest.main()
