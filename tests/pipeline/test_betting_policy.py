"""馬券環境（ticket_env）と方策（policy）の単体テスト。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pipeline.betting import policy as pol
from src.pipeline.betting import ticket_env as te


def _race(rid: str, n: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.ones(n) * 2)
    t2 = np.clip(w * 2 + 0.05, 0, 1)
    t3 = np.clip(w * 3 + 0.1, 0, 1)
    pred = pd.DataFrame({"race_id": rid, "horse_id": [f"h{i}" for i in range(n)], "horse_number": np.arange(1, n + 1), "p_win": w, "p_top2": t2 / t2.sum() * 2, "p_top3": t3 / t3.sum() * 3})
    odds = 0.8 / w
    wp = pd.DataFrame({"race_id": rid, "horse_id": pred["horse_id"], "horse_number": np.arange(1, n + 1), "win_odds": odds, "place_odds_min": np.maximum(odds / 3, 1.1), "place_odds_max": odds / 2})
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            rows.append({"race_id": rid, "bet_type": "umaren", "combo": f"{i + 1}-{j + 1}", "odds": 0.775 / (w[i] * w[j] * 2), "odds_min": np.nan})
            rows.append({"race_id": rid, "bet_type": "wide", "combo": f"{i + 1}-{j + 1}", "odds": np.nan, "odds_min": 1.5 + i + j})
        for j in range(n):
            if i != j:
                rows.append({"race_id": rid, "bet_type": "umatan", "combo": f"{i + 1}>{j + 1}", "odds": 0.75 / (w[i] * w[j]), "odds_min": np.nan})
    pair = pd.DataFrame(rows)
    pay = pd.DataFrame({"race_id": rid, "bet_type": ["tansho", "fukusho", "fukusho", "umaren", "umatan", "wide"], "combo": ["1", "1", "2", "1-2", "1>2", "1-2"],
                        "payout": [500, 150, 220, 1200, 2500, 300]})
    return pred, wp, pair, pay


class TicketEnvTest(unittest.TestCase):
    def test_market_probs_and_benter(self):
        p = te.market_probs(np.array([2.0, 4.0, 4.0]))
        self.assertAlmostEqual(p.sum(), 1.0)
        self.assertAlmostEqual(p[0], 0.5)
        q = te.benter_blend(np.log([0.5, 0.5]), np.log([0.9, 0.1]), 0.0, 1.0)
        np.testing.assert_allclose(q, [0.9, 0.1])

    def test_fit_benter_prefers_informative_source(self):
        rng = np.random.default_rng(0)
        groups = []
        for _ in range(400):
            true = rng.dirichlet(np.ones(8))
            w = rng.choice(8, p=true)
            noisy_m = true * np.exp(rng.normal(0, 1.0, 8))
            noisy_k = true * np.exp(rng.normal(0, 0.2, 8))
            groups.append((np.log(noisy_m / noisy_m.sum()), np.log(noisy_k / noisy_k.sum()), int(w)))
        a, b = te.fit_benter(groups)
        self.assertGreater(b, a)

    def test_build_race_tickets(self):
        pred, wp, pair, pay = _race("r1")
        t = te.build_race_tickets(pred.drop(columns="horse_number"), wp, pair, pay, a=0.5, b=0.5, K=4)
        self.assertEqual(set(t["type"]), set(te.TYPES))
        self.assertTrue((t["odds"] > 0).all() and (t["p_comb"] > 0).all())
        hit = t[(t["type"] == "tansho") & (t["combo"] == "1")]
        self.assertAlmostEqual(float(hit["mult"].iloc[0]), 5.0) if len(hit) else None
        self.assertTrue(((t["type"] == "umatan") & (t["combo"] == "1>2") & (t["mult"] == 25.0)).any() or not ((t["type"] == "umatan") & (t["combo"] == "1>2")).any())
        self.assertLessEqual(t[t["type"] == "tansho"]["p_comb"].sum(), 1.0 + 1e-9)

    def test_imputes_runner_missing_from_prediction(self):
        pred, wp, pair, pay = _race("r1")
        full = te.build_race_tickets(pred, wp, pair, pay, a=0.5, b=0.5, K=20)
        part = te.build_race_tickets(pred.iloc[:-1], wp, pair, pay, a=0.5, b=0.5, K=20)
        self.assertFalse(part.empty)
        last = str(len(pred))
        self.assertTrue((part[(part["type"] == "tansho")]["combo"] == last).any())          # 補った馬の買い目も候補に残る
        self.assertAlmostEqual(float(part[part["type"] == "tansho"]["p_comb"].sum()), 1.0, places=6)
        self.assertAlmostEqual(float(full[full["type"] == "tansho"]["p_comb"].sum()), 1.0, places=6)

    def test_scratched_runner_with_negative_odds_is_ignored(self):
        pred, wp, pair, pay = _race("r1")
        ghost = wp.iloc[[0]].copy()
        ghost["horse_id"], ghost["horse_number"], ghost["win_odds"], ghost["place_odds_min"], ghost["place_odds_max"] = "ghost", 99, -3.0, -3.0, -3.0     # 取消・除外
        wp2 = pd.concat([wp, ghost], ignore_index=True)
        a = te.build_race_tickets(pred, wp, pair, pay, a=0.5, b=0.5, K=20)
        b = te.build_race_tickets(pred, wp2, pair, pay, a=0.5, b=0.5, K=20)
        self.assertEqual(len(a), len(b))
        np.testing.assert_allclose(a["p_comb"].to_numpy(), b["p_comb"].to_numpy())
        self.assertFalse((b["combo"] == "99").any())
        out = te.impute_missing_runners(pred, wp2)
        self.assertEqual(len(out), len(pred))

    def test_race_dists_match_tickets(self):
        pred, wp, pair, pay = _race("r1")
        d = te.race_dists(pred.drop(columns="horse_number"), wp, a=0.5, b=0.5)
        hn, win, a2, a3 = d["r1"]
        self.assertAlmostEqual(float(win.sum()), 1.0)
        t = te.build_race_tickets(pred, wp, pair, pay, a=0.5, b=0.5, K=20)
        tans = t[t["type"] == "tansho"].set_index("combo")["p_comb"]
        for h, p in zip(hn, win):
            self.assertAlmostEqual(float(tans[str(int(h))]), float(p), places=9)

    def test_impute_keeps_totals(self):
        pred, wp, _, _ = _race("r1")
        out = te.impute_missing_runners(pred.iloc[:-1], wp)
        self.assertEqual(len(out), len(pred))
        self.assertAlmostEqual(float(out["p_win"].sum()), 1.0, places=6)
        self.assertAlmostEqual(float(out["p_top2"].sum()), 2.0, places=5)
        self.assertAlmostEqual(float(out["p_top3"].sum()), 3.0, places=5)


class PolicyTest(unittest.TestCase):
    def _tickets(self, races: int = 60):
        parts = []
        for k in range(races):
            pred, wp, pair, pay = _race(f"r{k}", seed=k)
            parts.append(te.build_race_tickets(pred, wp, pair, pay, a=0.5, b=0.5, K=3))
        return pd.concat(parts, ignore_index=True)

    def test_pack_and_settle(self):
        t = self._tickets(10)
        pk, mu, sd = pol.pack(t)
        self.assertEqual(pk.X.shape[0], 10)
        self.assertEqual(int(pk.mask.sum()), len(t))
        w = pol.alloc_ev_threshold(pk, tau=-1.0)
        np.testing.assert_allclose(w.sum(1), 1.0)
        s = pol.settle(pk, w)
        self.assertEqual(len(s), 10)
        np.testing.assert_allclose(s["profit"], s["ret"] - s["stake"])
        pk2, _, _ = pol.pack(t, mu, sd)
        np.testing.assert_allclose(pk.X, pk2.X)

    def test_baselines_respect_budget(self):
        t = self._tickets(10)
        pk, _, _ = pol.pack(t)
        for w in (pol.alloc_top_p(pk, "tansho"), pol.alloc_favorite(pk), pol.alloc_kelly(pk), pol.alloc_random(pk), pol.alloc_ev_threshold(pk, 0.0, top=2)):
            self.assertTrue((w.sum(1) <= 1.0 + 1e-9).all())
            self.assertTrue((w >= 0).all())
            self.assertTrue((w[~pk.mask] == 0).all())

    def test_summarize_roi(self):
        s = pd.DataFrame({"race_id": ["a", "b"], "stake": [1.0, 1.0], "ret": [3.0, 0.0]})
        s["profit"] = s["ret"] - s["stake"]
        m = pol.summarize(s, n_boot=50)
        self.assertAlmostEqual(m["roi"], 1.5)
        self.assertAlmostEqual(m["participation"], 1.0)

    def test_policy_learns_positive_ev(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch がありません")
        rng = np.random.default_rng(0)
        R, N = 400, 6
        rows = []
        for r in range(R):
            for i in range(N):
                p = float(rng.uniform(0.05, 0.4))
                odds = float(rng.choice([0.7, 1.4]) / p)       # 半分の買い目だけ期待回収率 1.4（プラス）、残りは 0.7
                rows.append({"race_id": f"r{r}", "type": "tansho", "combo": str(i), "p_comb": p, "p_model": p, "odds": odds, "mult": odds if rng.random() < p else 0.0,
                             "field_size": N, "win_entropy": 1.5, "win_max": 0.4})
        pk, _, _ = pol.pack(pd.DataFrame(rows))
        net, hist = pol.train_policy(pk, pol.TrainConfig(epochs=25, hidden=32, lr=5e-3, entropy=0.0, threads=2))
        w = pol.predict_alloc(net, pk)
        ev = pk.p * pk.odds - 1.0
        stake_pos = (w * (ev > 0.2)).sum()
        stake_neg = (w * (ev < -0.1)).sum()
        self.assertGreater(stake_pos, 3 * stake_neg)
        self.assertLess(hist[-1], hist[0])

    def test_stake_floor_forces_participation(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch がありません")
        t = self._tickets(40).assign(mult=0.0)          # 全部外れ → 自由なら見送りを学ぶ
        pk, _, _ = pol.pack(t)
        free, _ = pol.train_policy(pk, pol.TrainConfig(epochs=40, hidden=16, lr=1e-2, entropy=0.0, batch_races=8, threads=2))
        forced, _ = pol.train_policy(pk, pol.TrainConfig(epochs=40, hidden=16, lr=1e-2, entropy=0.0, batch_races=8, threads=2, target_stake=0.5))
        s_free = pol.predict_alloc(free, pk).sum(1).mean()
        s_forced = pol.predict_alloc(forced, pk).sum(1).mean()
        self.assertGreater(s_forced, s_free)
        self.assertGreater(s_forced, 0.3)
        self.assertLess(s_free, 0.3)

    def test_ev_selective_participation(self):
        t = self._tickets(20)
        pk, _, _ = pol.pack(t)
        w = pol.alloc_ev_selective(pk, 0.25, top=2)
        part = (w.sum(1) > 0).mean()
        self.assertLessEqual(part, 0.4)
        self.assertTrue((w.sum(1) <= 1.0 + 1e-9).all())

    def test_kelly_mode_and_features(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch がありません")
        t = self._tickets(30)
        t["cal_ev"] = t["p_comb"] * t["odds"]
        pk, _, _ = pol.pack(t, groups=("base", "cross", "cal"))
        self.assertEqual(pk.X.shape[2], 17 + 3 + 2)
        net, hist = pol.train_policy(pk, pol.TrainConfig(mode="kelly", epochs=3, hidden=16, threads=2, target_stake=0.3))
        w = pol.predict_alloc(net, pk)
        self.assertTrue((w.sum(1) <= 1.0 + 1e-6).all())

    def test_perturb_odds_keeps_mult(self):
        t = self._tickets(10)
        n = pol.perturb_odds(t, 0.2, seed=1)
        np.testing.assert_allclose(n["mult"], t["mult"])
        self.assertFalse(np.allclose(n["odds"], t["odds"]))
        self.assertTrue((n["odds"] >= 1.0).all())
        self.assertIs(pol.perturb_odds(t, 0.0), t)

    def test_pack_column_and_topk_score(self):
        t = self._tickets(12).reset_index(drop=True)
        pk, _, _ = pol.pack(t)
        sc = pol.pack_column(pk, t.assign(s=t["p_comb"] * t["odds"]), "s")
        self.assertEqual(int(np.isfinite(sc).sum()), len(t))
        w = pol.alloc_topk_score(pk, sc, top=2)
        self.assertTrue((np.abs(w.sum(1) - 1.0) < 1e-9).all())
        w2 = pol.alloc_topk_score(pk, sc, top=2, frac=0.25)
        self.assertLessEqual(float((w2.sum(1) > 0).mean()), 0.5)

    def test_reinforce_mode_runs(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch がありません")
        t = self._tickets(30)
        pk, _, _ = pol.pack(t)
        net, hist = pol.train_policy(pk, pol.TrainConfig(mode="reinforce", epochs=2, hidden=16, threads=2))
        w = pol.predict_alloc(net, pk)
        self.assertTrue((w.sum(1) <= 1.0 + 1e-6).all())


if __name__ == "__main__":
    unittest.main()
