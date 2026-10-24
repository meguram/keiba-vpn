"""tp-70 確率の較正（src/utils/prob_calibration.py）と、評価用部品（tp70_lib / tp70_fit）のテスト。"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from src.scripts.data import tp70_fit as F
from src.scripts.data import tp70_lib as L
from src.utils import prob_calibration as P


def race(n=12, seed=0):
    rng = np.random.default_rng(seed)
    odds = np.sort(rng.uniform(1.8, 120, n))
    q = (1 / odds) / (1 / odds).sum()
    w = rng.dirichlet(np.ones(n) * 1.5)
    from src.utils.race_probabilities import harville_top2_prob, harville_top3_prob

    return odds, {"win": w, "top2": harville_top2_prob(w), "top3": harville_top3_prob(w)}


class CalibrationTest(unittest.TestCase):
    def test_sums_and_range(self):
        odds, pm = race()
        out = P.calibrate_arrays(pm, odds)
        self.assertIsNotNone(out)
        for t, k in (("win", 1.0), ("top2", 2.0), ("top3", 3.0)):
            self.assertAlmostEqual(float(out[t].sum()), k, places=3)
            self.assertTrue(np.all(out[t] > 0) and np.all(out[t] < 1))

    def test_pulls_toward_market(self):
        odds, pm = race()
        out = P.calibrate_arrays(pm, odds)
        mk = P.market_probs(odds)
        d_raw = np.abs(pm["top3"] - mk["top3"]).sum()
        d_cal = np.abs(out["top3"] - mk["top3"]).sum()
        self.assertLess(d_cal, d_raw)       # 較正後は市場に近づく（モデルの自信過剰を縮める）

    def test_fallbacks(self):
        odds, pm = race()
        self.assertIsNone(P.calibrate_arrays(pm, odds[:5]))                  # 8 頭未満
        bad = odds.copy()
        bad[2] = 0.0
        self.assertIsNone(P.calibrate_arrays(pm, bad))                       # オッズ欠損
        entries = [{"horse_number": i + 1, "win_prob": pm["win"][i], "top2_prob": pm["top2"][i], "top3_prob": pm["top3"][i]} for i in range(len(odds))]
        om = {i + 1: float(odds[i]) for i in range(len(odds))}
        with mock.patch.dict(os.environ, {"KEIBA_PROB_CALIBRATION": ""}):
            self.assertIsNone(P.calibrate_entries(entries, om))              # 既定オフ
        with mock.patch.dict(os.environ, {"KEIBA_PROB_CALIBRATION": "1"}):
            self.assertIsNone(P.calibrate_entries(entries, om, market_is_real=False))   # 予測オッズは使わない
            res = P.calibrate_entries(entries, om)
            self.assertEqual(len(res), len(entries))
            self.assertAlmostEqual(sum(r["win_prob"] for r in res), 1.0, places=2)

    def test_methods_agree_in_shape(self):
        odds, pm = race(seed=3)
        a = P.calibrate_arrays(pm, odds, {"top3": "blend"})
        b = P.calibrate_arrays(pm, odds, {"top3": "blend_pop"})
        self.assertLess(np.abs(a["top3"] - b["top3"]).max(), 0.15)           # 人気区分別は同じ方向の小さな調整


class LibTest(unittest.TestCase):
    def test_scale_to_sum_matches_utils(self):
        p = np.array([0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.01])
        a = L.scale_to_sum(p, 3.0)
        b = P.scale_to_sum(p, 3.0)
        np.testing.assert_allclose(a, b, atol=1e-9)
        self.assertAlmostEqual(float(a.sum()), 3.0, places=4)

    def test_ece_and_logloss(self):
        y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 0, 1] * 20)
        p = np.full(len(y), y.mean())
        self.assertAlmostEqual(L.ece(y, p), 0.0, places=6)
        self.assertGreater(L.logloss(y, np.full(len(y), 0.05)), L.logloss(y, p))

    def test_fit_predict_improves_over_raw(self):
        rng = np.random.default_rng(0)
        n_race, n = 400, 10
        rows = []
        for r in range(n_race):
            q = rng.dirichlet(np.ones(n))
            truth = 0.5 * np.log(q) + 0.5 * rng.normal(0, 0.5, n)       # 実力は市場と無関係な成分を半分持つ
            w = np.exp(truth) / np.exp(truth).sum()
            winner = rng.choice(n, p=w)
            for i in range(n):
                rows.append({"race_id": f"r{r}", "year": 2022 + (r % 3), "n": n, "p_win": float(np.clip(q[i] * 1.8 if i == 0 else q[i] * 0.9, 1e-4, 0.9)), "m_win": q[i], "h1": int(i == winner), "pop": i + 1})
        df = pd.DataFrame(rows)
        for c in ("p_top2", "p_top3", "m_top2", "m_top3"):
            df[c] = df["p_win"] if c.startswith("p") else df["m_win"]
        df["h2"] = df["h1"]
        df["h3"] = df["h1"]
        tr, te = df[df["year"] < 2024], df[df["year"] == 2024]
        p = F.fit_predict(tr, te, "win", "C3")
        self.assertTrue(np.all((p > 0) & (p < 1)))
        self.assertLessEqual(L.logloss(te["h1"], p), L.logloss(te["h1"], te["p_win"]) + 1e-9)


if __name__ == "__main__":
    unittest.main()


class ApplyTest(unittest.TestCase):
    def preds(self, n=14, seed=1, **extra):
        rng = np.random.default_rng(seed)
        odds = np.sort(rng.uniform(1.8, 90, n)).round(1)
        w = rng.dirichlet(np.ones(n))
        from src.utils.race_probabilities import harville_top2_prob, harville_top3_prob

        t2, t3 = harville_top2_prob(w), harville_top3_prob(w)
        return [{"horse_number": i + 1, "win_prob": float(w[i]), "top2_prob": float(t2[i]), "top3_prob": float(t3[i]), "win_odds": float(odds[i]),
                 "top2_odds_est": 5.0, "place_odds_eff": 2.0, "mark_type": "none", **extra} for i in range(n)]

    def test_off_returns_input(self):
        r = {"predictions": self.preds(), "has_prediction": True}
        with mock.patch.dict(os.environ, {"KEIBA_PROB_CALIBRATION": ""}):
            self.assertIs(P.apply_prob_calibration(r), r)

    def test_applied_keeps_model_probs_and_sums(self):
        with mock.patch.dict(os.environ, {"KEIBA_PROB_CALIBRATION": "1"}):
            r = P.apply_prob_calibration({"predictions": self.preds(), "has_prediction": True}, odds_phase="t15")
        self.assertTrue(r["prob_calibration"]["applied"])
        self.assertEqual(r["prob_calibration"]["odds_phase"], "t15")
        ps = r["predictions"]
        self.assertAlmostEqual(sum(e["win_prob"] for e in ps), 1.0, places=2)
        self.assertAlmostEqual(sum(e["top3_prob"] for e in ps), 3.0, places=1)
        self.assertTrue(all("win_prob_model" in e and e["prob_calibrated"] for e in ps))
        self.assertTrue(all(e["ev_win"] == round(e["win_prob"] * e["win_odds"], 3) for e in ps))
        self.assertIn("honmei", {e["mark_type"] for e in ps})

    def test_not_applied_reasons(self):
        with mock.patch.dict(os.environ, {"KEIBA_PROB_CALIBRATION": "1"}):
            small = P.apply_prob_calibration({"predictions": self.preds(n=6), "has_prediction": True})
            pred_odds = P.apply_prob_calibration({"predictions": self.preds(odds_source="predicted"), "has_prediction": True})
            ps = self.preds()
            ps[3]["win_odds"] = None
            missing = P.apply_prob_calibration({"predictions": ps, "has_prediction": True})
        for r in (small, pred_odds, missing):
            self.assertFalse(r["prob_calibration"]["applied"])
            self.assertTrue(r["prob_calibration"]["reason"])
            self.assertNotIn("win_prob_model", r["predictions"][0])

    def test_no_double_application(self):
        with mock.patch.dict(os.environ, {"KEIBA_PROB_CALIBRATION": "1"}):
            once = P.apply_prob_calibration({"predictions": self.preds(), "has_prediction": True})
            twice = P.apply_prob_calibration(once)
        self.assertIs(once, twice)

    def test_pc_thresholds_for_star_and_kesi(self):
        from unittest import mock

        from src.utils import race_probabilities as RP

        with mock.patch("src.utils.race_probabilities.star_v1_config", return_value={}):       # 旧判定（生のEV）の水準の確認
            self._pc_thresholds_legacy(RP)

    def _pc_thresholds_legacy(self, RP):
        e = {"win_odds": 15.0, "ev_win": 0.85, "ev_place": 0.0, "top3_prob": 0.2}
        self.assertFalse(RP.qualifies_star_mark(dict(e), 0.07))                      # 従来の水準（EV>=1.0）では★にならない
        self.assertTrue(RP.qualifies_star_mark({**e, "prob_calibrated": True}, 0.07))  # 補正後の確率用の水準（0.80）では★
        ents = [{"horse_number": i + 1, "win_odds": 3.0 + 4 * i, "top3_prob": 0.07 if i == 11 else 0.4, "prob_calibrated": True, "mark_type": "none"} for i in range(12)]
        RP.assign_kesi_mark(ents)
        self.assertNotEqual(ents[11]["mark_type"], "kesi")                           # 補正後 7% は 6% 超（従来の 8% では消）
        ents[11]["top3_prob"] = 0.055
        RP.assign_kesi_mark(ents)
        self.assertEqual(ents[11]["mark_type"], "kesi")
