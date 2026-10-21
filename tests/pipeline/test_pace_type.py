"""tp-61: 展開タイプ（pace_type）の規則・特徴量の一致・API 付与の検査。"""

import itertools
import os
import unittest

import numpy as np
import pandas as pd

from src.pipeline.inference import pace_type as pt
from src.scripts.data import tp61_build as b
from src.scripts.data import tp61_hana as hn
from src.scripts.data import tp61_pre as pre

TH = {"bal_q25": -0.8, "bal_q75": 0.8, "by_band": {"マイル(1500~1800)": {"lull_amp": 0.35, "accel_len": 0.22}}}


class RuleTest(unittest.TestCase):
    def test_every_race_gets_exactly_one_type(self):
        seen = set()
        for bal, lull, acc in itertools.product(np.linspace(-6, 8, 57), np.linspace(-1, 2, 31), np.linspace(0, 0.5, 11)):
            t = pt.lap_type_rule(float(bal), float(lull), float(acc), TH, "マイル(1500~1800)")
            self.assertIn(t, pt.LAP_TYPES)
            seen.add(t)
        self.assertEqual(seen, set(pt.LAP_TYPES))

    def test_balance_decides_the_pace_family(self):
        self.assertTrue(pt.lap_type_rule(-1.5, 0.0, 0.0, TH, "マイル(1500~1800)").startswith("slow"))
        self.assertTrue(pt.lap_type_rule(0.0, 0.0, 0.0, TH, "マイル(1500~1800)").startswith("even"))
        self.assertTrue(pt.lap_type_rule(1.5, 0.0, 0.0, TH, "マイル(1500~1800)").startswith("high"))


class ParseTest(unittest.TestCase):
    def test_order_text_gaps(self):
        horses, gaps = b.parse_order("(*10,15)16(7,13)(1,9)(4,5,11)(8,3)-(2,12,6)-14")
        self.assertEqual(len(horses), 16)
        self.assertEqual(sum(gaps), 2)
        self.assertEqual(b.parse_order("13(12,14)=2")[1], [0, 0, 3])

    def test_margin(self):
        self.assertAlmostEqual(b.margin_len("1.1/4"), 1.25)
        self.assertAlmostEqual(b.margin_len("クビ"), 0.2)
        self.assertTrue(np.isnan(b.margin_len("")))


class FeatureParityTest(unittest.TestCase):
    def test_race_pf_stats_matches_training(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            n = int(rng.integers(8, 18))
            e = [None if rng.random() < 0.1 else float(rng.random()) for _ in range(n)]
            serv = pt.race_pf_stats(e)
            tr = pre.race_pf_features(pd.DataFrame({"race_id": "r", "pf_early": [np.nan if x is None else x for x in e]})).iloc[0].to_dict()
            for k, v in serv.items():
                self.assertAlmostEqual(float(v), float(tr[k]), places=9, msg=k)

    def test_hana_features_match_training(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            n = int(rng.integers(8, 17))
            n5 = rng.integers(0, 6, n).astype(float)
            led = np.array([rng.integers(0, int(m) + 1) if m > 0 else 0 for m in n5], dtype=float)
            lead = np.where(n5 > 0, led / np.where(n5 > 0, n5, 1), np.nan)
            front = np.where(n5 > 0, np.minimum(1.0, (led + rng.integers(0, 2, n)) / np.where(n5 > 0, n5, 1)), np.nan)
            nums = rng.permutation(np.arange(1, n + 1))
            stats = [{"lead5": a, "n_led5": c, "n5": d, "front5": f} for a, c, d, f in zip(lead, led, n5, front)]
            serv = pt.hana_race_features(stats, nums.tolist(), n)
            a = pd.DataFrame({"race_id": "r", "field_size": n, "lead5": lead, "n_led5": led, "n5": n5, "front5": front, "horse_number": nums})
            tr = hn.race_hana(a).iloc[0].to_dict()
            for k, v in serv.items():
                x, y = float(v), float(tr[k])
                if np.isnan(x):
                    self.assertTrue(np.isnan(y), k)
                else:
                    self.assertAlmostEqual(x, y, places=9, msg=k)

    def test_horse_lead_stats_uses_only_earlier_runs(self):
        past = [{"date": "2026.01.10", "raw": "2026.01.10 東京 12頭 1-1-1-1"}, {"date": "2026.01.03", "raw": "2026.01.03 中山 12頭 5-5-4-4"},
                {"date": "2025.12.20", "raw": "2025.12.20 阪神 12頭 1-1-2-2"}]
        s = pt.horse_lead_stats(past, "2026.01.10")
        self.assertEqual(s["n5"], 2.0)
        self.assertAlmostEqual(s["lead5"], 0.5)


class ServingTest(unittest.TestCase):
    def _resp(self):
        rng = np.random.default_rng(3)
        n = 16
        w = rng.dirichlet(np.ones(n) * 2)
        preds = []
        for i in range(n):
            preds.append({"horse_number": i + 1, "horse_id": f"h{i}", "win_prob": float(w[i]), "top2_prob": float(min(0.9, w[i] * 2.1)), "top3_prob": float(min(0.95, w[i] * 3.0)),
                          "position_forecast": {"early_pos_norm": float(rng.random())}})
        return {"venue": "東京", "surface": "芝", "distance": 1600, "predictions": preds,
                "pace_forecast": {"level": "mid", "front_runner_count": 2, "stalker_count": 4, "early_pressure": 0.4, "known_horses": n}}

    def test_disabled_by_default(self):
        os.environ.pop(pt.ENV_FLAG, None)
        r = self._resp()
        self.assertEqual(pt.attach_pace_type(r, None, "2026.10.24"), r)

    def test_attach_gives_consistent_probabilities(self):
        if pt.load_config() is None or pt._boosters() is None:
            self.skipTest("config/pace_type*.json または lightgbm が無い")
        os.environ[pt.ENV_FLAG] = "1"
        try:
            r = self._resp()
            out = pt.attach_pace_type(r, None, "2026.10.24", {"track_condition": "良", "grade": "3勝クラス"})
        finally:
            os.environ.pop(pt.ENV_FLAG, None)
        self.assertIn("pace_type", out)
        p = out["pace_type"]
        self.assertEqual(len(p["probs"]), 9)
        self.assertAlmostEqual(sum(p["probs"].values()), 1.0, places=2)
        self.assertAlmostEqual(sum(p["pace_probs"].values()), 1.0, places=2)
        self.assertEqual(set(p["style_fit"]), {"nige", "senko", "sashi", "oikomi"})
        sf = out["scenario_forecast"]
        for k in p["probs"]:
            for kind, tot in (("win", 1.0), ("top2", 2.0), ("top3", 3.0)):
                s = sum(h["by_scenario"][k][kind] for h in sf["horses"])
                self.assertAlmostEqual(s, tot, delta=0.02, msg=(k, kind))
        self.assertEqual(out["predictions"], r["predictions"])           # 既存の項目は変えない


if __name__ == "__main__":
    unittest.main()
