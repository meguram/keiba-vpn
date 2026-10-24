"""導入可否の確認（readiness-audit）の集計関数のテスト。基準の JSON が壊れていないことも確かめる。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import readiness_ev_exotic as ee
from src.scripts.data import readiness_eval as ev
from src.scripts.data import readiness_stage as rs

ROOT = Path(__file__).resolve().parents[2]


class CriteriaTest(unittest.TestCase):
    def test_every_model_has_must_with_rationale(self):
        c = json.loads((ROOT / "src/scripts/data/readiness_criteria.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(c["models"]), 6)
        for name, m in c["models"].items():
            self.assertTrue(m["must"], name)
            for item in m["must"] + m.get("should", []):
                self.assertTrue(item["id"] and item["text"] and item["rationale"], (name, item.get("id")))


class EnsembleMetricsTest(unittest.TestCase):
    def test_bin_ll_perfect_and_uniform(self):
        y = np.array([1.0, 0.0])
        self.assertLess(ev.bin_ll(y, np.array([0.999, 0.001])).mean(), 0.01)
        self.assertAlmostEqual(float(ev.bin_ll(y, np.array([0.5, 0.5])).mean()), np.log(2), places=6)

    def test_calibrated_probabilities_have_slope_near_one(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.02, 0.6, 40000)
        y = (rng.uniform(size=p.size) < p).astype(float)
        a, b = ev.calib_slope(p, y)
        self.assertAlmostEqual(b, 1.0, delta=0.08)
        self.assertAlmostEqual(a, 0.0, delta=0.08)
        self.assertLess(ev.ece(p, y), 0.01)

    def test_overconfident_probabilities_have_slope_below_one(self):
        rng = np.random.default_rng(1)
        true = rng.uniform(0.1, 0.5, 40000)
        y = (rng.uniform(size=true.size) < true).astype(float)
        z = np.log(true / (1 - true)) * 1.8
        p = 1 / (1 + np.exp(-z))
        self.assertLess(ev.calib_slope(p, y)[1], 0.75)

    def test_attach_targets_drops_cancelled_horses(self):
        res = pd.DataFrame({"race_id": ["r"] * 4, "horse_id": list("abcd"), "finish_position": [1, 2, -1, 3], "date": pd.Timestamp("2026-01-01")})
        pred = pd.DataFrame({"race_id": ["r"] * 4, "horse_id": list("abcd"), "p_win": 0.25, "p_top2": 0.5, "p_top3": 0.75})
        m = ev.attach_targets(res, pred)
        self.assertEqual(len(m), 3)
        self.assertEqual(m["y_win"].sum(), 1.0)
        self.assertEqual(m["y_top3"].sum(), 3.0)

    def test_sums_check_flags_bad_race(self):
        pred = pd.DataFrame({"race_id": ["a", "a", "b", "b"], "horse_id": list("wxyz"), "p_win": [0.5, 0.5, 0.5, 0.4], "p_top2": [1.0, 1.0, 1.0, 1.0], "p_top3": [1.5, 1.5, 1.5, 1.5]})
        s = ev.sums_check(pred)
        self.assertEqual(s["bad"]["win"], 1)
        self.assertEqual(s["bad"]["top2"], 0)

    def test_paired_boot_detects_improvement(self):
        rng = np.random.default_rng(2)
        rows = []
        for r in range(300):
            for h in range(10):
                p = 0.1
                y = float(rng.uniform() < p)
                rows.append((f"r{r}", f"h{h}", y, p, 0.1 + (0.05 if y else -0.005)))
        d = pd.DataFrame(rows, columns=["race_id", "horse_id", "y_win", "pa", "pb"])
        a = d.assign(p_win=d["pa"])[["race_id", "horse_id", "y_win", "p_win"]]
        b = d.assign(p_win=d["pb"])[["race_id", "horse_id", "y_win", "p_win"]]
        out = ev.paired_boot(a, b, "win", n_boot=300)
        self.assertLess(out["hi"], 0.0)


class EvExoticTest(unittest.TestCase):
    def test_mark_stats_flags_violations(self):
        counts = pd.DataFrame({"honmei": [1, 2], "pair": [1, 1], "show_val": [1, 1], "star": [0, 1], "anchor": [3, 5]}, index=["a", "b"])
        s = ee.mark_stats(counts, pd.Series([4, 4], index=["a", "b"]))
        self.assertEqual(s["races_single_mark_over1"], 1)
        self.assertEqual(s["races_anchor_over_limit"], 1)
        self.assertAlmostEqual(s["marks_per_race_mean"], 8.0)

    def test_decile_gap_zero_when_ev_matches_realized(self):
        rng = np.random.default_rng(3)
        n = 20000
        p = rng.uniform(0.02, 0.4, n)
        odds = rng.uniform(2, 30, n)
        ret = (rng.uniform(size=n) < p) * odds
        race = np.repeat(np.arange(n // 10), 10)
        out = ee.decile_gap(p * odds, ret, race, n_boot=200)
        self.assertLess(abs(out["overall_ev"] - out["overall_realized"]), 0.05)
        lo, hi = out["top_decile_gap_ci95"]
        self.assertLess(lo, 0.0 + 0.2)
        self.assertEqual(len(out["deciles"]), 10)

    def test_ece_bins_zero_for_calibrated_constant_groups(self):
        p = np.repeat([0.1, 0.5], 1000)
        y = np.concatenate([np.r_[np.ones(100), np.zeros(900)], np.r_[np.ones(500), np.zeros(500)]])
        self.assertAlmostEqual(ee.ece_bins(p, y, bins=2), 0.0, places=9)


class StageTest(unittest.TestCase):
    def test_build_labels_signs_and_hurdle(self):
        d = Path(__file__).parent / "_tmp_readiness"
        (d / "2026").mkdir(parents=True, exist_ok=True)
        try:
            n = 6
            df = pd.DataFrame({"race_id": ["r1"] * n + ["r2"] * n, "horse_id": [f"h{i}" for i in range(2 * n)],
                               "surface": ["芝"] * n + ["障害"] * n, "finish_position": list(range(1, n + 1)) * 2,
                               "time_sec": [100 + i for i in range(n)] * 2, "last_3f": [34 + 0.1 * i for i in range(n)] * 2})
            df.to_parquet(d / "2026" / "race_result_flat.parquet")
            lab = rs.build_labels(d, 2026)
            r1 = lab[lab["race_id"] == "r1"].reset_index(drop=True)
            self.assertAlmostEqual(r1["t_rank_norm"].iloc[0], 0.0)
            self.assertAlmostEqual(r1["t_rank_norm"].iloc[-1], 1.0)
            self.assertLess(r1["t_time_dev"].iloc[0], 0)
            self.assertTrue(lab[lab["race_id"] == "r2"]["t_l3f_dev"].isna().all())
        finally:
            for f in (d / "2026").glob("*"):
                f.unlink()
            (d / "2026").rmdir()
            d.rmdir()

    def test_judge_ratio_and_missing(self):
        ref = {"missing_rate": {"s_rank_pred": 0.01}, "s_rank_pred": {"spearman_within_race": 0.40}}
        ok = {"missing_rate": {"s_rank_pred": 0.03}, "s_rank_pred": {"spearman_within_race": 0.38}}
        bad = {"missing_rate": {"s_rank_pred": 0.10}, "s_rank_pred": {"spearman_within_race": 0.30}}
        self.assertTrue(rs.judge(ref, ok, ["s_rank_pred"])["s_rank_pred"]["corr_ok"])
        self.assertTrue(rs.judge(ref, ok, ["s_rank_pred"])["s_rank_pred"]["missing_ok"])
        j = rs.judge(ref, bad, ["s_rank_pred"])["s_rank_pred"]
        self.assertFalse(j["corr_ok"])
        self.assertFalse(j["missing_ok"])


if __name__ == "__main__":
    unittest.main()
