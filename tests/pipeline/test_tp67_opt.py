"""tp-67 最適化の道具（Clopper-Pearson 上側限界・期間別の集計・運用の予測成績からの再確認）のテスト。"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.scripts.data import tp67_opt
from src.scripts.data.tp67_kill import cp_upper


class TestTp67Opt(unittest.TestCase):
    def test_clopper_pearson_upper(self):
        self.assertAlmostEqual(cp_upper(0, 100), 0.0362, places=3)          # 0/100 の 95% 上側限界 ≈ 3.6%
        self.assertAlmostEqual(cp_upper(5, 100), 0.1128, places=3)
        self.assertEqual(cp_upper(0, 0), 1.0)
        self.assertEqual(cp_upper(10, 10), 1.0)

    def test_stats_counts_hits_and_coverage(self):
        sub = pd.DataFrame({"race_id": ["a", "a", "b"], "hit3": [0, 1, 0]})
        rest = pd.DataFrame({"race_id": ["a", "b"], "hit3": [1, 0]})
        s = tp67_opt.stats(sub, rest, n_races=4)
        self.assertEqual((s["n"], s["hit"]), (3, 1))
        self.assertAlmostEqual(s["per_race"], 0.75)
        self.assertAlmostEqual(s["race_share"], 0.5)
        self.assertAlmostEqual(s["rest_rate"], 0.5)
        self.assertAlmostEqual(s["top3_removed_share"], 1 / 12)

    def test_recheck_aggregates_daily_results(self):
        with tempfile.TemporaryDirectory() as d:
            for day, n, f in (("20261024", 40, {"f1": 0, "f2": 1, "f3": 0, "f4p": 39}), ("20261025", 60, {"f1": 1, "f2": 0, "f3": 1, "f4p": 58})):
                doc = {"date": day, "models": {"ensemble_v5": 1}, "summary": {"marks": {"kesi": {"n": n, **f}}}}
                Path(d, f"{day}.json").write_text(json.dumps(doc))
            Path(d, "20261026.json").write_text(json.dumps({"date": "20261026", "summary": {"marks": {}}}))     # 消が無い旧い集計は読み飛ばす
            out = Path(d, "out")
            out.mkdir()
            tp67_opt.recheck(SimpleNamespace(results_dir=d, out=str(out)))
            res = json.loads((out / "tp67_opt_recheck.json").read_text())
        self.assertEqual((res["all"]["n"], res["all"]["hit"]), (100, 3))
        self.assertEqual(list(res["by_month"]), ["202610"])
        self.assertEqual(res["config"]["p3_max"], 0.08)
        self.assertFalse(res["all"]["ok"])                       # 100 件では上側限界が 5% を超える（判定は件数が貯まってから）


if __name__ == "__main__":
    unittest.main()
