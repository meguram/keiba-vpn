"""optimize_betting の応答形式の回帰テスト（T-032 で学習PCが検出した KeyError: 'roi_pct' の再発防止）。

BettingOptimizer と保存先はスタブに差し替え、外部接続はしない。
"""
import unittest
from unittest.mock import patch

from src.api.v1 import delegates


class _FakeStorage:
    def load(self, category, key):
        return {}


class _FakeOptimizer:
    """本物の optimize() が返すキー構成（betting.py: total_bet / expected_return / expected_roi / candidates ...）。"""

    def __init__(self, config):
        pass

    def optimize(self, predictions, pair_odds, bankroll, single_odds=None):
        return {
            "total_bet": 1000,
            "expected_return": 1050.0,
            "expected_roi": 1.05,   # 期待払戻 / 賭け金（1.0 が損益分岐の回収率）
            "remaining": bankroll - 1000,
            "candidates": [],
        }


def _call(body, cached=None):
    with patch("src.pipeline.inference.race_prediction_service.load_cached", return_value=cached), \
         patch("src.scraper.storage.HybridStorage", _FakeStorage), \
         patch("src.pipeline.inference.betting.BettingOptimizer", _FakeOptimizer):
        return delegates.optimize_betting(body)


class OptimizeBettingTest(unittest.TestCase):
    PRED = {"predictions": [{"horse_number": 1, "pred_score": 1.0}]}

    def test_success_payload_has_expected_keys(self):
        payload, code = _call({"race_id": "202606010101", "bankroll": 50000}, self.PRED)
        self.assertEqual(code, 200)
        self.assertEqual(
            sorted(payload),
            ["bankroll", "candidates", "expected_return", "race_id", "roi_pct", "total_bet"],
        )
        self.assertEqual(payload["bankroll"], 50000)

    def test_roi_pct_is_recovery_rate_percent(self):
        # expected_roi=1.05（回収率 105%）→ roi_pct=105.0。100 が損益分岐。利益率(5%)ではない
        payload, _ = _call({"race_id": "202606010101"}, self.PRED)
        self.assertEqual(payload["roi_pct"], 105.0)

    def test_requires_race_id(self):
        self.assertEqual(_call({}, self.PRED)[1], 400)

    def test_no_cached_predictions_is_404(self):
        self.assertEqual(_call({"race_id": "000000000000"}, None)[1], 404)


if __name__ == "__main__":
    unittest.main()
