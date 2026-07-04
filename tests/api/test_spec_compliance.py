"""仕様準拠 API / 回収率 / パス / Redis キーのユニットテスト。"""

import os
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from src.api.cache.redis_cache import (
    lap_prediction_key,
    odds_latest_key,
    prediction_key,
    ttl_until_post_time,
)
from src.config import data_paths
from src.pipeline.recovery import calculate_recovery_rate, is_value_bet


class TestDataPaths(unittest.TestCase):
    def setUp(self):
        os.environ["GCS_BUCKET"] = "test-bucket"

    def test_race_path(self):
        path = data_paths.race_path("race_shutuba", "202506010811")
        self.assertEqual(
            path,
            "gs://test-bucket/chuou/data/preprocessed/netkeiba/pc/race_shutuba/2025/202506010811.json",
        )

    def test_horse_path(self):
        path = data_paths.horse_path("horse_result", "2019105678")
        self.assertEqual(
            path,
            "gs://test-bucket/chuou/data/preprocessed/netkeiba/pc/horse_result/2019/2019105678.json",
        )

    def test_gcs_blob_path(self):
        blob = data_paths.gcs_blob_path("race_odds", "202506010811", "race")
        self.assertEqual(
            blob,
            "chuou/data/preprocessed/netkeiba/pc/race_odds/2025/202506010811.json",
        )


class TestRecoveryRate(unittest.TestCase):
    def test_calculate_recovery_rate(self):
        result = calculate_recovery_rate(0.1823, 5.2, 0.4815, 2.1)
        self.assertEqual(result["win_roi"], round(0.1823 * 5.2 * 100, 2))
        self.assertEqual(result["show_roi"], round(0.4815 * 2.1 * 100, 2))

    def test_is_value_bet(self):
        self.assertTrue(is_value_bet(94.8, 101.1))
        self.assertFalse(is_value_bet(94.8, 99.0))
        self.assertTrue(is_value_bet(100.0, None))


class TestRedisCacheKeys(unittest.TestCase):
    def test_key_formats(self):
        self.assertEqual(prediction_key("202506010811", "v1.2.0"), "prediction:202506010811:v1.2.0")
        self.assertEqual(
            lap_prediction_key("202506010811", "v1.2.0"),
            "lap:prediction:202506010811:v1.2.0",
        )
        self.assertEqual(odds_latest_key("202506010811"), "odds:latest:202506010811")

    def test_ttl_after_post(self):
        past = datetime.now(timezone.utc) - timedelta(minutes=5)
        self.assertEqual(ttl_until_post_time(past), 60)

    def test_ttl_before_post(self):
        future = datetime.now(timezone.utc) + timedelta(hours=2)
        ttl = ttl_until_post_time(future)
        self.assertGreater(ttl, 3600)


class TestFlaskApi(unittest.TestCase):
    def setUp(self):
        from src.api.flask_app import create_app

        self.init_patcher = patch("src.api.flask_app.init_engine")
        self.init_patcher.start()
        self.app = create_app()
        self.client = self.app.test_client()

    def tearDown(self):
        self.init_patcher.stop()

    def test_health(self):
        resp = self.client.get("/api/v1/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["status"], "ok")

    @patch("src.api.flask_app.get_predictions_cached")
    def test_predictions_endpoint(self, mock_get):
        mock_get.return_value = {
            "race_id": "202506010811",
            "model_version": "v1.2.0",
            "predicted_at": "2025-06-01T08:30:00+09:00",
            "pace_prediction": {"pace_category": "MIDDLE", "lap_times": []},
            "horses": [{
                "horse_id": "2019105678",
                "post_no": 3,
                "win_prob": 0.1823,
                "expected_win_roi": 94.8,
                "expected_show_roi": 101.1,
                "is_value_bet": True,
            }],
        }
        resp = self.client.get("/api/v1/races/202506010811/predictions")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["race_id"], "202506010811")
        self.assertTrue(body["horses"][0]["is_value_bet"])


if __name__ == "__main__":
    unittest.main()
