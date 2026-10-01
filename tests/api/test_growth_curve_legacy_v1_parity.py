"""成長曲線: legacy (FastAPI) と v1 (Flask) のレスポンス値一致を確認する統合テスト。

両エンドポイントは同じ `growth_curve_service.get_growth_curve` を呼ぶ設計
（docs/git_management/todo/growth-curve.md「現状の実装」参照）だが、
これまで自動テストが無く手動確認のみだったため追加する。

- legacy: GET /api/growth-curve/{horse_id} (src/api/app.py)
- v1    : GET /api/v1/horse/<horse_id>/growth-curve (src/api/flask_app.py)

`get_growth_curve` をモックして同一ペイロードを返し、両エンドポイントが
同じ公開フィールド値・同じステータスコードを返すこと、および両エンドポイントが
`get_growth_curve` に渡す引数（fetch_speed_index / force_refresh /
allow_compute_on_miss / jra_only / race_index_gcs / enqueue_missing）が
同一の導出規則で一致することを検証する
（tests/api/test_tracking_difficulty_legacy_v1_parity.py と同様の構成）。
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

# test_endpoints.py と同様、DB/GCS 未接続でもローカル動作させるための既定値
os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("GCS_BUCKET", "")
os.environ.setdefault("HORSE_NAME_INDEX_DISABLE_BOOTSTRAP", "1")

from fastapi.testclient import TestClient

from src.api.app import app as fastapi_app

FAKE_HORSE_ID = "2020100001"

# growth_curve_service.get_growth_curve() が返す形式に合わせたサンプル
# (build_growth_curve_response + filter_growth_curve_for_jra 適用後の形)。
# "_from_cache" のように "_" で始まるキーは legacy 側 (_growth_curve_public_payload) で
# 除去される内部メタ情報。
SAMPLE_PAYLOAD = {
    "horse_id": FAKE_HORSE_ID,
    "horse_name": "テストホース",
    "total_races": 2,
    "avg_weight": 480.5,
    "weight_range": [478, 483],
    "best_rank": 1,
    "avg_rank": 2.5,
    "total_all_races": 2,
    "debut_weight": 478,
    "debut_date": "2023-01-05",
    "races": [
        {
            "race_id": "202301010101",
            "date": "2023-01-05",
            "venue": "東京",
            "race_name": "デビュー戦",
            "surface": "芝",
            "distance": 1600,
            "track_condition": "良",
            "rank": 4,
            "field_size": 16,
            "weight": 478,
            "weight_diff": None,
            "weight_change": None,
            "interval_days": None,
            "time": "1:34.5",
            "time_index": 50.1,
        },
        {
            "race_id": "202301020202",
            "date": "2023-02-05",
            "venue": "中山",
            "race_name": "2戦目",
            "surface": "芝",
            "distance": 1800,
            "track_condition": "良",
            "rank": 1,
            "field_size": 14,
            "weight": 483,
            "weight_diff": 5,
            "weight_change": 5,
            "interval_days": 31,
            "time": "1:48.0",
            "time_index": 55.3,
        },
    ],
    "jra_filter_active": True,
    "excluded_non_jra_count": 0,
    "_from_cache": True,
    "_cache_meta": {"version": 1, "artifact_key": "growth_curve"},
    "_data_path": "/tmp/fake/growth_curve/2020100001.json",
}

# legacy の _growth_curve_public_payload が公開するフィールド
# (= "_" で始まらないキー) と同じ集合で比較する。
PUBLIC_FIELDS = [
    "horse_id",
    "horse_name",
    "total_races",
    "avg_weight",
    "weight_range",
    "best_rank",
    "avg_rank",
    "total_all_races",
    "debut_weight",
    "debut_date",
    "races",
    "jra_filter_active",
    "excluded_non_jra_count",
]


class TestGrowthCurveLegacyV1Parity(unittest.TestCase):
    """legacy `/api/growth-curve/{id}` と v1
    `/api/v1/horse/<id>/growth-curve` が同じ値・同じ呼び出し規約を使うことを確認する。
    """

    def setUp(self):
        self.fastapi_client = TestClient(fastapi_app, raise_server_exceptions=False)

        from src.api.flask_app import create_app

        self.init_patcher = patch("src.api.flask_app.init_engine")
        self.init_patcher.start()
        self.flask_client = create_app().test_client()

    def tearDown(self):
        self.init_patcher.stop()

    @patch("src.pipeline.inference.growth_curve_service.get_growth_curve")
    def test_public_fields_match_between_legacy_and_v1(self, mock_get_growth_curve):
        mock_get_growth_curve.return_value = dict(SAMPLE_PAYLOAD)

        legacy_resp = self.fastapi_client.get(f"/api/growth-curve/{FAKE_HORSE_ID}")
        v1_resp = self.flask_client.get(f"/api/v1/horse/{FAKE_HORSE_ID}/growth-curve")

        self.assertEqual(legacy_resp.status_code, 200)
        self.assertEqual(v1_resp.status_code, 200)

        legacy_body = legacy_resp.json()
        v1_body = v1_resp.get_json()

        for field in PUBLIC_FIELDS:
            self.assertEqual(
                legacy_body.get(field),
                v1_body.get(field),
                f"legacy/v1 で '{field}' の値が一致しない: "
                f"legacy={legacy_body.get(field)!r} v1={v1_body.get(field)!r}",
            )

        # races 配列（表示上の核心値）も一致すること。
        self.assertEqual(legacy_body["races"], v1_body["races"])

        # legacy は内部メタ ("_" prefix) を除去する。v1 も同じ入力ペイロードを
        # そのまま返すため、SAMPLE_PAYLOAD にそれらのキーが無い前提で両者一致を確認。
        # (本テストでは "_" キーの除去有無そのものは比較対象にしない —
        #  legacy は明示的に strip するが v1 は strip しない既存実装のため、
        #  ユーザー向けデータである PUBLIC_FIELDS の一致を本テストの基準とする)
        for key in legacy_body:
            self.assertFalse(str(key).startswith("_"))

    @patch("src.pipeline.inference.growth_curve_service.get_growth_curve")
    def test_call_kwargs_match_default(self, mock_get_growth_curve):
        """パラメータ無し (デフォルト) 時、legacy/v1 どちらも同じ導出規則で
        get_growth_curve を呼ぶこと。"""
        mock_get_growth_curve.return_value = dict(SAMPLE_PAYLOAD)

        self.fastapi_client.get(f"/api/growth-curve/{FAKE_HORSE_ID}")
        self.flask_client.get(f"/api/v1/horse/{FAKE_HORSE_ID}/growth-curve")

        self.assertEqual(mock_get_growth_curve.call_count, 2)
        legacy_kwargs, v1_kwargs = (c.kwargs for c in mock_get_growth_curve.call_args_list)
        self.assertEqual(legacy_kwargs, v1_kwargs)
        self.assertEqual(
            legacy_kwargs,
            {
                "fetch_speed_index": False,
                "force_refresh": False,
                "limit": None,
                "jra_only": True,
                "allow_compute_on_miss": True,
                "race_index_gcs": False,
                "enqueue_missing": False,
            },
        )

    @patch("src.pipeline.inference.growth_curve_service.get_growth_curve")
    def test_call_kwargs_match_fetch_speed_index(self, mock_get_growth_curve):
        """fetch_speed_index=true 時、legacy/v1 どちらも race_index_gcs=True を
        明示的に転送すること（省略すると get_growth_curve 側の環境変数依存デフォルトに
        落ちて legacy と動作がずれるため、ここが本来の退行検知ポイント）。"""
        mock_get_growth_curve.return_value = dict(SAMPLE_PAYLOAD)

        self.fastapi_client.get(f"/api/growth-curve/{FAKE_HORSE_ID}?fetch_speed_index=true")
        self.flask_client.get(
            f"/api/v1/horse/{FAKE_HORSE_ID}/growth-curve?fetch_speed_index=true"
        )

        self.assertEqual(mock_get_growth_curve.call_count, 2)
        legacy_kwargs, v1_kwargs = (c.kwargs for c in mock_get_growth_curve.call_args_list)
        self.assertEqual(legacy_kwargs, v1_kwargs)
        self.assertTrue(legacy_kwargs["fetch_speed_index"])
        self.assertTrue(legacy_kwargs["race_index_gcs"])
        self.assertFalse(legacy_kwargs["enqueue_missing"])

    @patch("src.pipeline.inference.growth_curve_service.get_growth_curve")
    def test_call_kwargs_match_force_refresh(self, mock_get_growth_curve):
        """force_refresh=true 時、legacy/v1 どちらも fetch_speed_index/
        allow_compute_on_miss/race_index_gcs/enqueue_missing を True にすること。"""
        mock_get_growth_curve.return_value = dict(SAMPLE_PAYLOAD)

        self.fastapi_client.get(f"/api/growth-curve/{FAKE_HORSE_ID}?force_refresh=true")
        self.flask_client.get(
            f"/api/v1/horse/{FAKE_HORSE_ID}/growth-curve?force_refresh=true"
        )

        self.assertEqual(mock_get_growth_curve.call_count, 2)
        legacy_kwargs, v1_kwargs = (c.kwargs for c in mock_get_growth_curve.call_args_list)
        self.assertEqual(legacy_kwargs, v1_kwargs)
        for key in (
            "fetch_speed_index",
            "force_refresh",
            "allow_compute_on_miss",
            "race_index_gcs",
            "enqueue_missing",
        ):
            self.assertTrue(legacy_kwargs[key], f"{key} should be True")

    @patch("src.pipeline.inference.growth_curve_service.get_growth_curve")
    def test_not_precomputed_status_matches(self, mock_get_growth_curve):
        """未計算 (not_precomputed, allow_compute=false 相当) 時、
        legacy/v1 ともに 404 + 同じ status を返すこと。"""
        mock_get_growth_curve.return_value = {
            "horse_id": FAKE_HORSE_ID,
            "error": "成長曲線の事前計算データがありません。",
            "status": "not_precomputed",
            "races": [],
        }

        legacy_resp = self.fastapi_client.get(f"/api/growth-curve/{FAKE_HORSE_ID}")
        v1_resp = self.flask_client.get(f"/api/v1/horse/{FAKE_HORSE_ID}/growth-curve")

        self.assertEqual(legacy_resp.status_code, 404)
        self.assertEqual(v1_resp.status_code, 404)
        self.assertEqual(
            legacy_resp.json().get("status"), v1_resp.get_json().get("status")
        )

    @patch("src.pipeline.inference.growth_curve_service.get_growth_curve")
    def test_no_horse_result_status_matches(self, mock_get_growth_curve):
        """horse_result が存在しない馬 ID の場合、legacy/v1 ともに 404 を返すこと。"""
        mock_get_growth_curve.return_value = {
            "horse_id": FAKE_HORSE_ID,
            "error": f"馬ID {FAKE_HORSE_ID} のデータが見つかりません",
            "status": "no_horse_result",
            "races": [],
        }

        legacy_resp = self.fastapi_client.get(f"/api/growth-curve/{FAKE_HORSE_ID}")
        v1_resp = self.flask_client.get(f"/api/v1/horse/{FAKE_HORSE_ID}/growth-curve")

        self.assertEqual(legacy_resp.status_code, 404)
        self.assertEqual(v1_resp.status_code, 404)
        self.assertEqual(
            legacy_resp.json().get("status"), v1_resp.get_json().get("status")
        )


if __name__ == "__main__":
    unittest.main()
