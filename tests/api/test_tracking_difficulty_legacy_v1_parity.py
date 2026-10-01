"""追走難度: legacy (FastAPI) と v1 (Flask) のレスポンス値一致を確認する統合テスト。

両エンドポイントは同じ `tracking_difficulty_service.get_or_compute` を呼ぶ設計
（docs/git_management/todo/tracking-difficulty.md「現状の実装」参照）だが、
これまで自動テストが無く手動確認のみだったため追加する。

- legacy: GET /api/race/{race_id}/tracking-difficulty (src/api/app.py)
- v1    : GET /api/v1/races/<race_id>/tracking-difficulty (src/api/flask_app.py)

`get_or_compute` をモックして同一ペイロードを返し、両エンドポイントが
同じ公開フィールド値・同じステータスコードを返すことを検証する。
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

FAKE_RACE_ID = "202605021012"

# tracking_difficulty_service.build_tracking_difficulty_response() が返す形式に合わせたサンプル。
# "_compute_meta" のように "_" で始まるキーは legacy 側 (_tracking_difficulty_public_payload) で
# 除去される内部メタ情報。
SAMPLE_PAYLOAD = {
    "race_id": FAKE_RACE_ID,
    "race_date": "20260502",
    "result_viewable": False,
    "result_kind": None,
    "result_view_url": None,
    "race_name": "テストレース",
    "venue": "東京",
    "surface": "芝",
    "distance": 1600,
    "track_condition": "良",
    "field_size": 2,
    "pace_prediction": {"pace_type": "ミドル", "last_3f_baseline_sec": 35.1},
    "position_flow": [
        {
            "horse_number": 1,
            "positions": {
                "early": {"position": 1},
                "mid": {"position": 1},
                "late": {"position": 1},
            },
        },
        {
            "horse_number": 2,
            "positions": {
                "early": {"position": 2},
                "mid": {"position": 2},
                "late": {"position": 2},
            },
        },
    ],
    "entries": [
        {"horse_number": 1, "horse_id": "2019105678", "tracking_difficulty": 42.0},
        {"horse_number": 2, "horse_id": "2019105679", "tracking_difficulty": 55.0},
    ],
    "field_prev_stats": {"avg_last_3f": 35.0},
    "_compute_meta": {
        "elapsed_ms": 1.0,
        "lgbm_backend": "test",
        "model_key": "tracking_difficulty",
        "n_horses": 2,
        "pre_race_only": True,
        "data_source": "race_shutuba",
    },
}

# legacy の _tracking_difficulty_public_payload が公開するフィールド
# (= "_" で始まらないキー) と同じ集合で比較する。
PUBLIC_FIELDS = [
    "race_id",
    "race_date",
    "result_viewable",
    "result_kind",
    "result_view_url",
    "race_name",
    "venue",
    "surface",
    "distance",
    "track_condition",
    "field_size",
    "pace_prediction",
    "position_flow",
    "entries",
    "field_prev_stats",
]


class TestTrackingDifficultyLegacyV1Parity(unittest.TestCase):
    """legacy `/api/race/{id}/tracking-difficulty` と v1
    `/api/v1/races/<id>/tracking-difficulty` が同じ値を返すことを確認する。
    """

    def setUp(self):
        self.fastapi_client = TestClient(fastapi_app, raise_server_exceptions=False)

        from src.api.flask_app import create_app

        self.init_patcher = patch("src.api.flask_app.init_engine")
        self.init_patcher.start()
        self.flask_client = create_app().test_client()

    def tearDown(self):
        self.init_patcher.stop()

    @patch("src.pipeline.inference.tracking_difficulty_service.get_or_compute")
    def test_public_fields_match_between_legacy_and_v1(self, mock_get_or_compute):
        mock_get_or_compute.return_value = dict(SAMPLE_PAYLOAD)

        legacy_resp = self.fastapi_client.get(
            f"/api/race/{FAKE_RACE_ID}/tracking-difficulty"
        )
        v1_resp = self.flask_client.get(
            f"/api/v1/races/{FAKE_RACE_ID}/tracking-difficulty"
        )

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

        # entries 内の個々の追走難度値（表示上の核心値）も一致すること。
        legacy_entries = {e["horse_number"]: e["tracking_difficulty"] for e in legacy_body["entries"]}
        v1_entries = {e["horse_number"]: e["tracking_difficulty"] for e in v1_body["entries"]}
        self.assertEqual(legacy_entries, v1_entries)

        # 両エンドポイントが同じ get_or_compute を同じ呼び出し規約で使っていることも確認する
        # (refresh 未指定時: force_refresh/allow_scrape/allow_compute_on_miss はすべて False)。
        self.assertEqual(mock_get_or_compute.call_count, 2)
        for call in mock_get_or_compute.call_args_list:
            kwargs = call.kwargs
            self.assertFalse(kwargs.get("force_refresh"))
            self.assertFalse(kwargs.get("allow_scrape"))
            self.assertFalse(kwargs.get("allow_compute_on_miss"))
            self.assertEqual(call.args[-1], FAKE_RACE_ID)

    @patch("src.pipeline.inference.tracking_difficulty_service.get_or_compute")
    def test_refresh_true_forwarded_identically(self, mock_get_or_compute):
        """refresh=true 指定時、legacy/v1 どちらも force_refresh/allow_scrape/
        allow_compute_on_miss=True を get_or_compute に渡すこと。"""
        mock_get_or_compute.return_value = dict(SAMPLE_PAYLOAD)

        self.fastapi_client.get(
            f"/api/race/{FAKE_RACE_ID}/tracking-difficulty?refresh=true"
        )
        self.flask_client.get(
            f"/api/v1/races/{FAKE_RACE_ID}/tracking-difficulty?refresh=true"
        )

        self.assertEqual(mock_get_or_compute.call_count, 2)
        for call in mock_get_or_compute.call_args_list:
            kwargs = call.kwargs
            self.assertTrue(kwargs.get("force_refresh"))
            self.assertTrue(kwargs.get("allow_scrape"))
            self.assertTrue(kwargs.get("allow_compute_on_miss"))

    @patch("src.pipeline.inference.tracking_difficulty_service.get_or_compute")
    def test_not_precomputed_status_matches(self, mock_get_or_compute):
        """未計算 (not_precomputed) 時、legacy/v1 ともに 404 + 同じ status を返すこと。"""
        mock_get_or_compute.return_value = {
            "race_id": FAKE_RACE_ID,
            "error": "追走難度の事前計算データがありません。",
            "status": "not_precomputed",
            "entries": [],
        }

        legacy_resp = self.fastapi_client.get(
            f"/api/race/{FAKE_RACE_ID}/tracking-difficulty"
        )
        v1_resp = self.flask_client.get(
            f"/api/v1/races/{FAKE_RACE_ID}/tracking-difficulty"
        )

        self.assertEqual(legacy_resp.status_code, 404)
        self.assertEqual(v1_resp.status_code, 404)
        self.assertEqual(
            legacy_resp.json().get("status"), v1_resp.get_json().get("status")
        )


if __name__ == "__main__":
    unittest.main()
