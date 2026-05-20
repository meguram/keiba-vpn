"""
API エンドポイント ランダムユーザー挙動テスト

全エンドポイントのバリデーション・エラーハンドリングを確認する。
195エンドポイントの中で特に重要なグループをカバーし、以下を検証:
  A. パラメータバリデーション (欠損・型エラー・範囲外)
  B. 不正なIDフォーマット (空・長大・XSS・SQLi・ Unicode)
  C. 日付フォーマットバリデーション
  D. クエリパラメータインジェクション
  E. 常にJSON 200を返すべき基本エンドポイント
  F. POST エンドポイントの不正ボディ
"""
from __future__ import annotations

import os
import unittest

# Auth を無効化 + GCS を空にしてローカル動作を強制
os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("GCS_BUCKET", "")

from fastapi.testclient import TestClient
from src.api.app import app

# raise_server_exceptions=False でサーバ内部例外を 5xx として受け取り、
# テスト自体がクラッシュしないようにする
client = TestClient(app, raise_server_exceptions=False)

# ─── helpers ──────────────────────────────────────────────

XSS_PAYLOAD = "<script>alert(1)</script>"
SQLI_PAYLOAD = "'; DROP TABLE races; --"
LONG_STRING = "A" * 1024
UNICODE_PAYLOAD = "🐎競馬テスト🏇"
# 存在しないが形式上正しいID群
FAKE_RACE_ID = "202501010101"
FAKE_HORSE_ID = "2020100001"
FAKE_PERSON_ID = "00001"


def _is_json(r) -> bool:
    """レスポンスが valid JSON であるかチェック。"""
    try:
        r.json()
        return True
    except Exception:
        return False


def _no_html_echo(r, payload: str) -> bool:
    """HTML エコーバックされていないことを確認 (XSS防止)。"""
    ct = r.headers.get("content-type", "")
    if "text/html" in ct:
        # HTML ページが返った場合、ペイロードがそのままあるかチェック
        return payload not in r.text
    return True  # JSON なら OK


# ═══════════════════════════════════════════════════════
# E. 常にJSON 200 を返すべき基本エンドポイント
# ═══════════════════════════════════════════════════════

class TestHealthAndAuth(unittest.TestCase):
    """ヘルス / 認証ステータス — 常に 200 + JSON を返す。"""

    def test_health_returns_200(self):
        r = client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        self.assertIn("status", r.json())

    def test_health_has_uptime(self):
        r = client.get("/api/health")
        self.assertIn("uptime_seconds", r.json())

    def test_auth_status_returns_200(self):
        r = client.get("/api/auth/status")
        self.assertEqual(r.status_code, 200)
        self.assertIn("is_developer", r.json())

    def test_auth_status_is_json(self):
        r = client.get("/api/auth/status")
        self.assertTrue(_is_json(r))

    def test_health_no_server_error(self):
        r = client.get("/api/health")
        self.assertNotEqual(r.status_code, 500)


# ═══════════════════════════════════════════════════════
# C. 日付フォーマットバリデーション — /api/race-list/{date}
# ═══════════════════════════════════════════════════════

class TestRaceListDateFormats(unittest.TestCase):
    """race-list/{date} への異常入力 — 500 を返さないことを確認。"""

    def test_valid_future_date_returns_empty(self):
        """未来日付は空リストを返す（500 ではない）。"""
        r = client.get("/api/race-list/20991231")
        self.assertNotEqual(r.status_code, 500)
        self.assertTrue(_is_json(r))

    def test_zero_date_no_500(self):
        r = client.get("/api/race-list/00000000")
        self.assertNotEqual(r.status_code, 500)

    def test_dashed_date_format_no_500(self):
        """YYYY-MM-DD 形式 (ハイフンあり) — 500 NG。"""
        r = client.get("/api/race-list/2024-01-01")
        self.assertNotEqual(r.status_code, 500)

    def test_invalid_string_date_no_500(self):
        r = client.get("/api/race-list/invalid-date")
        self.assertNotEqual(r.status_code, 500)

    def test_xss_in_date_no_html_echo(self):
        r = client.get(f"/api/race-list/{XSS_PAYLOAD}")
        self.assertNotEqual(r.status_code, 500)
        self.assertTrue(_no_html_echo(r, XSS_PAYLOAD))

    def test_long_date_no_500(self):
        r = client.get(f"/api/race-list/{LONG_STRING[:50]}")
        self.assertNotEqual(r.status_code, 500)

    def test_race_list_response_has_races_key(self):
        """存在しない日付でも races キーを返す。"""
        r = client.get("/api/race-list/20240101")
        self.assertNotEqual(r.status_code, 500)
        if r.status_code == 200:
            data = r.json()
            # 最低限 races または error キーがある
            self.assertTrue("races" in data or "error" in data or "date" in data)

    def test_sqli_in_date_no_500(self):
        r = client.get(f"/api/race-list/{SQLI_PAYLOAD}")
        self.assertNotEqual(r.status_code, 500)


# ═══════════════════════════════════════════════════════
# E (続き). 常に 200 を返すべきエンドポイント
# ═══════════════════════════════════════════════════════

class TestScrapeDates(unittest.TestCase):
    """scrape-dates — 200 + JSON。"""

    def test_scrape_dates_returns_200(self):
        r = client.get("/api/scrape-dates")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(_is_json(r))

    def test_scrape_dates_has_dates_key(self):
        r = client.get("/api/scrape-dates")
        data = r.json()
        self.assertIn("dates", data)

    def test_scrape_dates_dates_is_list(self):
        r = client.get("/api/scrape-dates")
        data = r.json()
        self.assertIsInstance(data.get("dates"), list)


# ═══════════════════════════════════════════════════════
# B. race_id の不正フォーマット
# ═══════════════════════════════════════════════════════

class TestRaceIdBadFormats(unittest.TestCase):
    """race/{race_id} — 不正 ID で 500 が出ないことを確認。"""

    def test_nonexistent_valid_format_race_id(self):
        """存在しないが形式正常な race_id → 200 (データ空) or 404、5xx NG。"""
        r = client.get(f"/api/race/{FAKE_RACE_ID}")
        self.assertNotIn(r.status_code, [500, 502, 503])
        self.assertTrue(_is_json(r))

    def test_xss_race_id_no_500(self):
        r = client.get(f"/api/race/{XSS_PAYLOAD}")
        self.assertNotEqual(r.status_code, 500)
        self.assertTrue(_no_html_echo(r, XSS_PAYLOAD))

    def test_sqli_race_id_no_500(self):
        r = client.get(f"/api/race/{SQLI_PAYLOAD}")
        self.assertNotEqual(r.status_code, 500)

    def test_long_race_id_no_500(self):
        r = client.get(f"/api/race/{LONG_STRING}")
        self.assertNotEqual(r.status_code, 500)

    def test_unicode_race_id_no_500(self):
        r = client.get(f"/api/race/{UNICODE_PAYLOAD}")
        self.assertNotEqual(r.status_code, 500)

    def test_race_predictions_nonexistent_returns_404(self):
        """予測データなし → 404 (5xx ではなく)。"""
        r = client.get(f"/api/race/{FAKE_RACE_ID}/predictions")
        self.assertIn(r.status_code, [404, 200])
        self.assertTrue(_is_json(r))

    def test_race_predictions_xss_no_500(self):
        r = client.get(f"/api/race/{XSS_PAYLOAD}/predictions")
        self.assertNotEqual(r.status_code, 500)

    def test_race_bundle_nonexistent_no_500(self):
        r = client.get(f"/api/race/{FAKE_RACE_ID}/bundle")
        self.assertNotIn(r.status_code, [500, 502, 503])
        self.assertTrue(_is_json(r))

    def test_race_bundle_xss_no_500(self):
        r = client.get(f"/api/race/{XSS_PAYLOAD}/bundle")
        self.assertNotEqual(r.status_code, 500)

    def test_race_bundle_long_id_no_500(self):
        r = client.get(f"/api/race/{LONG_STRING}/bundle")
        self.assertNotEqual(r.status_code, 500)


# ═══════════════════════════════════════════════════════
# B. horse_id の不正フォーマット
# ═══════════════════════════════════════════════════════

class TestHorseIdBadFormats(unittest.TestCase):
    """horse/{horse_id} — 不正 ID で 500 が出ないことを確認。"""

    def test_nonexistent_horse_detail_no_500(self):
        r = client.get(f"/api/horse/{FAKE_HORSE_ID}/detail")
        self.assertNotIn(r.status_code, [500, 502, 503])
        self.assertTrue(_is_json(r))

    def test_xss_horse_detail_no_html_echo(self):
        r = client.get(f"/api/horse/{XSS_PAYLOAD}/detail")
        self.assertNotEqual(r.status_code, 500)
        self.assertTrue(_no_html_echo(r, XSS_PAYLOAD))

    def test_sqli_horse_detail_no_500(self):
        r = client.get(f"/api/horse/{SQLI_PAYLOAD}/detail")
        self.assertNotEqual(r.status_code, 500)

    def test_long_horse_id_no_500(self):
        r = client.get(f"/api/horse/{LONG_STRING}/detail")
        self.assertNotEqual(r.status_code, 500)

    def test_unicode_horse_detail_no_500(self):
        r = client.get(f"/api/horse/{UNICODE_PAYLOAD}/detail")
        self.assertNotEqual(r.status_code, 500)

    def test_growth_curve_nonexistent_no_500(self):
        r = client.get(f"/api/growth-curve/{FAKE_HORSE_ID}")
        self.assertNotIn(r.status_code, [500, 502, 503])

    def test_growth_curve_xss_no_500(self):
        r = client.get(f"/api/growth-curve/{XSS_PAYLOAD}")
        self.assertNotEqual(r.status_code, 500)

    def test_growth_curve_long_id_no_500(self):
        r = client.get(f"/api/growth-curve/{LONG_STRING}")
        self.assertNotEqual(r.status_code, 500)


# ═══════════════════════════════════════════════════════
# A. /api/data/{category}/{key} — パラメータ検証
# ═══════════════════════════════════════════════════════

class TestRawDataEndpoint(unittest.TestCase):
    """data/{category}/{key} — 404 と不正入力ハンドリング。"""

    def test_unknown_category_returns_404(self):
        r = client.get("/api/data/nonexistent_cat/somekey")
        self.assertEqual(r.status_code, 404)
        self.assertIn("error", r.json())

    def test_valid_category_missing_key_returns_404(self):
        r = client.get(f"/api/data/race_shutuba/{FAKE_RACE_ID}")
        self.assertEqual(r.status_code, 404)

    def test_xss_category_no_html_echo(self):
        r = client.get(f"/api/data/{XSS_PAYLOAD}/key123")
        self.assertNotEqual(r.status_code, 500)
        self.assertTrue(_no_html_echo(r, XSS_PAYLOAD))

    def test_xss_key_no_html_echo(self):
        r = client.get(f"/api/data/race_shutuba/{XSS_PAYLOAD}")
        self.assertNotEqual(r.status_code, 500)
        self.assertTrue(_no_html_echo(r, XSS_PAYLOAD))

    def test_sqli_category_no_500(self):
        r = client.get(f"/api/data/{SQLI_PAYLOAD}/key123")
        self.assertNotEqual(r.status_code, 500)

    def test_long_key_no_500(self):
        r = client.get(f"/api/data/race_shutuba/{LONG_STRING}")
        self.assertNotEqual(r.status_code, 500)

    def test_race_performance_short_key_returns_404(self):
        """race_performance カテゴリで短すぎる key → 404 (not 500)。"""
        r = client.get("/api/data/race_performance/abc")
        self.assertNotIn(r.status_code, [500])
        self.assertTrue(_is_json(r))


# ═══════════════════════════════════════════════════════
# A. /api/person/{ptype}/{person_id}/stats — enum バリデーション
# ═══════════════════════════════════════════════════════

class TestPersonStatsEndpoint(unittest.TestCase):
    """ptype が jockey / trainer 以外なら 400 を返す。"""

    def test_invalid_ptype_returns_400(self):
        r = client.get(f"/api/person/invalid/12345/stats")
        self.assertEqual(r.status_code, 400)
        self.assertIn("error", r.json())

    def test_valid_ptype_jockey_no_500(self):
        r = client.get(f"/api/person/jockey/{FAKE_PERSON_ID}/stats")
        self.assertNotIn(r.status_code, [500, 502, 503])

    def test_valid_ptype_trainer_no_500(self):
        r = client.get(f"/api/person/trainer/{FAKE_PERSON_ID}/stats")
        self.assertNotIn(r.status_code, [500, 502, 503])

    def test_xss_ptype_returns_400_or_no_echo(self):
        r = client.get(f"/api/person/{XSS_PAYLOAD}/12345/stats")
        self.assertNotEqual(r.status_code, 500)
        self.assertTrue(_no_html_echo(r, XSS_PAYLOAD))

    def test_sqli_person_id_no_500(self):
        r = client.get(f"/api/person/jockey/{SQLI_PAYLOAD}/stats")
        self.assertNotEqual(r.status_code, 500)

    def test_long_person_id_no_500(self):
        # BUG: _fetch_person_profile() constructs `data/local/meta/person/jockey_<id>.json`
        # with Path.exists() which raises OSError: File name too long for IDs > ~240 chars.
        # Fix: add `if len(person_id) > 240: return {}` guard before path construction.
        r = client.get(f"/api/person/jockey/{LONG_STRING}/stats")
        self.assertNotEqual(r.status_code, 500)


# ═══════════════════════════════════════════════════════
# E. bloodline-cluster メタ / クラスタ — 200 + JSON
# ═══════════════════════════════════════════════════════

class TestBloodlineClusterMeta(unittest.TestCase):
    """meta / clusters — アーティファクト未生成でも 5xx にならない。"""

    def test_meta_no_500(self):
        r = client.get("/api/bloodline-cluster/meta")
        self.assertNotIn(r.status_code, [500, 502])
        self.assertTrue(_is_json(r))

    def test_clusters_no_500(self):
        # BUG: list_clusters() returns dicts with NaN float values (from pandas).
        # json.dumps() raises ValueError: Out of range float values are not JSON compliant: nan
        # Fix: use allow_nan=False + replace NaN with None, or use a NaN-safe JSON encoder.
        r = client.get("/api/bloodline-cluster/clusters")
        self.assertNotIn(r.status_code, [500, 502])
        self.assertTrue(_is_json(r))

    def test_tags_no_500(self):
        r = client.get("/api/bloodline-cluster/tags")
        self.assertNotIn(r.status_code, [500, 502])
        self.assertTrue(_is_json(r))


# ═══════════════════════════════════════════════════════
# A. bloodline-cluster/lookup — 必須パラメータ name
# ═══════════════════════════════════════════════════════

class TestBloodlineClusterLookup(unittest.TestCase):
    """lookup / lookup-by-id — 必須パラメータ欠損→422、不正値→no 500。"""

    def test_lookup_missing_name_returns_422(self):
        """name パラメータ必須 (Query(..., min_length=1)) → 欠損で 422。"""
        r = client.get("/api/bloodline-cluster/lookup")
        self.assertEqual(r.status_code, 422)

    def test_lookup_xss_name_no_500(self):
        r = client.get(f"/api/bloodline-cluster/lookup?name={XSS_PAYLOAD}")
        self.assertNotEqual(r.status_code, 500)
        self.assertTrue(_no_html_echo(r, XSS_PAYLOAD))

    def test_lookup_long_name_no_500(self):
        r = client.get(f"/api/bloodline-cluster/lookup?name={LONG_STRING}")
        self.assertNotEqual(r.status_code, 500)

    def test_lookup_by_id_missing_returns_422(self):
        """horse_id パラメータ必須 → 欠損で 422。"""
        r = client.get("/api/bloodline-cluster/lookup-by-id")
        self.assertEqual(r.status_code, 422)

    def test_lookup_by_id_fake_id_no_500(self):
        r = client.get(f"/api/bloodline-cluster/lookup-by-id?horse_id={FAKE_HORSE_ID}")
        self.assertNotEqual(r.status_code, 500)

    def test_lookup_by_id_xss_no_html_echo(self):
        r = client.get(f"/api/bloodline-cluster/lookup-by-id?horse_id={XSS_PAYLOAD}")
        self.assertNotEqual(r.status_code, 500)
        self.assertTrue(_no_html_echo(r, XSS_PAYLOAD))

    def test_suggest_default_no_500(self):
        """prefix なしデフォルト呼び出しは正常動作。"""
        r = client.get("/api/bloodline-cluster/suggest")
        self.assertNotIn(r.status_code, [500, 502])

    def test_suggest_limit_too_small_422(self):
        """limit=0 は ge=1 に違反 → 422。"""
        r = client.get("/api/bloodline-cluster/suggest?limit=0")
        self.assertEqual(r.status_code, 422)

    def test_suggest_limit_too_large_422(self):
        """limit=100 は le=50 に違反 → 422。"""
        r = client.get("/api/bloodline-cluster/suggest?limit=100")
        self.assertEqual(r.status_code, 422)

    def test_suggest_limit_string_422(self):
        """limit=abc は int 型エラー → 422。"""
        r = client.get("/api/bloodline-cluster/suggest?limit=abc")
        self.assertEqual(r.status_code, 422)

    def test_suggest_xss_prefix_no_echo(self):
        r = client.get(f"/api/bloodline-cluster/suggest?prefix={XSS_PAYLOAD}")
        self.assertNotEqual(r.status_code, 500)
        self.assertTrue(_no_html_echo(r, XSS_PAYLOAD))


# ═══════════════════════════════════════════════════════
# A. pedigree-map/condition-ranking — パラメータ範囲検証
# ═══════════════════════════════════════════════════════

class TestPedigreeConditionRanking(unittest.TestCase):
    """条件ランキング — パラメータ範囲違反は 422 を返す。"""

    def test_default_call_no_500(self):
        r = client.get("/api/pedigree-map/condition-ranking")
        self.assertNotIn(r.status_code, [500, 502])
        self.assertTrue(_is_json(r))

    def test_min_n_zero_is_422(self):
        """min_n=0 は ge=1 に違反 → 422。"""
        r = client.get("/api/pedigree-map/condition-ranking?min_n=0")
        self.assertEqual(r.status_code, 422)

    def test_top_over_max_is_422(self):
        """top=999 は le=200 に違反 → 422。"""
        r = client.get("/api/pedigree-map/condition-ranking?top=999")
        self.assertEqual(r.status_code, 422)

    def test_top_zero_is_422(self):
        """top=0 は ge=1 に違反 → 422。"""
        r = client.get("/api/pedigree-map/condition-ranking?top=0")
        self.assertEqual(r.status_code, 422)

    def test_min_n_string_is_422(self):
        r = client.get("/api/pedigree-map/condition-ranking?min_n=abc")
        self.assertEqual(r.status_code, 422)

    def test_xss_in_side_no_500(self):
        r = client.get(f"/api/pedigree-map/condition-ranking?side={XSS_PAYLOAD}")
        self.assertNotEqual(r.status_code, 500)
        self.assertTrue(_no_html_echo(r, XSS_PAYLOAD))

    def test_sqli_in_venues_no_500(self):
        r = client.get(f"/api/pedigree-map/condition-ranking?venues={SQLI_PAYLOAD}")
        self.assertNotEqual(r.status_code, 500)


# ═══════════════════════════════════════════════════════
# A. sire-heatmap — 必須 stallion_id
# ═══════════════════════════════════════════════════════

class TestSireHeatmap(unittest.TestCase):
    """sire-heatmap — stallion_id 必須。"""

    def test_missing_stallion_id_returns_422(self):
        r = client.get("/api/bloodline-cluster/sire-heatmap")
        self.assertEqual(r.status_code, 422)

    def test_fake_stallion_id_no_500(self):
        r = client.get(f"/api/bloodline-cluster/sire-heatmap?stallion_id={FAKE_HORSE_ID}")
        self.assertNotEqual(r.status_code, 500)

    def test_xss_stallion_id_no_echo(self):
        r = client.get(f"/api/bloodline-cluster/sire-heatmap?stallion_id={XSS_PAYLOAD}")
        self.assertNotEqual(r.status_code, 500)
        self.assertTrue(_no_html_echo(r, XSS_PAYLOAD))

    def test_long_stallion_id_no_500(self):
        r = client.get(f"/api/bloodline-cluster/sire-heatmap?stallion_id={LONG_STRING}")
        self.assertNotEqual(r.status_code, 500)


# ═══════════════════════════════════════════════════════
# E. track-speed — 200 + JSON
# ═══════════════════════════════════════════════════════

class TestTrackSpeedEndpoints(unittest.TestCase):
    """track-speed 系 — 基本エンドポイントが 5xx にならない。"""

    def test_meta_no_500(self):
        r = client.get("/api/track-speed/meta")
        self.assertNotIn(r.status_code, [500, 502])
        self.assertTrue(_is_json(r))

    def test_status_no_500(self):
        r = client.get("/api/track-speed/status")
        self.assertNotIn(r.status_code, [500, 502])
        self.assertTrue(_is_json(r))

    def test_dates_no_500(self):
        r = client.get("/api/track-speed/dates")
        self.assertNotIn(r.status_code, [500, 502])
        self.assertTrue(_is_json(r))

    def test_venues_missing_date_returns_422(self):
        """date パラメータ必須 → 欠損で 422。"""
        r = client.get("/api/track-speed/venues")
        self.assertEqual(r.status_code, 422)

    def test_venues_invalid_date_no_500(self):
        r = client.get("/api/track-speed/venues?date=invalid-date")
        self.assertNotEqual(r.status_code, 500)

    def test_day_missing_date_returns_422(self):
        r = client.get("/api/track-speed/day")
        self.assertEqual(r.status_code, 422)

    def test_day_invalid_date_no_500(self):
        r = client.get("/api/track-speed/day?date=invalid-date")
        self.assertNotEqual(r.status_code, 500)

    def test_day_xss_date_no_echo(self):
        r = client.get(f"/api/track-speed/day?date={XSS_PAYLOAD}")
        self.assertNotEqual(r.status_code, 500)
        self.assertTrue(_no_html_echo(r, XSS_PAYLOAD))

    def test_day_future_date_no_500(self):
        r = client.get("/api/track-speed/day?date=20991231")
        self.assertNotEqual(r.status_code, 500)


# ═══════════════════════════════════════════════════════
# E. pedigree-map/tags — 200 + JSON
# ═══════════════════════════════════════════════════════

class TestPedigreeMapTags(unittest.TestCase):
    """pedigree-map/tags — アーティファクト未生成でも 5xx にならない。"""

    def test_tags_no_500(self):
        r = client.get("/api/pedigree-map/tags")
        self.assertNotIn(r.status_code, [500, 502])
        self.assertTrue(_is_json(r))

    def test_tags_min_lift_string_422(self):
        """min_lift=abc は float 型エラー → 422。"""
        r = client.get("/api/pedigree-map/tags?min_lift=abc")
        self.assertEqual(r.status_code, 422)

    def test_tags_min_lift_negative_no_500(self):
        """min_lift=-1.0 は FastAPI で範囲制限なし → 422 ではなく動作する。"""
        r = client.get("/api/pedigree-map/tags?min_lift=-1.0")
        self.assertNotIn(r.status_code, [500, 502])


# ═══════════════════════════════════════════════════════
# A. /api/horse-names/search — パラメータ検証
# ═══════════════════════════════════════════════════════

class TestHorseNameSearch(unittest.TestCase):
    """horse-names/search — 空クエリ・不正入力ハンドリング。"""

    def test_empty_query_returns_empty_results(self):
        r = client.get("/api/horse-names/search")
        self.assertNotEqual(r.status_code, 500)
        self.assertTrue(_is_json(r))
        data = r.json()
        # q が空なら空リストを返す仕様
        if r.status_code == 200:
            self.assertIn("results", data)
            self.assertEqual(data["results"], [])

    def test_xss_query_no_html_echo(self):
        r = client.get(f"/api/horse-names/search?q={XSS_PAYLOAD}")
        self.assertNotEqual(r.status_code, 500)
        self.assertTrue(_no_html_echo(r, XSS_PAYLOAD))

    def test_unicode_query_no_500(self):
        r = client.get(f"/api/horse-names/search?q={UNICODE_PAYLOAD}")
        self.assertNotEqual(r.status_code, 500)

    def test_long_query_no_500(self):
        r = client.get(f"/api/horse-names/search?q={LONG_STRING}")
        self.assertNotEqual(r.status_code, 500)

    def test_limit_string_422(self):
        r = client.get("/api/horse-names/search?q=test&limit=abc")
        self.assertEqual(r.status_code, 422)

    def test_limit_negative_no_crash(self):
        """limit=-1 に対して FastAPI がエラーまたは空リストを返す。"""
        r = client.get("/api/horse-names/search?q=test&limit=-1")
        self.assertNotEqual(r.status_code, 500)

    def test_sqli_in_query_no_500(self):
        r = client.get(f"/api/horse-names/search?q={SQLI_PAYLOAD}")
        self.assertNotEqual(r.status_code, 500)


# ═══════════════════════════════════════════════════════
# A. /api/pedigree-race-stats/query — パラメータ検証
# ═══════════════════════════════════════════════════════

class TestPedigreeRaceStatsQuery(unittest.TestCase):
    """pedigree-race-stats/query — パラメータ型・範囲エラー。"""

    def test_default_call_no_500(self):
        r = client.get("/api/pedigree-race-stats/query")
        self.assertNotIn(r.status_code, [500, 502])
        self.assertTrue(_is_json(r))

    def test_dist_min_string_is_422(self):
        r = client.get("/api/pedigree-race-stats/query?dist_min=abc")
        self.assertEqual(r.status_code, 422)

    def test_dist_max_string_is_422(self):
        r = client.get("/api/pedigree-race-stats/query?dist_max=abc")
        self.assertEqual(r.status_code, 422)

    def test_top_n_string_is_422(self):
        r = client.get("/api/pedigree-race-stats/query?top_n=abc")
        self.assertEqual(r.status_code, 422)

    def test_xss_in_venues_no_500(self):
        r = client.get(f"/api/pedigree-race-stats/query?venues={XSS_PAYLOAD}")
        self.assertNotEqual(r.status_code, 500)
        self.assertTrue(_no_html_echo(r, XSS_PAYLOAD))

    def test_sqli_in_surface_no_500(self):
        r = client.get(f"/api/pedigree-race-stats/query?surface={SQLI_PAYLOAD}")
        self.assertNotEqual(r.status_code, 500)

    def test_unicode_in_venues_no_500(self):
        r = client.get(f"/api/pedigree-race-stats/query?venues={UNICODE_PAYLOAD}")
        self.assertNotEqual(r.status_code, 500)


# ═══════════════════════════════════════════════════════
# A. /api/cushion/* — パラメータ検証
# ═══════════════════════════════════════════════════════

class TestCushionEndpoints(unittest.TestCase):
    """cushion/* — データなし / 型エラーのハンドリング。"""

    def test_cushion_data_no_crash(self):
        """データファイル未生成でも 404 を返す (5xx NG)。"""
        r = client.get("/api/cushion/data")
        self.assertNotIn(r.status_code, [500, 502])
        self.assertTrue(_is_json(r))

    def test_cushion_stats_no_crash(self):
        r = client.get("/api/cushion/stats")
        self.assertNotIn(r.status_code, [500, 502])
        self.assertTrue(_is_json(r))

    def test_cushion_scrape_status_no_crash(self):
        r = client.get("/api/cushion/scrape/status")
        self.assertNotIn(r.status_code, [500, 502])
        self.assertTrue(_is_json(r))

    def test_cushion_data_year_string_is_422(self):
        """year=abc は int 型エラー → 422。"""
        r = client.get("/api/cushion/data?year=abc")
        self.assertEqual(r.status_code, 422)

    def test_cushion_data_xss_venue_no_echo(self):
        r = client.get(f"/api/cushion/data?venue_name={XSS_PAYLOAD}")
        self.assertNotIn(r.status_code, [500, 502])
        self.assertTrue(_no_html_echo(r, XSS_PAYLOAD))

    def test_cushion_data_sqli_venue_code_no_500(self):
        r = client.get(f"/api/cushion/data?venue_code={SQLI_PAYLOAD}")
        self.assertNotIn(r.status_code, [500, 502])


# ═══════════════════════════════════════════════════════
# B. /api/odds/{history,predict} — race_id 不正フォーマット
# ═══════════════════════════════════════════════════════

class TestOddsEndpoints(unittest.TestCase):
    """odds/history・predict — 不正 race_id で 500 が出ない。"""

    def test_odds_history_nonexistent_no_500(self):
        r = client.get(f"/api/odds/history/{FAKE_RACE_ID}")
        self.assertNotEqual(r.status_code, 500)
        self.assertTrue(_is_json(r))

    def test_odds_history_empty_snapshots(self):
        """存在しない race_id → snapshots 空リストを返す。"""
        r = client.get(f"/api/odds/history/{FAKE_RACE_ID}")
        if r.status_code == 200:
            data = r.json()
            self.assertIn("snapshots", data)

    def test_odds_history_xss_no_500(self):
        r = client.get(f"/api/odds/history/{XSS_PAYLOAD}")
        self.assertNotEqual(r.status_code, 500)
        self.assertTrue(_no_html_echo(r, XSS_PAYLOAD))

    def test_odds_history_long_id_no_500(self):
        # BUG: get_odds_history() calls `ODDS_HISTORY_DIR / f"{race_id}.json"` then
        # `history_path.exists()` which raises OSError: File name too long for IDs > ~240 chars.
        # Fix: add `if len(race_id) > 200: return JSONResponse({"snapshots": [], ...})` guard.
        r = client.get(f"/api/odds/history/{LONG_STRING}")
        self.assertNotEqual(r.status_code, 500)

    def test_odds_predict_nonexistent_no_crash(self):
        """特徴量構築失敗 → 400 or 500 だが、テストは「500 でないこと」より
        「サーバがクラッシュしないこと」に注目。"""
        r = client.get(f"/api/odds/predict/{FAKE_RACE_ID}")
        # 400/500 のどちらかはあり得るが、サーバ自体は応答を返す
        self.assertTrue(_is_json(r))


# ═══════════════════════════════════════════════════════
# E. /api/stallion-sire-tree — 200 or graceful error
# ═══════════════════════════════════════════════════════

class TestStallionSireTree(unittest.TestCase):
    """stallion-sire-tree — アーティファクト未生成でも 5xx にならない。"""

    def test_stallion_sire_tree_no_500(self):
        r = client.get("/api/stallion-sire-tree")
        self.assertNotIn(r.status_code, [500, 502])
        self.assertTrue(_is_json(r))


# ═══════════════════════════════════════════════════════
# F. POST エンドポイント — 不正ボディ
# ═══════════════════════════════════════════════════════

class TestPostEndpointsInvalidBody(unittest.TestCase):
    """POST エンドポイントへ空/不正 JSON を送っても 5xx にならない。"""

    def test_scrape_queue_add_empty_body_returns_400_or_422(self):
        """race_id が必須。空 JSON → 400 または 422。"""
        r = client.post("/api/scrape-queue/add", json={})
        # auth が必要なエンドポイントは 401 も許容
        self.assertIn(r.status_code, [400, 401, 422])

    def test_scrape_queue_add_wrong_type_no_500(self):
        """race_id が int 型 → 受け入れるか 400/422 を返す。"""
        r = client.post("/api/scrape-queue/add", json={"race_id": 12345})
        self.assertNotEqual(r.status_code, 500)

    def test_scrape_queue_add_extra_fields_no_crash(self):
        """未知フィールドは無視されるべき。"""
        r = client.post(
            "/api/scrape-queue/add",
            json={"race_id": FAKE_RACE_ID, "unknown_field": "value"}
        )
        # 401 (auth) or 2xx/4xx は OK
        self.assertNotEqual(r.status_code, 500)

    def test_bloodline_analyze_empty_body_no_500(self):
        r = client.post("/api/bloodline/analyze", json={})
        self.assertNotIn(r.status_code, [500, 502])

    def test_bloodline_analyze_null_body_no_500(self):
        # BUG: api_bloodline_analyze() calls `body = await request.json()` which returns None
        # for a `null` JSON body, then `body.get("years")` raises AttributeError.
        # Fix: add `if not isinstance(body, dict): body = {}` guard after json() call.
        r = client.post(
            "/api/bloodline/analyze",
            content=b"null",
            headers={"Content-Type": "application/json"},
        )
        self.assertNotIn(r.status_code, [500, 502])

    def test_cushion_scrape_empty_body_no_crash(self):
        """cushion/scrape は body.years が None でも動作 (auth が通れば)。"""
        # BUG: api_cushion_scrape() calls `_cushion_job.update(... started_at=time.time())`
        # but `time` is not imported in this module — the module uses `_time` alias.
        # Fix: replace `time.time()` with `_time.time()` at line 9464.
        r = client.post("/api/cushion/scrape", json={})
        # 409 (already running) / 2xx / 401 は全部 OK
        self.assertNotEqual(r.status_code, 500)

    def test_cushion_scrape_years_wrong_type_no_500(self):
        # BUG: same `time` NameError as above triggers on any call to this endpoint.
        r = client.post("/api/cushion/scrape", json={"years": "not-a-list"})
        self.assertNotEqual(r.status_code, 500)


# ═══════════════════════════════════════════════════════
# D. クエリパラメータインジェクション共通テスト
# ═══════════════════════════════════════════════════════

class TestQueryParamInjection(unittest.TestCase):
    """各種エンドポイントへ XSS / SQLi を混入させても HTML エコーバックしない。"""

    def test_race_list_xss_date_no_echo(self):
        r = client.get(f"/api/race-list/{XSS_PAYLOAD}")
        self.assertTrue(_no_html_echo(r, XSS_PAYLOAD))

    def test_horse_detail_xss_id_no_echo(self):
        r = client.get(f"/api/horse/{XSS_PAYLOAD}/detail")
        self.assertTrue(_no_html_echo(r, XSS_PAYLOAD))

    def test_data_endpoint_xss_no_echo(self):
        r = client.get(f"/api/data/{XSS_PAYLOAD}/key")
        self.assertTrue(_no_html_echo(r, XSS_PAYLOAD))

    def test_person_xss_ptype_no_echo(self):
        r = client.get(f"/api/person/{XSS_PAYLOAD}/12345/stats")
        self.assertTrue(_no_html_echo(r, XSS_PAYLOAD))

    def test_horse_names_search_xss_no_echo(self):
        r = client.get(f"/api/horse-names/search?q={XSS_PAYLOAD}")
        self.assertTrue(_no_html_echo(r, XSS_PAYLOAD))

    def test_pedigree_query_xss_venues_no_echo(self):
        r = client.get(f"/api/pedigree-race-stats/query?venues={XSS_PAYLOAD}")
        self.assertTrue(_no_html_echo(r, XSS_PAYLOAD))

    def test_condition_ranking_xss_side_no_echo(self):
        r = client.get(f"/api/pedigree-map/condition-ranking?side={XSS_PAYLOAD}")
        self.assertTrue(_no_html_echo(r, XSS_PAYLOAD))


# ═══════════════════════════════════════════════════════
# B. /api/growth-curve limit パラメータ
# ═══════════════════════════════════════════════════════

class TestGrowthCurveParams(unittest.TestCase):
    """growth-curve — limit パラメータ検証。"""

    def test_limit_string_is_422(self):
        r = client.get(f"/api/growth-curve/{FAKE_HORSE_ID}?limit=abc")
        self.assertEqual(r.status_code, 422)

    def test_limit_negative_no_500(self):
        """limit=-1: FastAPI は int 型なら受け入れる。"""
        r = client.get(f"/api/growth-curve/{FAKE_HORSE_ID}?limit=-1")
        self.assertNotEqual(r.status_code, 500)

    def test_fetch_speed_index_string_is_422(self):
        """bool 型に不正文字列 → 422。"""
        r = client.get(f"/api/growth-curve/{FAKE_HORSE_ID}?fetch_speed_index=maybe")
        self.assertEqual(r.status_code, 422)


if __name__ == "__main__":
    unittest.main()
