"""research.race.race_quality_model のユニットテスト（ストレージなしで検証可能な部分）。"""

from __future__ import annotations

import unittest

import numpy as np

from src.research.race.race_quality_model import (
    build_entrants_aptitude_response,
    build_horse_aptitude_cache_payload,
    compute_pace_shape,
    distance_surface_segment,
    extract_lap_times_from_blob,
    get_horse_aptitude_cache,
    get_race_quality_meta,
    going_archetype_multiplier,
    history_distance_band,
    segment_archetype_prior,
    _column_minmax_nonneg,
    _fit_mixture,
    _hand_segment_archetype_prior,
    _nine_probs,
    _parse_lap_string,
)


class _FakeStorageMissingAptitude:
    """血統(horse_pedigree_5gen)・戦歴(horse_result)が未取得の馬を模すストレージ。

    GCS 未接続・未取得時と同じく load() は None を返す（HybridStorage.load の
    実挙動に合わせる）。entrants-aptitude のフォールバック経路を検証する。
    """

    RACE_WITH_MISSING_HORSES = "RACE_MISSING_TEST"

    def __init__(self) -> None:
        self.saved: dict[tuple[str, str], dict] = {}

    def load(self, category: str, key: str, bypass_cache: bool = False):
        if category == "race_result" and key == self.RACE_WITH_MISSING_HORSES:
            return {
                "race_name": "テスト(血統・戦歴欠損馬あり)",
                "field_size": 3,
                "distance": 2000,
                "surface": "芝",
                "track_condition": "良",
                "entries": [
                    {"horse_id": "2020999999", "horse_number": 1, "bracket_number": 1, "finish_position": 1},
                    {"horse_id": "2020888888", "horse_number": 2, "bracket_number": 2, "finish_position": 2},
                    {"horse_id": "2020777777", "horse_number": 3, "bracket_number": 3, "finish_position": 3},
                ],
            }
        # horse_result / horse_pedigree_5gen / race_index / race_barometer /
        # race_lap / race_result_lap / horse_race_quality_aptitude すべて未取得を模す
        return None

    def save(self, category: str, key: str, payload: dict) -> None:
        self.saved[(category, key)] = payload


class TestRaceQualityModel(unittest.TestCase):
    def test_distance_surface_segment(self):
        self.assertEqual(distance_surface_segment("芝", 1200), "芝_1200-1399")
        self.assertEqual(distance_surface_segment("ダート", 1800), "ダート_1800-1999")
        self.assertIn("障害", distance_surface_segment("障", 3200))

    def test_history_distance_band(self):
        self.assertEqual(history_distance_band(1200), "短距離")
        self.assertEqual(history_distance_band(1600), "マイル")
        self.assertEqual(history_distance_band(2000), "中距離")
        self.assertEqual(history_distance_band(2400), "中長距離")
        self.assertEqual(history_distance_band(3200), "長距離")

    def test_segment_prior_positive(self):
        w = segment_archetype_prior("芝_1200-1399")
        self.assertEqual(w.shape, (8,))
        self.assertTrue(np.all(w > 0))

    def test_hand_prior_matches_segment_without_json(self):
        a = _hand_segment_archetype_prior("ダート_1800-1999")
        b = segment_archetype_prior("ダート_1800-1999")
        np.testing.assert_array_almost_equal(a, b)

    def test_parse_lap_string(self):
        self.assertGreaterEqual(len(_parse_lap_string("12.3-11.4-10.5")), 3)

    def test_extract_lap_blob(self):
        blob = {"entries": [{"lap_times": "11.1-10.2-9.8"}]}
        v = extract_lap_times_from_blob(blob)
        self.assertGreaterEqual(len(v), 3)

    def test_going_multiplier_mud(self):
        m = going_archetype_multiplier("重")
        self.assertGreater(m[7], 1.0)
        m2 = going_archetype_multiplier("良")
        self.assertLess(m2[7], m[7])

    def test_pace_shape_grind(self):
        p = compute_pace_shape({"first_half_3f": 33.0, "second_half_3f": 36.0}, [], 1600)
        self.assertEqual(p["has_half_pace"], 1.0)
        self.assertGreater(p["grind_index"], 0.5)
        self.assertLess(p["burst_index"], 0.5)

    def test_pace_shape_even_laps(self):
        laps = [11.0, 11.1, 11.0, 11.05, 11.0]
        p = compute_pace_shape({}, laps, 2000)
        self.assertGreater(p["lap_evenness"], 0.5)

    def test_nnls_pipeline(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(0.1, 1.0, size=(10, 8))
        Xn = _column_minmax_nonneg(X)
        y = rng.uniform(0.2, 1.0, size=10)
        coef, r2, _ = _fit_mixture(Xn, y)
        self.assertEqual(coef.shape, (8,))
        probs, _ = _nine_probs(coef, r2, 10)
        self.assertEqual(len(probs), 9)
        self.assertLess(abs(sum(probs) - 1.0), 1e-5)

    def test_get_meta(self):
        m = get_race_quality_meta()
        self.assertEqual(m["version"], 1)
        self.assertEqual(len(m["axes"]), 9)
        self.assertIn("api", m)

    def test_aptitude_payload_missing_pedigree_and_history_has_fallback(self):
        """血統・戦歴が未取得(storage.load が None)でも例外を出さず既定値を返す。"""
        storage = _FakeStorageMissingAptitude()
        payload = build_horse_aptitude_cache_payload(
            storage, "2020999999", stats_data={"sires": {}, "axes": [], "meta": {}}
        )
        self.assertIsNotNone(payload)
        self.assertEqual(payload["pedigree_vector"], [0.0] * 8)
        self.assertEqual(payload["history"]["starts"], 0.0)
        self.assertEqual(payload["history"]["last3f_fast"], 0.5)

        cache, from_store = get_horse_aptitude_cache(
            storage, "2020999999", stats_data={"sires": {}, "axes": [], "meta": {}}
        )
        self.assertIsNotNone(cache)
        self.assertFalse(from_store)

    def test_entrants_aptitude_response_no_crash_when_all_entrants_missing_data(self):
        """出走馬全員の血統・戦歴が未取得でも entrants-aptitude がエラー落ちせず
        妥当なフォールバック(ゼロ寄りの8軸・既定の戦歴統計)を返す。"""
        storage = _FakeStorageMissingAptitude()
        resp = build_entrants_aptitude_response(
            storage,
            storage.RACE_WITH_MISSING_HORSES,
            stats_data={"sires": {}, "axes": [], "meta": {}},
        )
        self.assertIsNotNone(resp)
        self.assertEqual(len(resp["entrants"]), 3)
        self.assertEqual(resp["cache_hits"], 0)
        self.assertEqual(resp["cache_misses"], 3)
        for entrant in resp["entrants"]:
            self.assertEqual(len(entrant["base_row"]), 8)
            self.assertTrue(all(v >= 0.0 for v in entrant["base_row"]))

    def test_entrants_aptitude_response_none_for_unknown_race(self):
        """出馬表・結果が未取得の race_id は None（API 層で 404 として扱われる）。"""
        storage = _FakeStorageMissingAptitude()
        resp = build_entrants_aptitude_response(
            storage, "RACE_DOES_NOT_EXIST", stats_data={"sires": {}, "axes": [], "meta": {}}
        )
        self.assertIsNone(resp)


if __name__ == "__main__":
    unittest.main()
