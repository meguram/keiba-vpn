"""追走難度: 通過順パースとペース予想。"""
from __future__ import annotations

import unittest

from src.pipeline.models.tracking_difficulty import (
    _assign_unique_positions,
    _build_early_forward_composite,
    _classify_style_jra,
    _filter_history_before_race,
    _horse_last3f_ability,
    _parse_passing,
    _race_passing_metrics,
    blend_form_style,
    build_horse_profile,
    build_neighbor_gate_factors,
    extract_prev_race_features,
    extract_recent3_features,
    predict_expected_last_3f,
    predict_position_flow,
    predict_race_pace,
)


class TestPassingMetrics(unittest.TestCase):
    def test_nige_jra_any_corner_before_final(self):
        m = _race_passing_metrics(_parse_passing("2-3-4-5", 16), 16)
        self.assertNotEqual(m["style_jra"], "逃げ")
        self.assertEqual(m["ever_led_before_final"], 0)

        m2 = _race_passing_metrics(_parse_passing("1-2-3-4", 16), 16)
        self.assertEqual(m2["style_jra"], "逃げ")
        self.assertEqual(m2["ever_led_before_final"], 1)
        self.assertEqual(m2["best_raw"], 1)

    def test_best_norm_across_corners(self):
        m = _race_passing_metrics(_parse_passing("8-6-3-2", 16), 16)
        self.assertLess(m["best_norm"], m["t1f_norm"])
        self.assertEqual(m["best_raw"], 2)

    def test_build_horse_profile_ever_led(self):
        hist = [
            {"passing_order": "1-2-3-4", "field_size": 16, "surface": "芝", "distance": 1600},
            {"passing_order": "10-8-6-5", "field_size": 16, "surface": "芝", "distance": 1600},
        ]
        p = build_horse_profile(hist)
        self.assertEqual(p["ever_led_any"], 1)
        self.assertLess(p["best_norm_pos_min"], 0.2)


class TestParsePassing(unittest.TestCase):
    def test_hyphenated(self):
        self.assertEqual(_parse_passing("3-4-4-2", 16), [3, 4, 4, 2])

    def test_compact_two_digit(self):
        self.assertEqual(_parse_passing("1055", 16), [10, 5, 5])

    def test_single_digit(self):
        self.assertEqual(_parse_passing("3", 18), [3])


class TestExtractPrevRaceFeatures(unittest.TestCase):
    def _history(self):
        return [
            {
                "date": "2026/05/17",
                "race_id": "202605170101",
                "field_size": 16,
                "passing_order": "2-2-3-1",
                "last_3f": 33.5,
                "finish_position": 1,
            },
            {
                "date": "2026/03/22",
                "race_id": "202603220501",
                "field_size": 18,
                "passing_order": "5-4-4-3",
                "last_3f": 34.2,
                "finish_position": 4,
            },
        ]

    def test_excludes_race_on_or_after_analysis_date(self):
        feat = extract_prev_race_features(
            self._history(),
            before_date="20260517",
            exclude_race_id="202605170101",
        )
        self.assertEqual(feat["prev_t1f_raw"], 5)
        self.assertAlmostEqual(feat["prev_last3f"], 34.2)

    def test_enriches_passing_from_race_result_storage(self):
        class FakeStorage:
            def load(self, category, key):
                if category == "race_result" and key == "202605170101":
                    return {
                        "field_size": 16,
                        "entries": [
                            {
                                "horse_id": "H1",
                                "passing_order": "3-3-2-2",
                                "last_3f": 35.1,
                            }
                        ],
                    }
                return None

        history = [
            {
                "date": "2026/05/17",
                "race_id": "202605170101",
                "field_size": 16,
                "passing_order": "",
                "last_3f": 0,
            },
            {
                "date": "2026/03/22",
                "race_id": "202603220501",
                "field_size": 18,
                "passing_order": "4-4-3-3",
                "last_3f": 34.0,
            },
        ]
        feat = extract_prev_race_features(
            history,
            before_date="20260520",
            horse_id="H1",
            storage=FakeStorage(),
        )
        self.assertEqual(feat["prev_t1f_raw"], 3)
        self.assertAlmostEqual(feat["prev_last3f"], 35.1)

    def test_filter_history_before(self):
        rows = _filter_history_before_race(
            self._history(),
            before_date="20260517",
            exclude_race_id="",
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["race_id"], "202603220501")


class TestNeighborSandwich(unittest.TestCase):
    def test_sandwich_when_both_neighbors_prev_top3(self):
        profile = {"typical_norm_pos": 0.55, "style": "差し", "pos_std": 0.1}
        left_prev = {"prev_t1f_raw": 2, "prev_field_size": 16}
        right_prev = {"prev_t1f_raw": 3, "prev_field_size": 16}
        nf = build_neighbor_gate_factors(
            5, 16, profile, {}, {}, left_prev=left_prev, right_prev=right_prev
        )
        self.assertTrue(nf["sandwich_front"])
        self.assertGreaterEqual(nf["sandwich_push"], 0.15)
        self.assertEqual(nf["front_neighbor_count"], 2)

    def test_no_sandwich_when_one_neighbor_back(self):
        profile = {"typical_norm_pos": 0.55, "style": "差し", "pos_std": 0.1}
        nf = build_neighbor_gate_factors(
            5, 16, profile, {}, {},
            left_prev={"prev_t1f_raw": 2, "prev_field_size": 16},
            right_prev={"prev_t1f_raw": 10, "prev_field_size": 16},
        )
        self.assertFalse(nf["sandwich_front"])
        self.assertEqual(nf["front_neighbor_count"], 1)


class TestRecent3Features(unittest.TestCase):
    def test_weighted_average_of_three(self):
        history = [
            {"date": "2026/05/01", "race_id": "R1", "field_size": 16, "passing_order": "2-2-2-1"},
            {"date": "2026/04/01", "race_id": "R2", "field_size": 16, "passing_order": "8-7-6-5"},
            {"date": "2026/03/01", "race_id": "R3", "field_size": 16, "passing_order": "10-9-8-7"},
        ]
        r3 = extract_recent3_features(history, before_date="20260601")
        self.assertEqual(r3["n_races"], 3)
        self.assertEqual(r3["t1f_raws"], [2, 8, 10])
        self.assertGreater(r3["t1f_raw_avg"], 2)
        self.assertLess(r3["t1f_raw_avg"], 8)
        self.assertIn(r3["style"], ("逃げ", "先行", "差し", "追込"))

    def test_blend_form_style_prefers_recent3(self):
        self.assertEqual(blend_form_style("逃げ", "差し", n_recent=3), "差し")


class TestPredictRacePace(unittest.TestCase):
    def test_uses_baseline_when_profiles_empty(self):
        race_info = {
            "venue": "東京",
            "surface": "芝",
            "distance": 1600,
            "track_condition": "良",
            "race_name": "ヴィクトリアマイル",
            "grade": "OP",
        }
        entries = [{"horse_number": i} for i in range(1, 15)]
        profiles = {
            i: {
                "typical_norm_pos": 0.45,
                "style": "差し",
                "pos_std": 0.1,
                "n_races": 5,
            }
            for i in range(1, 15)
        }
        out = predict_race_pace(entries, profiles, race_info)
        self.assertIn(out["pace_type"], ("ハイ", "ミドル", "スロー"))
        fh3 = out["lap_times"]["first_3f"]
        self.assertGreater(fh3, 33.0)
        self.assertLess(fh3, 37.5)
        self.assertIn(
            out["pace_factors"]["baseline_source"],
            ("cohort+empirical", "cohort+empirical+G1"),
        )


class TestPositionFlowAllocation(unittest.TestCase):
    def test_predict_position_flow_after_zero_horse_number_normalize(self):
        from src.utils.race_entries import normalize_race_entries

        field_size = 4
        raw = [{"horse_number": 0, "horse_name": f"H{i}"} for i in range(1, field_size + 1)]
        entries = normalize_race_entries(raw)
        profiles = {
            i: {"typical_norm_pos": 0.3, "style": "先行", "pos_std": 0.1, "n_races": 2}
            for i in range(1, field_size + 1)
        }
        tracking = [
            {
                "horse_number": i,
                "tracking_difficulty": {"ease_pct": 50},
                "gate_factors": {"bracket_number": 1},
                "neighbor_factors": {},
                "prev_race": {},
            }
            for i in range(1, field_size + 1)
        ]
        out = predict_position_flow(
            entries, profiles, tracking, {"pace_type": "ミドル", "pace_index": 50},
        )
        self.assertEqual(len(out), field_size)

    def test_assign_unique_ranks(self):
        ranks = _assign_unique_positions({3: 0.9, 1: 0.1, 2: 0.5}, 3)
        self.assertEqual(ranks[1], 1)
        self.assertEqual(ranks[2], 2)
        self.assertEqual(ranks[3], 3)

    def test_predict_position_flow_covers_all_horses(self):
        field_size = 6
        entries = [{"horse_number": i, "horse_name": f"H{i}"} for i in range(1, field_size + 1)]
        profiles = {
            i: {
                "typical_norm_pos": (i - 1) / (field_size - 1),
                "style": "先行" if i <= 2 else "差し",
                "pos_std": 0.1,
                "n_races": 3,
            }
            for i in range(1, field_size + 1)
        }
        tracking = []
        for i in range(1, field_size + 1):
            tracking.append({
                "horse_number": i,
                "horse_name": f"H{i}",
                "tracking_difficulty": {"ease_pct": 50},
                "gate_factors": {"horse_number": i, "bracket_number": i, "gate_zone": 0},
                "neighbor_factors": {"sandwich_push": 0},
                "prev_race": {"pos_ratio": (i - 1) / (field_size - 1)},
            })
        field_prev = {
            "normalized_t1f": {i: (i - 1) / (field_size - 1) for i in range(1, field_size + 1)},
        }
        pace = {"pace_type": "ミドル", "pace_index": 50}
        out = predict_position_flow(
            entries, profiles, tracking, pace, field_prev_stats=field_prev,
        )
        self.assertEqual(len(out), field_size)
        for stage in ("early", "mid", "late"):
            positions = [r["positions"][stage]["position"] for r in out]
            self.assertEqual(sorted(positions), list(range(1, field_size + 1)))

    def test_composite_favors_t1f_and_ease_for_forward_slots(self):
        entries = [
            {"horse_number": 1, "horse_name": "A"},
            {"horse_number": 2, "horse_name": "B"},
        ]
        profiles = {
            1: {"typical_norm_pos": 0.2, "style": "逃げ", "pos_std": 0.1, "n_races": 3},
            2: {"typical_norm_pos": 0.7, "style": "追込", "pos_std": 0.1, "n_races": 3},
        }
        tracking = [
            {
                "horse_number": 1,
                "tracking_difficulty": {"ease_pct": 70},
                "gate_factors": {"bracket_number": 1, "gate_zone": 0},
                "neighbor_factors": {},
                "prev_race": {"t1f_raw": 2, "field_size": 16, "pos_ratio": 0.1},
            },
            {
                "horse_number": 2,
                "tracking_difficulty": {"ease_pct": 35},
                "gate_factors": {"bracket_number": 7, "gate_zone": 2},
                "neighbor_factors": {},
                "prev_race": {"t1f_raw": 12, "field_size": 16, "pos_ratio": 0.7},
            },
        ]
        td_by = {r["horse_number"]: r for r in tracking}
        composite = _build_early_forward_composite(
            entries, profiles, td_by,
            field_prev_stats={"normalized_t1f": {1: 0.1, 2: 0.8}},
            field_size=16,
            pace_type="ミドル",
        )
        ranks = _assign_unique_positions(composite, 2)
        self.assertEqual(ranks[1], 1)
        self.assertEqual(ranks[2], 2)


class TestExpectedLast3f(unittest.TestCase):
    def test_horse_last3f_ability_weighted(self):
        hist = [
            {"last_3f": 33.0},
            {"last_3f": 34.5},
        ]
        prof = _horse_last3f_ability(hist, prev_last3f=33.8)
        self.assertGreater(prof["n_samples"], 0)
        self.assertAlmostEqual(prof["ability_sec"], 33.5, delta=0.6)

    def test_position_flow_includes_expected_last_3f(self):
        field_size = 4
        entries = [
            {"horse_number": i, "horse_name": f"H{i}", "horse_id": f"id{i}"}
            for i in range(1, field_size + 1)
        ]
        profiles = {
            i: {"typical_norm_pos": 0.2 if i == 1 else 0.75, "style": "逃げ" if i == 1 else "追込", "pos_std": 0.1, "n_races": 3}
            for i in range(1, field_size + 1)
        }
        tracking = []
        for i in range(1, field_size + 1):
            tracking.append({
                "horse_number": i,
                "horse_name": f"H{i}",
                "horse_id": f"id{i}",
                "tracking_difficulty": {"ease_pct": 50},
                "gate_factors": {"bracket_number": i, "gate_zone": 0},
                "neighbor_factors": {"sandwich_push": 0},
                "prev_race": {
                    "last3f": 34.0,
                    "t1f_raw": 2 if i == 1 else 12,
                    "field_size": 16,
                    "pos_ratio": 0.1 if i == 1 else 0.7,
                },
                "course_comparison": {"dist_change": 0, "same_surface": True},
            })
        pace = {
            "pace_type": "ハイ",
            "pace_index": 62,
            "lap_times": {"first_3f": 34.0},
        }
        out = predict_position_flow(
            entries,
            profiles,
            tracking,
            pace,
            field_prev_stats={"normalized_t1f": {i: 0.1 if i == 1 else 0.8 for i in range(1, field_size + 1)}},
            race_info={"distance": 1600, "surface": "芝", "track_condition": "良"},
        )
        for r in out:
            l3f = r.get("expected_last_3f") or {}
            self.assertGreater(l3f.get("seconds", 0), 32)
            self.assertLess(l3f.get("seconds", 99), 39)
            self.assertIn("rank", l3f)
        ranks = {
            (x.get("expected_last_3f") or {}).get("rank")
            for x in out
            if (x.get("expected_last_3f") or {}).get("rank")
        }
        self.assertEqual(len(ranks), field_size)
        # 同じ上がり実績でもハイペースで前残りの逃げは後方差しより遅くなりやすい
        same_l3f = [
            x for x in out
            if (x.get("expected_last_3f") or {}).get("ability_sec") == 34.0
        ]
        if len(same_l3f) >= 2:
            nige = next(x for x in same_l3f if x.get("style") == "逃げ")
            oik = next(x for x in same_l3f if x.get("style") == "追込")
            self.assertGreater(
                nige["expected_last_3f"]["seconds"],
                oik["expected_last_3f"]["seconds"],
            )

    def test_predict_expected_last_3f_closer_faster_on_high_pace(self):
        baseline = 34.2
        base_row = {
            "positions": {
                "early": {"position_norm": 0.7},
                "late": {"position_norm": 0.2},
            },
            "flow_pattern": "追い込み",
            "stamina": {"stamina_index": 68},
        }
        front_row = {
            "positions": {
                "early": {"position_norm": 0.05},
                "late": {"position_norm": 0.1},
            },
            "flow_pattern": "微後退",
            "stamina": {"stamina_index": 35},
        }
        pace = {"pace_type": "ハイ", "pace_index": 65}
        race_info = {"distance": 1600, "surface": "芝", "track_condition": "良"}
        closer = predict_expected_last_3f(
            horse_number=2,
            profile={"style": "追込"},
            td_data={"prev_race": {"last3f": 33.5}, "course_comparison": {"dist_change": 0, "same_surface": True}},
            position_flow_row=base_row,
            pace_prediction=pace,
            race_info=race_info,
            baseline_sec=baseline,
            horse_ability={"ability_sec": 33.5, "n_samples": 2},
        )
        leader = predict_expected_last_3f(
            horse_number=1,
            profile={"style": "逃げ"},
            td_data={"prev_race": {"last3f": 34.0}, "course_comparison": {"dist_change": 0, "same_surface": True}},
            position_flow_row=front_row,
            pace_prediction=pace,
            race_info=race_info,
            baseline_sec=baseline,
            horse_ability={"ability_sec": 34.0, "n_samples": 2},
        )
        self.assertLess(closer["seconds"], leader["seconds"])


if __name__ == "__main__":
    unittest.main()
