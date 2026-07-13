"""めぐ指数 common / predict / field_quality のユニットテスト。"""

import unittest

import pandas as pd

from src.pipeline.megu_index.common import (
    adjusted_time_to_megu,
    dist_band,
    megu_to_adjusted_time,
)
from src.pipeline.megu_index.field_quality import (
    attach_fq_and_delta_level,
    build_career_yen_before_race,
    compute_par_log_fq,
)
from src.pipeline.megu_index.predict import predict_megu_scores, weight_megu_delta
from src.pipeline.megu_index.predict_params import PredictTuning


class TestMeguCommon(unittest.TestCase):
    def test_dist_band_area11(self):
        self.assertEqual(dist_band(1400), "sprint")
        self.assertEqual(dist_band(1499), "sprint")
        self.assertEqual(dist_band(1500), "mile")
        self.assertEqual(dist_band(2400), "long")

    def test_megu_adj_roundtrip(self):
        par = 96.0
        megu = 110.0
        adj = megu_to_adjusted_time(megu, par)
        self.assertAlmostEqual(adj, 95.0, places=1)
        self.assertAlmostEqual(adjusted_time_to_megu(adj, par), megu, places=1)


class TestMeguPredict(unittest.TestCase):
    def test_one_point_equals_point_one_sec(self):
        par = 100.0
        h1 = [{"megu_index": 105.0, "par_time_sec": 100.0, "surface": "芝", "distance": 1600}]
        h2 = [{"megu_index": 104.0, "par_time_sec": 100.0, "surface": "芝", "distance": 1600}]
        p1 = predict_megu_scores(
            h1, par_time_target=par, surface_target="芝", distance_target=1600,
            jockey_weight=55.0, sex_age="牡4", beta_weight=0.612596, transfer_map={},
        )
        p2 = predict_megu_scores(
            h2, par_time_target=par, surface_target="芝", distance_target=1600,
            jockey_weight=55.0, sex_age="牡4", beta_weight=0.612596, transfer_map={},
        )
        diff = (p1["megu_final"] or 0) - (p2["megu_final"] or 0)
        self.assertAlmostEqual(diff, 1.0, places=1)

    def test_weight_delta_applied_to_final(self):
        base = predict_megu_scores(
            [{"megu_index": 100.0, "par_time_sec": 96.0, "surface": "芝", "distance": 2000}],
            par_time_target=96.0, surface_target="芝", distance_target=2000,
            jockey_weight=55.0, sex_age="牡4", beta_weight=0.612596, transfer_map={},
        )
        heavy = predict_megu_scores(
            [{"megu_index": 100.0, "par_time_sec": 96.0, "surface": "芝", "distance": 2000}],
            par_time_target=96.0, surface_target="芝", distance_target=2000,
            jockey_weight=57.0, sex_age="牡4", beta_weight=0.612596, transfer_map={},
        )
        self.assertLess((heavy["megu_final"] or 0), (base["megu_final"] or 0))
        self.assertLess((heavy["weight_megu_delta"] or 0), (base["weight_megu_delta"] or 0))

    def test_par_normalized_at_target_class(self):
        hist = [{"megu_index": 100.0, "par_time_sec": 72.0, "surface": "芝", "distance": 1600}]
        pred = predict_megu_scores(
            hist, par_time_target=68.0, surface_target="芝", distance_target=1600,
            jockey_weight=55.0, sex_age="牡4", beta_weight=0.0, transfer_map={},
            tuning=PredictTuning(par_blend=1.0),
        )
        # ability_adj=72.0 → target par 68 → megu 60
        self.assertAlmostEqual(pred["base_megu"], 60.0, places=1)

    def test_ability_bias_sec_shifts_megu_linearly(self):
        hist = [{"megu_index": 100.0, "par_time_sec": 96.0, "surface": "芝", "distance": 1600}]
        base = predict_megu_scores(
            hist, par_time_target=96.0, surface_target="芝", distance_target=1600,
            jockey_weight=55.0, sex_age="牡4", beta_weight=0.0, transfer_map={},
            tuning=PredictTuning(par_blend=0.0, ability_bias_sec=0.0),
        )
        shifted = predict_megu_scores(
            hist, par_time_target=96.0, surface_target="芝", distance_target=1600,
            jockey_weight=55.0, sex_age="牡4", beta_weight=0.0, transfer_map={},
            tuning=PredictTuning(par_blend=0.0, ability_bias_sec=1.0),
        )
        self.assertAlmostEqual((shifted["base_megu"] or 0) - (base["base_megu"] or 0), 10.0, places=1)

    def test_extreme_transfer_ignored_when_low_sample(self):
        hist = [{"megu_index": 100.0, "par_time_sec": 96.0, "surface": "芝", "distance": 1600}]
        tr = {("芝", "mile", "ダート", "long"): {"delta_sec": -15.0, "sample_count": 8}}
        pred = predict_megu_scores(
            hist, par_time_target=96.0, surface_target="ダート", distance_target=2400,
            jockey_weight=55.0, sex_age="牡4", beta_weight=0.612596, transfer_map=tr,
        )
        self.assertEqual(pred["megu_final"], pred["base_megu"])
        self.assertEqual(pred["megu_adjusted"], pred["base_megu"])

    def test_condition_transfer_applied_to_final(self):
        hist = [{"megu_index": 100.0, "par_time_sec": 96.0, "surface": "ダート", "distance": 1600}]
        tr = {("ダート", "mile", "芝", "mile"): {"delta_sec": 0.5, "sample_count": 100}}
        pred = predict_megu_scores(
            hist, par_time_target=96.0, surface_target="芝", distance_target=1600,
            jockey_weight=55.0, sex_age="牡4", beta_weight=0.0, transfer_map=tr,
            tuning=PredictTuning(transfer_strength=1.0),
        )
        self.assertAlmostEqual((pred["megu_final"] or 0) - (pred["base_megu"] or 0), 5.0, places=1)

    def test_condition_transfer_in_seconds(self):
        hist = [{"megu_index": 100.0, "par_time_sec": 96.0, "surface": "ダート", "distance": 1600}]
        tr = {("ダート", "mile", "芝", "mile"): {"delta_sec": 0.5, "sample_count": 100}}
        pred = predict_megu_scores(
            hist, par_time_target=96.0, surface_target="芝", distance_target=1600,
            jockey_weight=55.0, sex_age="牡4", beta_weight=0.0, transfer_map=tr,
            tuning=PredictTuning(transfer_strength=1.0),
        )
        self.assertAlmostEqual((pred["megu_adjusted"] or 0) - (pred["base_megu"] or 0), 5.0, places=1)
        self.assertAlmostEqual((pred["megu_final"] or 0) - (pred["base_megu"] or 0), 5.0, places=1)


class TestFieldQuality(unittest.TestCase):
    def test_career_yen_before_is_leak_safe(self):
        df = pd.DataFrame([
            {"horse_id": "h1", "race_id": "r1", "date": "2025-01-01", "finish_pos": 1,
             "grade": "未勝利", "race_name": "未勝利", "race_class": ""},
            {"horse_id": "h1", "race_id": "r2", "date": "2025-02-01", "finish_pos": 1,
             "grade": "未勝利", "race_name": "未勝利", "race_class": ""},
        ])
        cb = build_career_yen_before_race(df)
        r2 = cb[cb["race_id"] == "r2"].iloc[0]
        self.assertGreater(r2["career_yen_before"], 0)

    def test_attach_fq_adds_field_quality(self):
        hist = pd.DataFrame([
            {"race_id": "r0", "horse_id": "h1", "date": "2025-01-01", "finish_pos": 1,
             "grade": "未勝利", "race_name": "未勝利", "race_class": ""},
            {"race_id": "r0", "horse_id": "h2", "date": "2025-01-01", "finish_pos": 2,
             "grade": "未勝利", "race_name": "未勝利", "race_class": ""},
        ])
        df = pd.DataFrame([
            {"race_id": "r1", "horse_id": "h1", "date": "2025-03-01", "finish_pos": 1,
             "grade": "1勝", "race_name": "1勝", "race_class": ""},
            {"race_id": "r1", "horse_id": "h2", "date": "2025-03-01", "finish_pos": 2,
             "grade": "1勝", "race_name": "1勝", "race_class": ""},
        ])
        out = attach_fq_and_delta_level(
            df, hist, beta_level=-0.93,
            par_log_fq=compute_par_log_fq(pd.Series([5_000_000])),
        )
        self.assertTrue(out["field_quality"].notna().any())


if __name__ == "__main__":
    unittest.main()
