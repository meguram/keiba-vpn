"""推論時の LayerA 列（serving_layer_a）が、学習側の関数と同じ値になることの確認（合成データ）。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pipeline.features.historical_entity_stats import build_history_feature_table
from src.pipeline.features.serving_layer_a import BIAS_COLS, OUTPUT_COLS, PRIOR_COLS, build_layer_a_features, same_day_results
from src.pipeline.features.track_bias_pedigree import build_track_bias_pedigree_features


def _synthetic(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    horses = [f"H{i:03d}" for i in range(40)]
    jockeys = [f"J{i:02d}" for i in range(14)]
    trainers = [f"T{i:02d}" for i in range(12)]
    rows = []
    for d in range(6):
        date = (pd.Timestamp("2024-05-04") + pd.Timedelta(days=[0, 1, 7, 8, 14, 15][d])).strftime("%Y-%m-%d")
        for vi, venue in enumerate(("05", "06")):
            for r in range(1, 5):
                rid = f"2024{venue}01{d + 1:02d}{r:02d}"
                hs = rng.choice(horses, 8, replace=False)
                js = rng.choice(jockeys, 8, replace=False)
                ts = rng.choice(trainers, 8, replace=False)
                fin = rng.permutation(8) + 1
                for i in range(8):
                    rows.append({
                        "race_id": rid, "horse_number": i + 1, "horse_id": hs[i], "jockey_id": js[i], "trainer_id": ts[i], "date": date,
                        "venue_code": venue, "surface": "芝" if (r + vi) % 2 else "ダート", "start_time": f"{9 + r}:{30 if vi else 10}",
                        "finish_position": int(fin[i]), "last_3f": float(rng.uniform(33, 38)), "track_condition": ["良", "稍重"][d % 2],
                        "distance": [1200, 1600, 2000, 2400][r - 1],
                    })
    return pd.DataFrame(rows)


def _card(full: pd.DataFrame, race_ids: list[str]) -> pd.DataFrame:
    c = full[full["race_id"].isin(race_ids)].copy()
    c["sex_age"] = "牡3"
    for col, v in {"direction": "右", "grade": "OP", "race_class": "3勝", "weather": "晴", "weight_rule": "別定", "course_type": "", "field_size": 8,
                   "round": 1, "bracket_number": 1, "jockey_weight": 55.0, "weight": 480, "weight_change": 0}.items():
        c[col] = v
    return c.drop(columns=["finish_position", "last_3f"])


def _train_features(full: pd.DataFrame) -> pd.DataFrame:
    hist = build_history_feature_table(full[["race_id", "horse_number", "horse_id", "jockey_id", "trainer_id", "date", "finish_position"]])
    rr = full[["race_id", "horse_number", "date", "venue_code", "surface", "start_time", "finish_position", "last_3f", "track_condition", "distance"]]
    sb = full[["race_id", "horse_number"]].assign(sire="", dam_sire="")
    tb = build_track_bias_pedigree_features(rr, sb)
    return hist.merge(tb, on=["race_id", "horse_number"], how="left")


class ServingLayerATest(unittest.TestCase):
    def _compare(self, got: pd.DataFrame, ref: pd.DataFrame, cols: list[str]) -> None:
        m = got.merge(ref, on=["race_id", "horse_number"], suffixes=("", "_ref"))
        self.assertEqual(len(m), len(got))
        for c in cols:
            a, b = m[c], m[c + "_ref"]
            if c in ("track_bias_cond_bucket", "track_bias_dist_band"):
                self.assertTrue((a.astype(str) == b.astype(str)).all(), c)
                continue
            a, b = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
            # 父/母父（空）の面別累計勝率は、学習側が同じレースで馬番の小さい馬の結果まで数える（合成データでは件数が少なく差が目立つ）
            loose = c.startswith(("sire", "dam_sire"))
            np.testing.assert_allclose(a, b, rtol=0.05 if loose else 1e-9, atol=0.01 if loose else 1e-9, equal_nan=True, err_msg=c)

    def test_last_race_matches_training(self):
        full = _synthetic()
        target = "202406010604"
        ref = _train_features(full)
        got = build_layer_a_features(_card(full, [target]), full[full["race_id"] != target])
        self.assertEqual(list(got.columns), ["race_id", "horse_id"] + OUTPUT_COLS)
        self._compare(got, ref, PRIOR_COLS + BIAS_COLS)

    def test_unfinished_earlier_race_does_not_leak_into_later_one(self):
        full = _synthetic(1)
        a, b = "202405010603", "202405010604"          # 同日・同場の連続した 2 レース（a が先）
        only_b = build_layer_a_features(_card(full, [b]), full[~full["race_id"].isin([a, b])])
        both = build_layer_a_features(_card(full, [a, b]), full[~full["race_id"].isin([a, b])])
        x = both[both["race_id"] == b].reset_index(drop=True)
        y = only_b.reset_index(drop=True)
        pd.testing.assert_frame_equal(x, y, check_exact=False)

    def test_results_of_target_race_are_ignored(self):
        full = _synthetic(2)
        target = "202405010504"
        card = _card(full, [target])
        a = build_layer_a_features(card, full[full["race_id"] != target])
        b = build_layer_a_features(card, full)               # 履歴に対象レースの結果が入っていても同じ
        pd.testing.assert_frame_equal(a, b)

    def test_first_race_of_day_has_no_same_day_prior(self):
        full = _synthetic(3)
        target = "202405010601"
        got = build_layer_a_features(_card(full, [target]), full[full["race_id"] != target])
        self.assertTrue(got["track_bias_winner_3f_same_day_prior"].isna().all())

    def test_first_day_without_any_finished_race(self):
        full = _synthetic(4)
        day = full[full["date"] == "2024-05-04"]["race_id"].unique().tolist()      # 履歴に前日も同日の結果も無い最初の日
        got = build_layer_a_features(_card(full, day), full[~full["race_id"].isin(day)])
        self.assertEqual(len(got), 8 * len(day))
        self.assertTrue(got["track_bias_winner_3f_same_day_prior"].isna().all())
        self.assertTrue(got["track_bias_cond_bucket"].isin(["good", "yielding_soft"]).all())

    def test_same_day_results_from_storage_and_no_double_count(self):
        full = _synthetic(5)
        a, b = "202405010603", "202405010604"

        class Storage:
            def load(self, category, rid, bypass_cache=False):
                r = full[full["race_id"] == rid]
                keep = ["horse_number", "horse_id", "jockey_id", "trainer_id", "finish_position", "last_3f"]
                top = r.iloc[0]
                return {"race_id": rid, "date": top["date"], "venue_code": top["venue_code"], "surface": top["surface"], "distance": int(top["distance"]),
                        "track_condition": top["track_condition"], "start_time": top["start_time"], "entries": r[keep].to_dict("records")}

        base = full[~full["race_id"].isin([a, b])]
        sd = same_day_results(Storage(), [a])
        self.assertEqual(len(sd), 8)
        via_storage = build_layer_a_features(_card(full, [b]), pd.concat([base, sd], ignore_index=True))
        direct = build_layer_a_features(_card(full, [b]), full[full["race_id"] != b])
        pd.testing.assert_frame_equal(via_storage, direct, check_exact=False)
        twice = build_layer_a_features(_card(full, [b]), pd.concat([full[full["race_id"] != b], sd], ignore_index=True))
        pd.testing.assert_frame_equal(twice, direct)

    def test_abbreviated_going_in_card_matches_result_wording(self):
        full = _synthetic(6)
        target = "202406010604"
        card = _card(full, [target]).assign(track_condition="稍")
        ref = build_layer_a_features(_card(full, [target]).assign(track_condition="稍重"), full[full["race_id"] != target])
        got = build_layer_a_features(card, full[full["race_id"] != target])
        for c in ("track_bias_cond_bucket", "track_bias_winner_3f_prev_day", "track_bias_winner_3f_weighted"):
            self.assertEqual(got[c].tolist() if got[c].dtype == object else got[c].fillna(-1).tolist(), ref[c].tolist() if ref[c].dtype == object else ref[c].fillna(-1).tolist())
        self.assertTrue((got["track_condition"] == "稍").all())             # 出力の track_condition は出馬表の表記のまま（学習表と同じ）


if __name__ == "__main__":
    unittest.main()
