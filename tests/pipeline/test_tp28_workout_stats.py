"""tp-28 追い切りパターン別の成績表: 払戻の解釈・as-of（将来の結果を変えても過去の行が変わらない／同日の結果を使わない）。"""

from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from src.pipeline.features import tp28_workout_stats as T
from src.pipeline.features import training_laps as tl


def _payoff(win_no, win_pay, place_nos, place_pays):
    return json.dumps({"単勝": {"numbers": str(win_no), "payout": str(win_pay)},
                       "複勝": {"numbers": "".join(map(str, place_nos)), "payout": "".join(str(p) for p in place_pays)}}, ensure_ascii=False)


class PayoffTest(unittest.TestCase):
    def test_win_and_place(self):
        fin = {3: 1, 4: 2, 2: 3, 1: 4}
        p = T.parse_payoff(_payoff(3, 480, [3, 4, 2], [130, 110, 140]), fin)
        self.assertAlmostEqual(p["win"][3], 4.8)
        self.assertEqual(p["place"], {3: 1.3, 4: 1.1, 2: 1.4})

    def test_two_digit_horse_numbers_and_thousand_separator(self):
        fin = {10: 1, 11: 2, 1: 3}
        p = T.parse_payoff(json.dumps({"単勝": {"numbers": "10", "payout": "1,230"}, "複勝": {"numbers": "10111", "payout": "250,1,020,110"}}), fin)
        self.assertAlmostEqual(p["win"][10], 12.3)
        self.assertEqual(p["place"], {10: 2.5, 11: 10.2, 1: 1.1})

    def test_unparsable_is_empty(self):
        self.assertEqual(T.parse_payoff("not json", {1: 1}), {"win": {}, "place": {}})
        self.assertEqual(T.parse_payoff(None, {1: 1}), {"win": {}, "place": {}})

    def test_runner_returns(self):
        res = pd.DataFrame({"race_id": ["r"] * 4, "horse_number": [1, 2, 3, 4], "finish_position": [4, 3, 1, 2]})
        out = T.runner_returns(res, {"r": _payoff(3, 480, [3, 4, 2], [130, 110, 140])})
        self.assertEqual(list(out["win_ret"]), [0.0, 0.0, 4.8, 0.0])
        self.assertEqual(list(out["place_ret"]), [0.0, 1.4, 1.3, 1.1])


def _laps(n_horses=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(n_horses):
        for race_date in pd.to_datetime(["2022-03-06", "2022-04-03", "2022-05-01", "2022-06-05", "2022-07-03"]):
            for days, leg in ((3, "一杯"), (10, "強め"), (17, "馬也")):
                t1 = 12.4 + rng.normal(0, 0.3)
                rows.append({"horse_id": f"h{h}", "train_date": (race_date - pd.Timedelta(days=days)).strftime("%Y-%m-%d"), "race_date": race_date.strftime("%Y-%m-%d"), "course": "栗坂", "kind": "slope",
                             "center": "栗", "cond": "良", "rider": "助手", "position": np.nan, "leg_color": leg, "rank": "B", "evaluation": "", "t1": t1, "t2": t1 + 12.5,
                             "t3": 38.5 + rng.normal(0, 0.5), "t4": 52.5 + rng.normal(0, 0.6), "t5": np.nan, "t6": np.nan})
    return tl.add_lap_features(pd.DataFrame(rows))


def _runs(n_horses=60, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for k, d in enumerate(["2022-03-06", "2022-04-03", "2022-05-01", "2022-06-05", "2022-07-03"]):
        for g in range(n_horses // 10):
            fin = rng.permutation(10) + 1
            for j in range(10):
                h = g * 10 + j
                rows.append({"race_id": f"R{k}{g}", "horse_id": f"h{h}", "date": d, "trainer_id": f"t{h % 3}", "finish_position": int(fin[j]), "field_size": 10})
    return pd.DataFrame(rows)


class AsofTest(unittest.TestCase):
    def setUp(self):
        T.MIN_GLOBAL = 1
        self.laps, self.runs = _laps(), _runs()

    def _feats(self, runs):
        ret = pd.DataFrame({"win_ret": np.where(runs["finish_position"] == 1, 5.0, 0.0), "place_ret": np.where(runs["finish_position"] <= 3, 1.5, 0.0)}, index=runs.index)
        return T.build_features(runs, self.laps, ret)

    def test_future_results_do_not_change_past_rows(self):
        a = self._feats(self.runs)
        r2 = self.runs.copy()
        late = pd.to_datetime(r2["date"]) >= pd.Timestamp("2022-06-01")
        r2.loc[late, "finish_position"] = (r2.loc[late, "finish_position"] % 10) + 1
        b = self._feats(r2)
        early = (pd.to_datetime(self.runs["date"]) < pd.Timestamp("2022-06-01")).to_numpy()
        for g in a:
            pd.testing.assert_frame_equal(a[g][early].reset_index(drop=True), b[g][early].reset_index(drop=True), check_exact=False, atol=1e-6)

    def test_same_day_results_not_used(self):
        a = self._feats(self.runs)
        r2 = self.runs.copy()
        day = r2["date"] == "2022-05-01"
        r2.loc[day, "finish_position"] = (r2.loc[day, "finish_position"] % 10) + 1
        b = self._feats(r2)
        m = day.to_numpy()
        for g in a:
            pd.testing.assert_frame_equal(a[g][m].reset_index(drop=True), b[g][m].reset_index(drop=True), check_exact=False, atol=1e-6)

    def test_first_day_has_no_history_and_later_days_have_counts(self):
        f = self._feats(self.runs)["wp_f"]
        first = (self.runs["date"] == "2022-03-06").to_numpy()
        self.assertTrue(f.loc[first, "wf_P2_logn"].isna().all())            # 全体の累積が MIN_GLOBAL 未満
        later = (self.runs["date"] == "2022-07-03").to_numpy()
        self.assertTrue((f.loc[later, "wf_P2_logn"] > 0).all())

    def test_keys_and_shapes(self):
        out = self._feats(self.runs)
        for name, f in out.items():
            self.assertEqual(len(f), len(self.runs))
            self.assertEqual(list(f[["race_id", "horse_id"]].itertuples(index=False, name=None)), list(self.runs[["race_id", "horse_id"]].itertuples(index=False, name=None)))
        self.assertTrue(set(out) >= {"wp_f", "wp_w", "wp_fw", "roi", "horse_pat", "trainer_pat", "diff"})


if __name__ == "__main__":
    unittest.main()


class PayoffListFormTest(unittest.TestCase):
    def test_per_horse_items(self):
        d = json.dumps({"単勝": [{"numbers": "3", "payout": "480"}, {"numbers": "5", "payout": "530"}],
                        "複勝": [{"numbers": "3", "payout": "130"}, {"numbers": "5", "payout": "150"}, {"numbers": "2", "payout": "110"}]})
        p = T.parse_payoff(d, {3: 1, 5: 1, 2: 3})
        self.assertEqual(p["win"], {3: 4.8, 5: 5.3})
        self.assertEqual(p["place"], {3: 1.3, 5: 1.5, 2: 1.1})
