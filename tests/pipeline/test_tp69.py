"""tp-69: めぐ指数 v3b の集約層と、excusable（無視してよい低下）の判別のテスト。"""

import unittest

import numpy as np
import pandas as pd

from src.pipeline.inference import megu_excusable_attach as att
from src.pipeline.megu_index import excusable as ex
from src.pipeline.megu_index import v3b


def _runs(n_horses=6, n_runs=9, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    rid = 0
    for h in range(n_horses):
        base = rng.normal(40, 8)
        for k in range(n_runs):
            rid += 1
            rows.append({
                "horse_id": f"h{h}", "race_id": f"r{rid:04d}", "date": pd.Timestamp("2022-01-01") + pd.Timedelta(days=30 * k + h),
                "x": base + rng.normal(0, 8), "fp": int(rng.integers(1, 15)), "field_size": 14,
                "passing_order": "5-5-6-6", "last_3f": 35.0 + rng.normal(0, 1), "margin": "1/2", "distance": 1600, "surface": "芝",
                "weight_change": int(rng.integers(-8, 8)), "jockey_weight": 55.0, "track_condition": "良", "low_conf": 0.0,
            })
    return pd.DataFrame(rows)


class TestV3b(unittest.TestCase):
    def test_floor_by_winner(self):
        x = pd.Series([60.0, 10.0, 30.0, 5.0])
        rid = pd.Series(["a", "a", "a", "b"])
        fp = pd.Series([1, 5, 3, 1])
        out = v3b.floor_by_winner(x, rid, fp, 40.0)
        self.assertEqual(list(out), [60.0, 20.0, 30.0, 5.0])      # 勝ち馬 60 − 40 = 20 が下限。勝ち馬の指数があるレース b は変わらない

    def test_weighted_huber_is_robust_and_weighted(self):
        M = np.array([[50.0, 50.0, 50.0, 50.0, -200.0]])
        w = v3b.run_weights()[None, :]
        self.assertLess(abs(v3b.weighted_huber(M, w, 11.5)[0] - 50.0), 3.0)     # 極端な古い値に引っ張られない
        M2 = np.array([[60.0, 40.0, np.nan, np.nan, np.nan]])
        self.assertTrue(40.0 < v3b.weighted_huber(M2, w, 11.5)[0] < 60.0)
        self.assertTrue(np.isnan(v3b.weighted_huber(np.full((1, 5), np.nan), w, 11.5)[0]))

    def test_aggregate_weights(self):
        vals = [40.0, 40.0, 40.0, 40.0, 10.0]
        a = v3b.aggregate(vals)
        self.assertAlmostEqual(a, 40.0, delta=3.0)
        # 低信頼の走は重みが下がる / excusable の重みを 0 にするとその走は無視される
        v = [60.0, 20.0]
        full = v3b.aggregate(v)
        lc = v3b.aggregate(v, low_conf=[False, True])
        self.assertGreater(lc, full)
        ex0 = v3b.aggregate(v, excusable=[False, True], excusable_weight=0.0)
        self.assertAlmostEqual(ex0, 60.0, places=6)
        self.assertEqual(v3b.aggregate([]), None)
        self.assertEqual(v3b.aggregate([None, None]), None)

    def test_defaults_do_not_exclude(self):
        self.assertEqual(v3b.EXCUSABLE_WEIGHT, 1.0)       # tp-69: 除外すると次走予測が悪化したので既定は除かない


class TestParsers(unittest.TestCase):
    def test_margin_and_passing(self):
        self.assertEqual(ex.parse_margin_len("1/2"), 0.5)
        self.assertEqual(ex.parse_margin_len("1.1/4"), 1.25)
        self.assertEqual(ex.parse_margin_len("クビ"), 0.3)
        self.assertEqual(ex.parse_margin_len(""), 0.0)
        self.assertEqual(ex.parse_margin_len("3"), 3.0)
        self.assertTrue(np.isnan(ex.parse_margin_len("???")))
        self.assertEqual(ex.parse_passing("3-3-2-2"), (3.0, 2.0))
        self.assertTrue(np.isnan(ex.parse_passing(None)[0]))


class TestSignals(unittest.TestCase):
    def test_no_lookahead(self):
        d = _runs()
        full = ex.run_signals(d)
        cut = pd.Timestamp("2022-06-01")
        part = ex.run_signals(d[d["date"] < cut])
        cols = [c for c in ex.SIGNALS if c != "pace_dis"]
        a = full.loc[part.index, cols].to_numpy(float)
        b = part[cols].to_numpy(float)
        self.assertTrue(np.array_equal(np.nan_to_num(a, nan=-9e9), np.nan_to_num(b, nan=-9e9)))      # 切断日以降を消しても、それより前の信号は完全一致
        # 将来の指数を書き換えても、その前の走の信号は変わらない
        d2 = d.copy()
        d2.loc[d2["date"] >= cut, "x"] = 999.0
        alt = ex.run_signals(d2)
        before = d["date"] < cut
        self.assertTrue(np.array_equal(np.nan_to_num(full.loc[before, "own_drop"].to_numpy(float), nan=-9e9),
                                       np.nan_to_num(alt.loc[before, "own_drop"].to_numpy(float), nan=-9e9)))

    def test_rebound_label_uses_next_runs(self):
        d = _runs(n_horses=1, n_runs=8, seed=3).reset_index(drop=True)
        d["x"] = [50, 50, 50, 50, 50, 20, 50, 50.0]
        sig = ex.run_signals(d)
        d["prior_med"] = sig["prior_med"]
        lab = ex.rebound_label(d)
        self.assertEqual(lab.iloc[5], 1.0)          # 20 のあと 50, 50 に戻った
        self.assertTrue(np.isnan(lab.iloc[7]))      # 次の走が無い
        self.assertAlmostEqual(sig["own_drop"].iloc[5], (50 - 20) / 11.5, places=6)


class TestModel(unittest.TestCase):
    def test_fit_score_roundtrip_and_reasons(self):
        rng = np.random.default_rng(1)
        n = 600
        sig = pd.DataFrame({c: rng.normal(0, 1, n) for c in ex.SIGNALS})
        y = pd.Series((sig["l3f_vs_fp"] + 0.3 * rng.normal(0, 1, n) > 0).astype(float))
        m = ex.fit_model(sig, y, pd.Series(True, index=sig.index))
        p = m.score(sig)
        self.assertTrue(((p > 0) & (p < 1)).all())
        self.assertGreater(p[y == 1].mean(), p[y == 0].mean())
        m2 = ex.ExcusableModel.from_json(m.to_json())
        self.assertTrue(np.allclose(m.score(sig), m2.score(sig)))
        row = sig.iloc[0].copy()
        row["l3f_vs_fp"] = 5.0
        self.assertIn(ex.REASON_JA["l3f_vs_fp"], m.reasons(row))

    def test_missing_signals_are_neutral(self):
        m = ex.load_default_model()
        if m is None:
            self.skipTest("config/megu_excusable.json が無い")
        p = m.score(pd.DataFrame({c: [np.nan] for c in ex.SIGNALS}))
        self.assertTrue(0 < float(p.iloc[0]) < 1)


class TestAttach(unittest.TestCase):
    def test_annotate(self):
        m = ex.load_default_model()
        runs = []
        recent = []
        vals = [50, 50, 52, 49, 51, 22]
        for k, v in enumerate(vals):
            rid = f"2025010{k + 1}01"
            ymd = f"2025{k + 1:02d}15"
            runs.append((ymd, float(v), False, "芝", 1600.0, "東京", 3, 55.0, 4, rid))
        for k in range(5):
            idx = len(vals) - 1 - k
            recent.append({"date": "x", "value": float(vals[idx]), "low": False, "race_id": runs[idx][9], "finish": 12 if idx == 5 else 3, "field": 16,
                           "passing": "10-10-9-9" if idx == 5 else "4-4-4-4", "l3f": 33.5 if idx == 5 else 35.0, "margin": "1.1/2" if idx == 5 else "ハナ",
                           "going": "良", "bw_diff": 2, "jw": 55.0, "surface": "芝", "distance": 1600.0})
        exd = {"recent": recent}
        att.annotate(exd, runs, m)
        self.assertIn("v3b", exd)
        self.assertTrue(40 < exd["v3b"]["value"] < 52)
        latest = exd["recent"][0]
        if m is not None:
            self.assertTrue(latest.get("low_run"))
            self.assertIn("excusable_p", latest)
            self.assertNotIn("low_run", exd["recent"][2] if exd["recent"][2].get("low_run") is None else {})

    def test_annotate_without_data_is_noop(self):
        att.annotate(None, [])
        e = {"recent": []}
        att.annotate(e, [])
        self.assertNotIn("v3b", e)


if __name__ == "__main__":
    unittest.main()
