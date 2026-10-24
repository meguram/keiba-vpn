"""消印（10 人気以下で複勝確率が低い馬。複数頭可）のテスト。"""
from __future__ import annotations

import unittest

from src.pipeline.inference import prediction_results as pr
from src.utils.race_probabilities import assign_kesi_mark, assign_mece_marks, kesi_config

KESI_MIN_POP = int(kesi_config()["min_pop"])
KESI_P3_MAX = float(kesi_config()["p3_max"])


def _race(n: int = 16, with_pop: bool = True, low_from: int = 11) -> list[dict]:
    """人気順（1 番人気が最も勝率が高い）の n 頭立て。low_from 番人気以降は複勝確率を KESI_P3_MAX 以下にしてある。"""
    out = []
    for i in range(n):
        win = 0.30 * (0.80 ** i)
        top3 = min(0.95, win * 2.6 + 0.05)
        if i + 1 >= low_from:
            top3 = 0.03
        e = {"horse_number": i + 1, "win_prob": win, "top2_prob": min(0.95, win * 1.8 + 0.03), "top3_prob": top3,
             "win_odds": 2.0 + i * 3.0}
        if with_pop:
            e["popularity"] = i + 1
        out.append(e)
    return out


class TestKesiMark(unittest.TestCase):
    def test_marks_all_low_probability_horses_from_10th_popularity(self):
        es = _race()
        assign_mece_marks(es)
        self.assertIsNone(assign_kesi_mark(es))
        kesi = sorted(e["horse_number"] for e in es if e["mark_type"] == "kesi")
        self.assertEqual(kesi, list(range(11, 17)))             # 複数頭に付く
        self.assertIn("11番人気", next(e for e in es if e["horse_number"] == 11)["kesi_reason"])

    def test_no_kesi_for_top_popularity_even_with_low_probability(self):
        es = _race(low_from=1)                                  # 全頭の複勝確率が低くても、9 番人気以上には付けない
        assign_mece_marks(es)
        assign_kesi_mark(es)
        for e in es:
            if e["horse_number"] < KESI_MIN_POP:
                self.assertNotEqual(e["mark_type"], "kesi")
        self.assertTrue(any(e["mark_type"] == "kesi" for e in es))

    def test_probability_above_threshold_gets_no_kesi(self):
        es = _race()
        for e in es:
            if e["horse_number"] >= KESI_MIN_POP:
                e["top3_prob"] = KESI_P3_MAX + 0.01
        assign_mece_marks(es)
        assign_kesi_mark(es)
        self.assertFalse(any(e["mark_type"] == "kesi" for e in es))   # 0 頭のレースもある

    def test_never_overrides_main_marks(self):
        es = _race()
        assign_mece_marks(es)
        main = {e["horse_number"]: e["mark_type"] for e in es}
        assign_kesi_mark(es)
        for e in es:
            if main[e["horse_number"]] in ("honmei", "pair", "show_val", "star"):
                self.assertEqual(e["mark_type"], main[e["horse_number"]])

    def test_main_marked_horse_is_skipped_but_others_marked(self):
        es = _race()
        for e in es:
            e["mark_type"] = "none"
        es[12]["mark_type"] = "star"                            # 13 番人気に★
        assign_kesi_mark(es)
        self.assertEqual(es[12]["mark_type"], "star")
        self.assertEqual(es[11]["mark_type"], "kesi")

    def test_overrides_maybe_mark(self):
        es = _race()
        for e in es:
            e["mark_type"] = "none"
        es[10]["mark_type"] = "anchor"
        assign_kesi_mark(es)
        self.assertEqual(es[10]["mark_type"], "kesi")

    def test_small_field_gets_no_kesi(self):
        es = _race(n=7, low_from=3)
        assign_mece_marks(es)
        reason = assign_kesi_mark(es)
        self.assertIsNotNone(reason)
        self.assertFalse(any(e["mark_type"] == "kesi" for e in es))

    def test_field_under_10_has_nobody_at_10th_popularity(self):
        es = _race(n=9, low_from=3)
        assign_mece_marks(es)
        self.assertIsNone(assign_kesi_mark(es))
        self.assertFalse(any(e["mark_type"] == "kesi" for e in es))

    def test_missing_odds_gets_no_kesi(self):
        es = _race()
        es[4]["win_odds"] = None
        assign_mece_marks(es)
        reason = assign_kesi_mark(es)
        self.assertIsNotNone(reason)
        self.assertFalse(any(e["mark_type"] == "kesi" for e in es))

    def test_missing_top3_probability_is_not_marked(self):
        es = _race()
        es[13]["top3_prob"] = None
        assign_mece_marks(es)
        assign_kesi_mark(es)
        self.assertNotEqual(es[13]["mark_type"], "kesi")

    def test_popularity_falls_back_to_odds_order(self):
        with_pop = _race(with_pop=True)
        without = _race(with_pop=False)
        for es in (with_pop, without):
            assign_mece_marks(es)
            assign_kesi_mark(es)
        self.assertEqual([e["mark_type"] for e in with_pop], [e["mark_type"] for e in without])

    def test_config_file_drives_threshold(self):
        import json
        import os
        import tempfile

        es = _race()
        for e in es:
            if e["horse_number"] >= KESI_MIN_POP:
                e["top3_prob"] = 0.10                     # 現行のしきい値（8%）では付かない
        assign_mece_marks(es)
        assign_kesi_mark(es)
        self.assertFalse(any(e["mark_type"] == "kesi" for e in es))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "kesi.json")
            json.dump({"version": 99, "p3_max": 0.12}, open(p, "w"))
            old = os.environ.get("KEIBA_KESI_CONFIG")
            os.environ["KEIBA_KESI_CONFIG"] = p
            try:
                es2 = [dict(e, mark_type="none") for e in es]
                assign_kesi_mark(es2)
                self.assertTrue(any(e["mark_type"] == "kesi" for e in es2))
                self.assertEqual(kesi_config()["version"], 99)
            finally:
                if old is None:
                    os.environ.pop("KEIBA_KESI_CONFIG", None)
                else:
                    os.environ["KEIBA_KESI_CONFIG"] = old
        self.assertEqual(kesi_config()["version"], 2)   # 環境変数を戻せば config/kesi_mark.json に戻る

    def test_missing_config_falls_back_to_defaults(self):
        import os

        old = os.environ.get("KEIBA_KESI_CONFIG")
        os.environ["KEIBA_KESI_CONFIG"] = "/nonexistent/kesi.json"
        try:
            self.assertEqual(kesi_config()["version"], 0)
            self.assertEqual(kesi_config()["p3_max"], 0.08)
        finally:
            if old is None:
                os.environ.pop("KEIBA_KESI_CONFIG", None)
            else:
                os.environ["KEIBA_KESI_CONFIG"] = old

    def test_idempotent(self):
        es = _race()
        assign_mece_marks(es)
        assign_kesi_mark(es)
        first = [e["mark_type"] for e in es]
        assign_kesi_mark(es)
        self.assertEqual(first, [e["mark_type"] for e in es])

    def test_prediction_results_tolerates_old_summaries_without_kesi(self):
        old = {"races": 1, "races_settled": 1, "races_no_prediction": 0, "races_no_result": 0,
               "top1": {"n": 1, "f1": 1, "f2": 0, "f3": 0, "f4p": 0},
               "marks": {m: {"n": 1, "f1": 1, "f2": 0, "f3": 0, "f4p": 0} for m in ("honmei", "pair", "star")},
               "bets": {"invested": 0, "returned": 0, "races_bought": 0, "races_hit": 0, "estimated_races": 0, "by_type": {}},
               "logloss": {"n": 0, "sum": 0.0}}
        merged = pr.merge_summaries([old, old])
        self.assertEqual(merged["marks"]["honmei"]["n"], 2)
        self.assertEqual(merged["marks"]["kesi"]["n"], 0)

    def test_summarize_counts_kesi_finish(self):
        rec = {"included": True, "top1": {"finish": 1},
               "marks": {"honmei": [{"finish": 1}], "pair": [], "star": [], "kesi": [{"finish": 7}, {"finish": 2}]},
               "bets": {}, "winner_logloss": None}
        s = pr.summarize([rec])
        self.assertEqual(s["marks"]["kesi"]["f4p"], 1)
        self.assertEqual(s["marks"]["kesi"]["n"], 2)


if __name__ == "__main__":
    unittest.main()
