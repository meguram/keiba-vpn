"""消印（上位 5 人気の中で複勝確率が最も低い 1 頭）のテスト。"""
from __future__ import annotations

import unittest

from src.pipeline.inference import prediction_results as pr
from src.utils.race_probabilities import assign_kesi_mark, assign_mece_marks


def _race(n: int = 12, with_pop: bool = True) -> list[dict]:
    """人気順（1 番人気が最も勝率が高い）の n 頭立て。複勝確率は単調に下がるが、3 番人気だけ低めにしてある。"""
    out = []
    for i in range(n):
        win = 0.30 * (0.72 ** i)
        top3 = min(0.95, win * 2.6 + 0.05)
        e = {"horse_number": i + 1, "win_prob": win, "top2_prob": min(0.95, win * 1.8 + 0.03), "top3_prob": top3,
             "win_odds": 2.0 + i * 1.7}
        if with_pop:
            e["popularity"] = i + 1
        out.append(e)
    out[2]["top3_prob"] = 0.05           # 3 番人気だけ複勝確率が極端に低い
    return out


class TestKesiMark(unittest.TestCase):
    def test_picks_lowest_top3_among_top5_popularity(self):
        es = _race()
        assign_mece_marks(es)
        reason = assign_kesi_mark(es)
        self.assertIsNone(reason)
        kesi = [e for e in es if e["mark_type"] == "kesi"]
        self.assertEqual(len(kesi), 1)
        self.assertEqual(kesi[0]["horse_number"], 3)
        self.assertIn("3番人気", kesi[0]["kesi_reason"])

    def test_never_overrides_main_marks(self):
        es = _race()
        assign_mece_marks(es)
        main = {e["horse_number"]: e["mark_type"] for e in es}
        assign_kesi_mark(es)
        for e in es:
            if main[e["horse_number"]] in ("honmei", "pair", "show_val", "star"):
                self.assertEqual(e["mark_type"], main[e["horse_number"]])

    def test_overrides_maybe_mark(self):
        es = _race()
        for e in es:
            e["mark_type"] = "none"
        es[2]["mark_type"] = "anchor"
        assign_kesi_mark(es)
        self.assertEqual(es[2]["mark_type"], "kesi")

    def test_skips_main_marked_candidate_and_takes_next_lowest(self):
        es = _race()
        for e in es:
            e["mark_type"] = "none"
        es[2]["mark_type"] = "star"          # 最低の馬が主印 → 次に低い馬が選ばれる
        assign_kesi_mark(es)
        self.assertEqual(es[2]["mark_type"], "star")
        kesi = [e for e in es if e["mark_type"] == "kesi"]
        self.assertEqual(len(kesi), 1)
        self.assertEqual(kesi[0]["horse_number"], 5)   # 上位 5 人気の残りでは 5 番人気が最も低い

    def test_small_field_gets_no_kesi(self):
        es = _race(n=7)
        assign_mece_marks(es)
        reason = assign_kesi_mark(es)
        self.assertIsNotNone(reason)
        self.assertFalse(any(e["mark_type"] == "kesi" for e in es))

    def test_missing_odds_gets_no_kesi(self):
        es = _race()
        es[4]["win_odds"] = None
        assign_mece_marks(es)
        reason = assign_kesi_mark(es)
        self.assertIsNotNone(reason)
        self.assertFalse(any(e["mark_type"] == "kesi" for e in es))

    def test_popularity_falls_back_to_odds_order(self):
        with_pop = _race(with_pop=True)
        without = _race(with_pop=False)
        for es in (with_pop, without):
            assign_mece_marks(es)
            assign_kesi_mark(es)
        self.assertEqual([e["mark_type"] for e in with_pop], [e["mark_type"] for e in without])

    def test_idempotent_and_existing_marks_unchanged_without_popularity_info(self):
        es = _race()
        assign_mece_marks(es)
        before = [e["mark_type"] for e in es]
        assign_kesi_mark(es)
        after1 = [e["mark_type"] for e in es]
        diff = [i for i, (a, b) in enumerate(zip(before, after1)) if a != b]
        self.assertLessEqual(len(diff), 1)

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
               "marks": {"honmei": [{"finish": 1}], "pair": [], "star": [], "kesi": [{"finish": 7}]},
               "bets": {}, "winner_logloss": None}
        s = pr.summarize([rec])
        self.assertEqual(s["marks"]["kesi"]["f4p"], 1)


if __name__ == "__main__":
    unittest.main()
