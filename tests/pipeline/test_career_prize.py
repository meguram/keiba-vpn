"""
career_prize モジュールのユニットテスト
"""
import unittest
import pandas as pd
from src.pipeline.megu_index.career_prize import (
    classify_grade,
    estimate_prize,
    compute_level_feature,
    build_career_prizes_from_flat,
    PRIZE_TABLE,
)


class TestClassifyGrade(unittest.TestCase):
    def test_direct_grade(self):
        self.assertEqual(classify_grade("G1"), "G1")
        self.assertEqual(classify_grade("未勝利"), "未勝利")
        self.assertEqual(classify_grade("1勝"), "1勝")
        self.assertEqual(classify_grade("OP"), "OP")

    def test_race_name_fallback(self):
        self.assertEqual(classify_grade(None, "皐月賞(G1)"), "G1")
        self.assertEqual(classify_grade("", "三条S(3勝クラス)"), "3勝")
        self.assertEqual(classify_grade(None, "3歳未勝利"), "未勝利")
        self.assertEqual(classify_grade(None, "浄土平特別(1勝)"), "1勝")

    def test_old_style_names(self):
        self.assertEqual(classify_grade("500万下"), "1勝")
        self.assertEqual(classify_grade("1000万下"), "2勝")
        self.assertEqual(classify_grade("1600万下"), "3勝")

    def test_unknown_fallback(self):
        # 特別競走など不明な場合は OP
        self.assertEqual(classify_grade("", "セントポーリア賞"), "OP")


class TestEstimatePrize(unittest.TestCase):
    def test_g1_winner(self):
        self.assertEqual(estimate_prize("G1", 1), 20000.0)

    def test_g1_2nd(self):
        self.assertEqual(estimate_prize("G1", 2), 8000.0)

    def test_below_rank(self):
        self.assertEqual(estimate_prize("未勝利", 6), 0.0)
        self.assertEqual(estimate_prize("1勝", 10), 0.0)

    def test_none_position(self):
        self.assertEqual(estimate_prize("2勝", None), 0.0)

    def test_prize_table_coverage(self):
        for grade in PRIZE_TABLE:
            for pos in [1, 2, 3, 4, 5]:
                prize = estimate_prize(grade, pos)
                self.assertGreaterEqual(prize, 0.0)


class TestComputeLevelFeature(unittest.TestCase):
    def test_zero_career(self):
        # 0万円なら必ず負
        lf = compute_level_feature(0, "未勝利")
        self.assertLess(lf, 0)

    def test_high_career_vs_low_grade(self):
        # G1馬が未勝利に出走 → 大きな正の値
        lf = compute_level_feature(50000, "未勝利")
        self.assertGreater(lf, 0)

    def test_equal_level(self):
        # 未勝利1着賞金(500万)のキャリアで未勝利レース → ほぼ0
        from src.pipeline.megu_index.career_prize import CAREER_PRIZE_REFERENCE
        ref = CAREER_PRIZE_REFERENCE["未勝利"]
        lf = compute_level_feature(ref, "未勝利")
        self.assertAlmostEqual(lf, 0.0, places=5)


class TestBuildCareerPrizes(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame([
            {"horse_id": "H001", "race_id": "R1", "date": "2024-01-15",
             "finish_position": 1, "grade": "未勝利", "race_name": "未勝利", "race_class": ""},
            {"horse_id": "H001", "race_id": "R2", "date": "2024-03-20",
             "finish_position": 1, "grade": "1勝", "race_name": "1勝クラス", "race_class": ""},
            {"horse_id": "H001", "race_id": "R3", "date": "2024-06-01",
             "finish_position": 2, "grade": "2勝", "race_name": "2勝クラス", "race_class": ""},
        ])

    def test_total_prize(self):
        result = build_career_prizes_from_flat(self.df)
        h001 = result[result["horse_id"] == "H001"].iloc[0]
        # 500 (未勝利1着) + 700 (1勝1着) + 440 (2勝2着) = 1640
        self.assertAlmostEqual(h001["career_prize_est"], 1640.0, places=1)

    def test_as_of_dates(self):
        # 2024-03-01時点 → 未勝利1着のみ
        as_of = {"H001": "20240301"}
        result = build_career_prizes_from_flat(self.df, as_of_dates=as_of)
        h001 = result[result["horse_id"] == "H001"].iloc[0]
        self.assertAlmostEqual(h001["career_prize_est"], 500.0, places=1)

    def test_before_first_race(self):
        # 最初のレースより前 → 0万円
        as_of = {"H001": "20230101"}
        result = build_career_prizes_from_flat(self.df, as_of_dates=as_of)
        h001 = result[result["horse_id"] == "H001"].iloc[0]
        self.assertAlmostEqual(h001["career_prize_est"], 0.0, places=1)


if __name__ == "__main__":
    unittest.main()
