"""megu_predict_race ヘルパーのユニットテスト。"""

import unittest

from src.api.megu_predict_race import is_actual_megu_displayed, is_race_finisher


class TestMeguPredictRaceHelpers(unittest.TestCase):
    def test_finisher_requires_pos_and_time(self):
        self.assertTrue(is_race_finisher(1, 98.5))
        self.assertFalse(is_race_finisher(-1, 98.5))
        self.assertFalse(is_race_finisher(1, None))
        self.assertFalse(is_race_finisher(0, 98.5))

    def test_actual_displayed_includes_out_of_range(self):
        self.assertTrue(is_actual_megu_displayed(101.2, "valid"))
        self.assertTrue(is_actual_megu_displayed(None, "out_of_range"))
        self.assertFalse(is_actual_megu_displayed(None, None))
        self.assertFalse(is_actual_megu_displayed(None, "invalid"))


if __name__ == "__main__":
    unittest.main()
