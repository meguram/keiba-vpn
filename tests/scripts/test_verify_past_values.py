import copy

from src.scripts.data.verify_past_values import check_race

GOOD = {
    "race_id": "201907030403", "date": "2019-07-07", "venue": "中京", "surface": "芝", "grade": "未勝利", "distance": 1600, "start_time": "11:05", "round": 3, "field_size": 2,
    "entries": [
        {"finish_status": "finished", "finish_position": 1, "bracket_number": 1, "horse_number": 1, "horse_id": "2017102171", "jockey_id": "00732", "trainer_id": "01123",
         "sex_age": "牡2", "jockey_weight": 54.0, "finish_time": "1:36.5", "time_sec": 96.5, "last_3f": 35.5, "odds": 6.2, "popularity": 2, "weight": 424},
        {"finish_status": "finished", "finish_position": 2, "bracket_number": 2, "horse_number": 2, "horse_id": "2017102172", "jockey_id": "00733", "trainer_id": "01124",
         "sex_age": "牝2", "jockey_weight": 54.0, "finish_time": "1:36.7", "time_sec": 96.7, "last_3f": 35.9, "odds": 3.1, "popularity": 1, "weight": 440},
    ],
    "payoff": {"単勝": {"numbers": "1", "payout": "620", "popularity": "2"}},
    "lap_times": [12.0] * 8,
}


def _with(**kw):
    d = copy.deepcopy(GOOD)
    d.update(kw)
    return d


def test_good_race_passes():
    assert check_race(GOOD, "201907030403") == []


def test_detects_value_problems():
    assert "venue" in check_race(_with(venue="不明"), "201907030403")
    assert "distance_range" in check_race(_with(distance=0), "201907030403")
    assert "field_size_vs_entries" in check_race(_with(field_size=5), "201907030403")
    assert "date_year_vs_race_id" in check_race(_with(date="2026-07-07"), "201907030403")
    assert "payoff_win_vs_winner" in check_race(_with(payoff={"単勝": {"numbers": "2", "payout": "620"}}), "201907030403")
    assert "lap_sum_vs_winner_time" in check_race(_with(lap_times=[15.0] * 8), "201907030403")
    d = copy.deepcopy(GOOD)
    d["entries"][0]["time_sec"] = 90.0
    assert "time_vs_time_sec" in check_race(d, "201907030403")
    assert "no_entries" in check_race(_with(entries=[]), "201907030403")


def test_detects_unmapped_grade_and_tolerates_old_format():
    assert "grade_unmapped" in check_race(_with(grade="その他"), "201907030403")
    d = copy.deepcopy(GOOD)
    d["payoff"]["単勝"]["payout"] = "1,410"
    d["entries"].append({"finish_status": "excluded", "bracket_number": 3, "horse_number": 3, "horse_id": "2017102173", "jockey_id": "1", "trainer_id": "2",
                         "sex_age": "牡2", "jockey_weight": 54.0, "odds": 0, "popularity": None})
    d["field_size"] = 3
    assert check_race(d, "201907030403") == []
