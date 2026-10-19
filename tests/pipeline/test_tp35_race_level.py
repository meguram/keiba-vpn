"""tp-35 のレース単位集計（母数を『該当馬の頭数』から『該当馬がいるレース』に揃える）の計算を小さな表で確かめる。"""
import numpy as np
import pandas as pd

from src.scripts.data import tp35_race_level as R


def _frame() -> pd.DataFrame:
    rows = [
        # race, fin, win_ret, place_ret, odds, p_win
        ("R1", 1, 3.0, 1.2, 3.0, 0.2), ("R1", 2, 0.0, 1.1, 5.0, 0.5), ("R1", 3, 0.0, 1.3, 8.0, 0.2), ("R1", 4, 0.0, 0.0, 9.0, 0.1),
        ("R2", 1, 5.0, 2.0, 5.0, 0.4), ("R2", 2, 0.0, 1.5, 4.0, 0.3), ("R2", 3, 0.0, 1.2, 6.0, 0.2), ("R2", 4, 0.0, 0.0, 7.0, 0.1),
    ]
    d = pd.DataFrame(rows, columns=["race_id", "finish_position", "win_ret", "place_ret", "win_odds", "p_win"])
    d["year"] = 2022
    d["horse_id"] = [f"h{i}" for i in range(len(d))]
    d["win_odds"] = d["win_odds"].astype(float)
    return R.prep(d.assign(year=2022))


def test_race_level_metrics() -> None:
    d = _frame()
    key = np.array(["c", "c", None, None, None, None, None, "c"], dtype=object)   # R1 は 1・2 着の 2 頭、R2 は 4 着の 1 頭
    S = R.summarize(R.race_table(d, key)).loc["c"]
    assert S["races"] == 2 and S["horses"] == 3
    assert np.isclose(S["horse_win"], 1 / 3) and np.isclose(S["race_win"], 0.5)
    assert np.isclose(S["exp_win"], (2 / 4 + 1 / 4) / 2)
    assert np.isclose(S["exp_top3"], (1.0 + 0.75) / 2)                      # R1: 4 頭中 2 頭が該当 → 3 着以内は必ず入る
    assert np.isclose(S["roi_horse_w"], 3.0 / 3) and np.isclose(S["roi_race_w"], (3.0 / 2 + 0.0) / 2)
    assert np.isclose(S["roi_fav_w"], (3.0 + 0.0) / 2)                      # R1 は該当 2 頭のうちオッズ最小の 1 着馬、R2 は該当 1 頭
    assert np.isclose(S["roi_mod_w"], (0.0 + 0.0) / 2)                      # R1 は p_win 最大の 2 着馬
    assert np.isclose(S["roi_race_p"], ((1.2 + 1.1) / 2 + 0.0) / 2)


def test_reality_check_pure_noise_is_not_significant() -> None:
    rng = np.random.default_rng(0)
    NUM = rng.normal(0, 1, (20, 2000))
    DEN = np.ones((20, 2000))
    r = R.reality_check(NUM, DEN, min_den=100, nb=200, seed=1)
    assert r["p"] > 0.05
