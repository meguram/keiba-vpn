"""重賞のルールベース予想家: 縮小・点・印・評価の指標。"""

import numpy as np
import pandas as pd

from src.research.graded import model as M
from src.research.graded import tally as TL


def _row(level="前走G2で3着内", main=(0, 0, 0, 0, 0.0, 0), sub=(0, 0, 0, 0, 0.0, 0), glob=(0, 0, 0, 0, 0.0, 0), all_main_n=100, all_sub_n=1000):
    r = {"level": level, "type": "B02", "row": 0, "all_main_n": all_main_n, "all_sub_n": all_sub_n}
    for p, v in (("main", main), ("sub", sub), ("sub4", sub), ("glob", glob)):
        n, w1, w2, w3, e3, ed = v
        r.update({f"{p}_n": n, f"{p}_w1": w1, f"{p}_w2": w2, f"{p}_w3": w3, f"{p}_e3": e3, f"{p}_ed": ed, f"{p}_edhit": ed, f"{p}_e1": e3 / 3})
    return r


def test_shrink_no_data_is_neutral():
    S = M.shrink(pd.DataFrame([_row()]), 10, 10, 10)
    assert np.isclose(S["x"].iloc[0], 0.0) and S["layer"].iloc[0] == "全体"


def test_shrink_hierarchy_and_thresholds():
    # 全体で 2 倍、副は無し、主は件数不足 → 全体の値（を 1 へ縮めたもの）
    T = pd.DataFrame([_row(glob=(1000, 200, 200, 200, 300.0, 300), main=(10, 3, 3, 3, 3.0, 3))])
    S = M.shrink(T, 10, 10, 10)
    assert S["layer"].iloc[0] == "全体"
    assert np.isclose(S["r_m"].iloc[0], (600 + 10) / (300 + 10))
    # 主の件数が足りると主が使われ、主が凡走なら全体の上振れを打ち消す
    T = pd.DataFrame([_row(glob=(1000, 200, 200, 200, 300.0, 300), main=(40, 0, 0, 1, 12.0, 8))])
    S = M.shrink(T, 10, 10, 10)
    assert S["layer"].iloc[0] == "主" and S["x"].iloc[0] < 0


def test_unknown_level_and_high_share_score_zero():
    S = M.shrink(pd.DataFrame([_row(level="不明", glob=(1000, 300, 300, 300, 300.0, 300))]), 10, 10, 10)
    assert S["x"].iloc[0] == 0.0
    S = M.shrink(pd.DataFrame([_row(main=(90, 20, 20, 20, 27.0, 9), all_main_n=100)]), 10, 10, 10)   # 主の 90% が当てはまる
    assert S["x"].iloc[0] == 0.0


def test_marks_order_and_tiebreak():
    sc = pd.DataFrame({"row": range(6), "race_id": "R", "score": [0.5, 0.9, 0.5, 0.1, -0.2, 0.5], "n_good_main": [1, 0, 3, 0, 0, 1],
                       "hn": [1, 2, 3, 4, 5, 6], "y3": 0, "y1": 0, "year": 2025})
    mk = M.marks(sc)
    assert mk.sort_values("rank")["hn"].tolist() == [2, 3, 1, 6, 4, 5]
    assert mk.sort_values("rank")["mark"].tolist() == ["◎", "○", "▲", "△", "△", ""]


def test_race_metrics_payouts():
    from src.research.graded.run import race_metrics
    mk = pd.DataFrame({"race_id": "R", "hn": [3, 5, 7, 1], "rank": [1, 2, 3, 4], "y1": [1, 0, 0, 0], "y3": [1, 1, 0, 1]})
    pay = {("R", "tansho"): {"3": 450}, ("R", "fukusho"): {"3": 160, "5": 200, "1": 300}, ("R", "umaren"): {"3-5": 1200}, ("R", "wide"): {"3-5": 400, "1-3": 900, "1-5": 800}}
    rt = race_metrics(mk, pay).iloc[0]
    assert rt["ret_tansho"] == 450 and rt["ret_fukusho"] == 160 and rt["ret_umaren"] == 1200 and rt["ret_wide"] == 400
    assert rt["cost_umaren"] == 200 and rt["top3_in3"] == 2


def test_type_z_positive_when_x_predicts():
    rng = np.random.default_rng(0)
    rows = []
    for r in range(200):
        x = rng.normal(size=12)
        y = (x + rng.normal(scale=0.5, size=12)).argsort().argsort() >= 9
        rows += [{"type": "A", "race_id": r, "x": a, "y3": float(b)} for a, b in zip(x, y)]
    z = M.type_z(pd.DataFrame(rows))
    assert z.loc["A", "z"] > 5
    assert TL.STATS[0] == "n"
