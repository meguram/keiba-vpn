import numpy as np
import pandas as pd

from src.scripts.data import final_week_pattern_eval as P


def _frame():
    rows = []
    for r in range(40):
        for h in range(8):
            rows.append({"race_id": f"r{r}", "year": 2021, "trainer_id": f"t{h % 2}", "trainer_name": f"名{h % 2}", "finish_position": h + 1,
                         "pat": "栗東坂路|A2" if h < 4 else "美浦坂路|X"})
    d = pd.DataFrame(rows)
    d["win"], d["t3"], d["t2"] = (d.finish_position == 1).astype(int), (d.finish_position <= 3).astype(int), (d.finish_position <= 2).astype(int)
    d["fs"] = 8
    for c in ("win_ret", "place_ret", "win_odds", "p_win"):
        d[c] = np.nan
    return d


def test_direction_b_shares_sum_to_one_and_counts():
    b1, b2, b3 = P.direction_b(_frame(), 2020, 2025)
    assert abs(b1["全出走の構成比"].sum() - 1) < 1e-9 and abs(b1["3着以内の構成比"].sum() - 1) < 1e-9
    assert b1.loc["栗東坂路|A2", "3着以内の頭数"] == 40 * 3 and b1.loc["美浦坂路|X", "3着以内の頭数"] == 0
    assert b2["t3_horses"].sum() == 40 * 3


def test_pattern_rel_is_one_when_cells_equal_trainer_baseline():
    d = _frame()
    d["trainer_id"] = "t0"
    d["pat"] = "栗東坂路|A2"
    C = P.cells(d, 2020, 2025)
    r = P.pattern_rel(C)
    assert abs(r.loc["栗東坂路|A2", "rel_top3"] - 1) < 1e-9 and abs(r.loc["栗東坂路|A2", "rel_win"] - 1) < 1e-9


def _two_pattern_frame():
    # 調教師 t0 は A パターンで必ず 1 着、t1 は B パターンで必ず 2 着（8 頭立て、20 レース）
    rows = []
    for r in range(20):
        for h in range(8):
            if h == 0:
                tr, pat, fp = "t0", "栗東坂路|A2", 1
            elif h == 1:
                tr, pat, fp = "t1", "美浦坂路|X", 2
            else:
                tr, pat, fp = "t2", "栗東坂路|A2" if h % 2 else "美浦坂路|X", h + 1
            rows.append({"race_id": f"r{r}", "year": 2021, "trainer_id": tr, "trainer_name": tr, "finish_position": fp, "pat": pat})
    d = pd.DataFrame(rows)
    d["win"], d["t3"], d["t2"] = (d.finish_position == 1).astype(int), (d.finish_position <= 3).astype(int), (d.finish_position <= 2).astype(int)
    d["fs"] = 8
    for c in ("win_ret", "place_ret", "win_odds", "p_win"):
        d[c] = np.nan
    return d


def test_direction_b_hand_computed_two_patterns():
    b1, b2, _ = P.direction_b(_two_pattern_frame(), 2020, 2025)
    # 3 着以内 = 各レース 1 着 t0(A)・2 着 t1(X)・3 着 h=2 の t2 (h%2==0 -> X)
    assert b1.loc["栗東坂路|A2", "3着以内の頭数"] == 20 and b1.loc["美浦坂路|X", "3着以内の頭数"] == 40
    c = b2[(b2.trainer_id == "t0") & (b2.pat == "栗東坂路|A2")].iloc[0]
    assert c["調教師の使用率"] == 1.0 and c["調教師の3着以内に占める割合"] == 1.0 and c["比"] == 1.0


def test_cells_rel_separates_trainer_baseline_from_pattern_effect():
    d = _two_pattern_frame()
    C = P.cells(d, 2020, 2025)
    # t2 は A(h=3,5,7)・X(h=2,4,6) を使い、着順は h+1。A の平均着順は X より下 → A の rel_top3 < X の rel_top3 のはず
    t2 = C[C.trainer_id == "t2"].set_index("pat")
    assert t2.loc["美浦坂路|X", "rel_top3"] > t2.loc["栗東坂路|A2", "rel_top3"]
    assert np.isfinite(C["rel_top3"].dropna()).all()
