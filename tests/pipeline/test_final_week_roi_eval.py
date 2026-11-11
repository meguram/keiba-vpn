import numpy as np
import pandas as pd

from src.scripts.data import final_week_roi_eval as RO


def _frame(years=(2020, 2021, 2023), races=60, persistent=True):
    """栗東坂路|A2 の馬（h<4）は 1 着が単勝 5.0 倍・3 着以内が複勝 1.5 倍。美浦坂路|X（h>=4）は全て外れ。"""
    rng = np.random.default_rng(0)
    rows = []
    for yr in years:
        for r in range(races):
            for h in range(8):
                fp = h + 1 if persistent else int(rng.permutation(8)[h]) + 1
                rows.append({"race_id": f"{yr}-{r}", "year": yr, "finish_position": fp, "pat": "栗東坂路|A2" if h < 4 else "美浦坂路|X", "fs": 8,
                             "win_odds": np.nan, "p_win": np.nan})
    d = pd.DataFrame(rows)
    d["win"], d["t2"], d["t3"] = (d.finish_position == 1).astype(int), (d.finish_position <= 2).astype(int), (d.finish_position <= 3).astype(int)
    d["win_ret"] = np.where(d.win == 1, 5.0, 0.0)
    d["place_ret"] = np.where(d.t3 == 1, 1.5, 0.0)
    return d


def test_roi_table_values_by_hand():
    T = RO.roi_table(_frame(), 2020, 2023)
    a = T.loc["栗東坂路|A2"]
    # 1 レース 4 頭: 1 着 1 頭（5.0）、3 着以内 3 頭（1.5 × 3）→ 単勝 5.0/4、複勝 4.5/4
    assert abs(a["roi_horse_w"] - 1.25) < 1e-9 and abs(a["roi_horse_p"] - 1.125) < 1e-9
    assert abs(a["roi_race_w"] - 1.25) < 1e-9 and abs(a["roi_race_p"] - 1.125) < 1e-9
    assert a["races"] == 180 and a["horses"] == 720
    x = T.loc["美浦坂路|X"]
    assert x["roi_horse_w"] == 0 and x["roi_horse_p"] == 0


def test_roi_table_excludes_missing_returns_and_ci_brackets_point():
    d = _frame()
    d.loc[d["race_id"] == "2020-0", ["win_ret", "place_ret"]] = np.nan
    T = RO.roi_table(d, 2020, 2023, ci=True)
    assert T.loc["栗東坂路|A2", "races"] == 179
    a = T.loc["栗東坂路|A2"]
    assert a["roi_horse_w_lo"] <= a["roi_horse_w"] <= a["roi_horse_w_hi"]


def test_baseline_is_mean_over_all_runners_and_over_pattern_tagged():
    b = RO.baseline(_frame(), 2020, 2023)
    assert abs(b["単勝"] - 5.0 / 8) < 1e-9 and abs(b["複勝"] - 4.5 / 8) < 1e-9 and b["頭数"] == 3 * 60 * 8
    d = _frame()
    d.loc[d["pat"] == "美浦坂路|X", "pat"] = np.nan   # 調教の記録がない馬（全て外れ）を除くと、パターン付きの基準だけが上がる
    b2 = RO.baseline(d, 2020, 2023)
    assert abs(b2["単勝"] - 5.0 / 8) < 1e-9 and abs(b2["単勝(パターン付き)"] - 1.25) < 1e-9 and b2["頭数(パターン付き)"] == 3 * 60 * 4


def _graded(persistent: bool):
    """10 パターン（P0〜P9）。persistent なら P_h の払戻は h が小さいほど高く、全期間で同じ。そうでなければ後の期間で順序が逆になる。"""
    rows = []
    for yr in (2020, 2021, 2023):
        for r in range(40):
            for h in range(10):
                k = h if (persistent or yr < 2023) else 9 - h
                rows.append({"race_id": f"{yr}-{r}", "year": yr, "finish_position": h + 1, "pat": f"栗東坂路|P{h}", "fs": 10, "win_odds": np.nan, "p_win": np.nan,
                             "win_ret": 3.0 - 0.3 * k, "place_ret": 1.5 - 0.15 * k})
    d = pd.DataFrame(rows)
    d["win"], d["t2"], d["t3"] = (d.finish_position == 1).astype(int), (d.finish_position <= 2).astype(int), (d.finish_position <= 3).astype(int)
    return d


def test_persistence_detects_persistent_difference_and_not_when_shuffled():
    r = RO.persistence(_graded(True), (2020, 2021), (2023, 2023), nmin=10)
    top = r[(r["賭け式"] == "複勝") & (r["対象"] == "選択期間で上位20%")].iloc[0]
    bot = r[(r["賭け式"] == "複勝") & (r["対象"] == "選択期間で下位20%")].iloc[0]
    assert top["セル数"] == 3 and bot["セル数"] == 2
    assert abs(top["検証期間の回収率"] - 1.35) < 1e-9 and abs(bot["検証期間の回収率"] - 0.225) < 1e-9 and top["相関(全セル)"] > 0.99
    r2 = RO.persistence(_graded(False), (2020, 2020), (2023, 2023), nmin=10)
    t2 = r2[(r2["賭け式"] == "複勝") & (r2["対象"] == "選択期間で上位20%")].iloc[0]
    assert t2["相関(全セル)"] < 0 and t2["検証期間の回収率"] < t2["選択期間の回収率"]
