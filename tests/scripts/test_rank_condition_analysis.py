"""rank_condition_analysis の小さな部品（分解の恒等式・区切りの凍結・買い目と的中・市場の並び・Holm・縮小）。"""

import numpy as np
import pandas as pd
import pytest

from src.scripts.data import rank_condition_analysis as rc


@pytest.mark.parametrize("h1,a1,h0,a0,c", [(0.3, 280, 0.33, 270, 100), (0.0, 0.0, 0.25, 700, 200), (0.5, 400, 0.45, 390, 200)])
def test_decompose_sums_to_roi_diff(h1, a1, h0, a0, c):
    hp, ap = rc.decompose(h1, a1, h0, a0, c)
    assert hp + ap == pytest.approx((h1 * a1 - h0 * a0) / c)


def test_decompose_signs():
    hp, ap = rc.decompose(0.30, 300, 0.33, 300, 100)        # 払戻は同じで的中率だけ低い → 的中率の分だけが負
    assert hp < 0 and ap == pytest.approx(0)
    hp, ap = rc.decompose(0.33, 250, 0.33, 300, 100)        # 的中率は同じで払戻だけ小さい → 払戻の分だけが負
    assert ap < 0 and hp == pytest.approx(0)


def test_cuts_frozen_from_valid_apply_to_other_period():
    valid = np.linspace(0, 1, 101)
    cuts = rc.quantile_cuts(valid)
    assert cuts == pytest.approx([0.2, 0.4, 0.6, 0.8])
    lab = rc.apply_cuts(np.array([-5.0, 0.1, 0.2, 0.79, 0.8, 5.0]), cuts)    # 範囲の外の値も、valid の区切りの両端の帯に入る
    assert [x[:2] for x in lab] == ["Q1", "Q1", "Q2", "Q4", "Q5", "Q5"]
    assert lab[0].startswith("Q1(<") and lab[-1].startswith("Q5(≥")


def _pay():
    return {("r1", "tansho"): {"3": 250}, ("r1", "fukusho"): {"3": 120, "1": 110, "2": 150},
            ("r1", "umaren"): {"1-3": 900}, ("r1", "wide"): {"1-3": 300, "2-3": 400, "1-2": 200},
            ("r2", "tansho"): {"5": 1000}, ("r2", "fukusho"): {"5": 300}, ("r2", "umaren"): {"4-5": 2000}, ("r2", "wide"): {"4-5": 600}}


def _runners():
    r1 = pd.DataFrame({"race_id": "r1", "horse_id": ["h1", "h2", "h3", "h4"], "horse_number": [1, 2, 3, 4],
                       "odds": [3.0, 5.0, 2.0, 20.0], "popularity": [2, 3, 1, 4]})
    r2 = pd.DataFrame({"race_id": "r2", "horse_id": ["k1", "k2", "k3", "k4", "k5"], "horse_number": [1, 2, 3, 4, 5],
                       "odds": [1.5, 4.0, 8.0, 12.0, 30.0], "popularity": [1, 2, 3, 4, 5]})
    return pd.concat([r1, r2], ignore_index=True)


def test_race_bets_hits_and_wide_double_hit():
    pred = pd.DataFrame({"race_id": ["r1"] * 4 + ["r2"] * 5, "horse_id": ["h1", "h2", "h3", "h4", "k1", "k2", "k3", "k4", "k5"],
                         "p_win": [0.2, 0.2, 0.5, 0.1, 0.4, 0.3, 0.1, 0.1, 0.1]})
    t = rc.race_bets(pred, _runners(), _pay()).set_index("race_id")
    # r1: 3 → 1・2（同じ確率は馬番の小さい順）。単勝 250、複勝 120、馬連 1-3 の 900、ワイドは 1-3 と 2-3 の両方で 700
    assert t.loc["r1", "ret_tansho"] == 250 and t.loc["r1", "ret_fukusho"] == 120
    assert t.loc["r1", "ret_umaren"] == 900 and t.loc["r1", "ret_wide"] == 700 and t.loc["r1", "hit_wide"] == 1.0
    assert t.loc["r1", "top1_pop"] == 1 and t.loc["r1", "p1"] == pytest.approx(0.5)
    # r2: 1 → 2・3。どれも外れ
    assert t.loc["r2", ["ret_tansho", "ret_fukusho", "ret_umaren", "ret_wide"]].sum() == 0
    assert t.loc["r2", "hit_tansho"] == 0.0


def test_race_bets_skips_races_without_all_payoffs():
    pay = _pay()
    del pay[("r2", "wide")]
    pred = pd.DataFrame({"race_id": ["r1"] * 4 + ["r2"] * 5, "horse_id": ["h1", "h2", "h3", "h4", "k1", "k2", "k3", "k4", "k5"], "p_win": 0.1})
    assert list(rc.race_bets(pred, _runners(), pay)["race_id"]) == ["r1"]


def test_market_pred_orders_by_popularity():
    m = rc.market_pred(_runners())
    t = rc.race_bets(m, _runners(), _pay(), score="score_mkt").set_index("race_id")
    assert t.loc["r1", "top1_hn"] == 3 and t.loc["r2", "top1_hn"] == 1


def test_race_conditions_counts_only_runners():
    flat = pd.DataFrame({"race_id": "r1", "date": "2023-01-07", "venue": "中山", "surface": "芝", "distance": 1600, "grade": "G1",
                         "race_class": "サラ系３歳以上 オープン", "track_condition": "良", "horse_number": range(1, 11),
                         "sex_age": ["牝4"] * 9 + ["牡4"], "scratched": [False] * 9 + [True]})
    r = rc.race_conditions(flat).iloc[0]
    assert r["field"] == "~9頭" and r["sex_cond"] == "牝馬限定" and r["class"] == "G1"   # 取消の牡馬は数えない


def test_holm():
    sig = rc.holm(np.array([0.001, 0.02, 0.04, 0.5]))        # 0.001 ≤ .05/4、0.02 > .05/3 で止まる
    assert sig.tolist() == [True, False, False, False]


def test_eb_shrink_bounds():
    dev = np.array([0.1, -0.1, 0.0, 0.3])
    se = np.array([0.05, 0.05, 0.05, 0.3])
    b = rc.eb_shrink(dev, se, np.ones(4, bool))
    assert np.all((b >= 0) & (b <= 1)) and b[3] < b[0]        # 誤差の大きいセルほど強く寄せる
    assert np.all(rc.eb_shrink(np.zeros(3), np.full(3, 0.1), np.ones(3, bool)) == 0)


def test_condition_helpers():
    assert rc.class_level("G1") == "G1" and rc.class_level("L") == "OP・L" and rc.class_level("未勝利") == "未勝利"
    assert rc.age_condition("サラ系３歳以上 １勝クラス") == "3歳以上" and rc.age_condition("サラ系２歳 新馬") == "2歳"
    assert rc.age_condition("サラ系３歳 未勝利") == "3歳"
    assert rc.season(12) == rc.season(2) and rc.season(9) == "秋(9-11月)"
    assert rc.pop_band(1) == "1番人気" and rc.pop_band(7) == "6番人気~" and rc.pop_band(float("nan")) == "不明"
    assert rc.odds_band(1.9) == "~1.9倍" and rc.odds_band(2.0) == "2.0-2.9倍" and rc.odds_band(10) == "10倍~"
