"""第 2 段（市場とのずれを学習するモデル）の道具のテスト。オッズ・人気は入力の特徴量に入れず、offset にだけ使う決まりも確かめる。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.scripts.data import model_cycles_market as mk


def test_market_features_are_rejected():
    mk.check_no_market_features(["f3_r", "sr_exp", "tl_f_t1"])            # 通常の列は通る
    for bad in (["win_odds"], ["popularity"], ["o_pop_rank"], ["mkt_q"], ["tansho_ret"], ["mk_vol"], ["win_pop"], ["chalk_n"]):
        with pytest.raises(SystemExit):
            mk.check_no_market_features(["f3_r"] + bad)


def test_q_from_odds_normalizes_and_drops_missing():
    race = np.array(["a", "a", "a", "b", "b"])
    q = mk.q_from_odds(race, np.array([2.0, 4.0, np.nan, 1.5, 3.0]))
    assert np.isnan(q[2])
    assert q[:2].sum() == pytest.approx(1.0)
    assert q[0] == pytest.approx(2 / 3)                                   # 1/2 : 1/4 → 2:1
    assert q[3:].sum() == pytest.approx(1.0)
    qb = mk.q_from_odds(race[:2], np.array([2.0, 4.0]), beta=2.0)          # β>1 で本命が重くなる
    assert qb[0] > q[0]


def _frame():
    # 2 レース。レース r1 は予測 1 位が 1 番人気（ずれ小）、2 番手は人気薄でずれが大きい
    return pd.DataFrame({
        "race_id": ["r1"] * 3 + ["r2"] * 3,
        "horse_id": list("abcdef"),
        "p_win": [0.5, 0.3, 0.2, 0.2, 0.5, 0.3],
        "q": [0.6, 0.1, 0.3, 0.3, 0.4, 0.3],
        "popularity": [1, 3, 2, 2, 1, 3],
        "finish_position": [2, 1, 3, 1, 2, 3],
        "ret_tan": [0, 900, 0, 330, 0, 0],
        "ret_fuku": [120, 250, 150, 140, 110, 160],
    }).assign(r=lambda d: np.log(d["p_win"] / d["q"]))


def test_pick_rules():
    d = _frame()
    assert list(mk.pick(d, "p")["horse_id"]) == ["a", "e"]                 # p_win の 1 位（どちらも 1 番人気）
    assert list(mk.pick(d, "nofav")["horse_id"]) == ["b", "f"]             # 1 番人気を除く
    assert list(mk.pick(d, "r")["horse_id"]) == ["b", "e"]                 # ずれの 1 位
    g = mk.pick(d, "gate:0.2")                                             # 1 番人気が 1 位でも、ずれ r ≥ 0.2 のときだけ採る
    assert list(g["horse_id"]) == ["b", "e"]                               # a は r=log(0.5/0.6)<0 → 除く、e は r=log(0.5/0.4)≈0.22 → 採る
    s = mk.summarize(mk.pick(d, "nofav"))
    assert s["fav_share"] == 0 and s["roi_tan"] == pytest.approx(900 / 200)


def test_pick_r_p05_falls_back_when_no_horse_reaches_5pct():
    d = pd.DataFrame({"race_id": ["r"] * 3, "horse_id": list("xyz"), "p_win": [0.04, 0.03, 0.02], "q": [0.1, 0.01, 0.2],
                      "popularity": [2, 3, 1], "finish_position": [1, 2, 3], "ret_tan": [500, 0, 0], "ret_fuku": [200, 150, 120]})
    d["r"] = np.log(d["p_win"] / d["q"])
    assert list(mk.pick(d, "r_p05")["horse_id"]) == ["x"]                  # 全頭 5% 未満なら p_win の 1 位


def test_drift_table_counts_favorite_agreement():
    pre = pd.DataFrame({"race_id": ["r1"] * 3, "horse_number": [1, 2, 3], "q": [0.5, 0.3, 0.2]})
    fin = pd.DataFrame({"race_id": ["r1"] * 3, "horse_number": [1, 2, 3], "q": [0.3, 0.5, 0.2]})
    t = mk.drift_table(pre, fin)
    assert t["races"] == 1 and t["fav_agree"] == 0.0
    t2 = mk.drift_table(pre, pre)
    assert t2["fav_agree"] == 1.0 and t2["logit_diff_sd"] == pytest.approx(0.0)
