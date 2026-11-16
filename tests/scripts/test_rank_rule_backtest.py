"""rank_rule_backtest の決まり・絞り込み・日ごとの損益。"""

import numpy as np
import pytest

from src.scripts.data import rank_rule_backtest as rb


def test_rules_add_bad_going_only_for_rank_models():
    assert "going_bad" in rb.rules("B3")["umaren"]["weak"] and "going_bad" in rb.rules("A6")["wide"]["weak"]
    assert "going_bad" not in rb.rules("v6e_prod")["umaren"]["weak"]
    assert rb.rules("B3")["umaren"]["weak"] is not rb.rules("v6e_prod")["umaren"]["weak"]        # 共有のリストを書き換えない
    assert "going_bad" not in rb.rules("v6e_2016")["wide"]["weak"]


def test_selections():
    c = {"a": np.array([True, False, False, False]), "b": np.array([False, True, False, False]),
         "s": np.array([True, False, True, False])}
    sel = rb.selections(c, {"weak": ["a", "b"], "strong": ["s"]})
    assert sel["all"].tolist() == [True] * 4
    assert sel["no_weak"].tolist() == [False, False, True, True]
    assert sel["strong"].tolist() == [True, False, True, False]
    assert sel["no_weak_strong"].tolist() == [False, False, True, False]
    assert sel["drop:a"].tolist() == [False, True, True, True]


def test_daily_stats_drawdown_and_skip_days():
    dates = np.array(["d1", "d1", "d2", "d3", "d3"])
    ret = np.array([0.0, 300.0, 0.0, 0.0, 0.0])
    mask = np.array([True, True, True, False, False])       # d3 は買わない日
    s = rb.daily_stats(dates, ret, 100.0, mask)
    assert s["days"] == 2                                     # 買った日だけ数える
    assert s["total_pl"] == pytest.approx(0.0)                # d1: 300−200=+100、d2: −100
    assert s["max_drawdown"] == pytest.approx(100.0)
    assert s["worst_day"] == pytest.approx(-100.0) and s["pos_day_share"] == pytest.approx(0.5)
    assert s["day_stake"] == pytest.approx(150.0) and s["dd_days"] == pytest.approx(100.0 / 150.0)   # d1 は 200 円、d2 は 100 円
    assert s["day_roi_sd"] == pytest.approx(np.std([1.5, 0.0], ddof=1))
