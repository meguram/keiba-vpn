"""穴馬予想モデルの道具（src/scripts/data/longshot_model.py）の純粋関数のテスト。"""

import argparse
import inspect

import numpy as np
import pandas as pd
import pytest

from src.scripts.data import longshot_model as L
from src.scripts.data import model_cycles_market as mm


def _X():
    return pd.DataFrame({
        "field_size": [8, 12, 16, 16],
        "weight_rule": pd.Categorical(["ハンデ", "定量", "ハンデ", "馬齢"]),
        "venue_code": pd.Categorical(["04", "05", "10", "06"]),
        "grade": pd.Categorical(["新馬", "未勝利", "2勝", "OP"]),
        "lag1_l3f_rank": [0.1, 0.9, np.nan, 0.2],
        "d_prev_fin": [9, 10, 5, 1],
        "d_prev_field": [11, 11, 9, 16],
        "f2_nknown": [8, 6, 8, 16],
        "f2_nfront": [2, 3, 4, 8],
        "a_style": [1.0, 0.0, 0.5, np.nan],
    })


def test_longshot_features_values():
    rc = pd.Series(["サラ系２歳 新馬", "サラ系３歳 未勝利", "サラ系３歳以上 ２勝クラス", "サラ系３歳以上 オープン"])
    o = L.longshot_features(_X(), rc)
    assert list(o.columns) == L.LS_COLS
    assert o["ls_field_band"].tolist() == [0, 1, 3, 3]
    assert o["ls_handicap"].tolist() == [1, 0, 1, 0]
    assert o["ls_local_handicap"].tolist() == [1, 0, 1, 0]
    assert o["ls_class_grp"].tolist() == [0, 1, 2, 3]
    assert np.isclose(o["ls_unlucky"].iloc[0], 0.8 - 0.1)            # 前走 9 着 / 11 頭 → (9−1)/(11−1)
    assert o["ls_unlucky_flag"].iloc[0] == 1 and o["ls_unlucky_flag"].iloc[1] == 0 and np.isnan(o["ls_unlucky_flag"].iloc[2])
    assert np.isclose(o["ls_front_ratio"].iloc[3], 0.5) and np.isclose(o["ls_front_ratio"].iloc[1], 0.5)   # 分母は脚質の分かる頭数
    assert np.isclose(o["ls_closer_x_front"].iloc[0], 0.25) and np.isclose(o["ls_front_x_front"].iloc[1], 0.5)


def test_class_group_uses_both_sources():
    grade = pd.Series(["", "", "2勝", "OP", "", "未勝利"])         # 2020 年からは 1〜3 勝の grade が空
    rc = pd.Series(["サラ系３歳以上 １勝クラス", "サラ系３歳以上 オープン", "サラ系４歳以上", "サラ系３歳以上", "サラ系２歳 新馬", ""])
    assert L.class_group(grade, rc).tolist() == [2, 3, 2, 3, 0, 1]


def test_ls_cols_groups_cover_all():
    assert sorted(L.LS_COLS) == sorted(L.LS_RACE_COLS + L.LS_HORSE_COLS)
    assert sorted(sum(L.LS_GROUPS.values(), [])) == sorted(L.LS_COLS)


def test_ls_cols_pass_market_check():
    mm.check_no_market_features(L.LS_COLS)


def test_permute_control():
    race = np.array(["a"] * 5 + ["b"] * 4 + ["c"] * 3 + ["d"] * 3)
    cl = np.array(["d1"] * 9 + ["d2"] * 6)                       # a・b は同じ開催日 × 場、c・d は別
    rv = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    df = pd.DataFrame({c: np.arange(len(race), dtype=float) for c in L.LS_COLS})
    for c in L.LS_RACE_COLS:
        df[c] = [rv[r] for r in race]
    changed_race = False
    for seed in range(10):
        p = L.permute_control(df, race, cl, seed)
        for r in "abcd":                                         # 馬ごとの列は、レースの中の値の集まりが変わらない
            assert sorted(p["ls_unlucky"][race == r]) == sorted(df["ls_unlucky"][race == r])
            assert p.loc[race == r, "ls_handicap"].nunique() == 1   # レースの列は、レースの中で同じ値のまま
        assert set(p.loc[cl == "d1", "ls_handicap"]) <= {1.0, 2.0}  # 開催日 × 場をまたがない
        changed_race |= p.loc[race == "a", "ls_handicap"].iloc[0] != 1.0
    assert changed_race


def test_calibration_corrects_bias():
    rng = np.random.default_rng(0)
    n = 20000
    lq = rng.normal(-2.5, 1.0, n)
    true = 1 / (1 + np.exp(-(1.2 * lq + 0.5)))              # 市場の logit を 1.2 倍した真の確率（本命-大穴の偏り）
    y = (rng.random(n) < true).astype(int)
    des = L.calib_design(lq, None, np.zeros(n, int), np.zeros(n, int))
    m = L.calibrate_fit(des, y)
    p = 1 / (1 + np.exp(-L.calibrate_logit(m, des)))
    assert np.mean(L.binary_logloss(y, p)) < np.mean(L.binary_logloss(y, 1 / (1 + np.exp(-lq))))


def test_cluster_boot_mean_interval():
    v = np.r_[np.full(100, 1.0), np.full(100, 3.0)]
    c = np.r_[np.arange(100) % 10, 10 + np.arange(100) % 10]
    m, lo, hi = L.cluster_boot_mean(v, c, n_boot=500)
    assert m == 2.0 and lo < 2.0 < hi


def test_fill_outside():
    assert L.fill_outside(np.array([0.1, 0.2, 0.3]), np.array([0.9, 0.8, 0.7]), np.array([False, True, True])).tolist() == [0.9, 0.2, 0.3]


def test_norm_race_cap():
    race = np.array(["a", "a", "a", "a"])
    p = L.norm_race_cap(race, np.array([0.9, 0.5, 0.3, 0.3]), 3.0)
    assert p.max() < 1.0 and p.min() > 0.0


def test_drift_pool_reads_no_outcome():
    src = inspect.getsource(L.drift_pool_2026)
    assert "NO_OUTCOME_COLS" in src
    assert "finish_position" not in L.NO_OUTCOME_COLS and "payoff" not in L.NO_OUTCOME_COLS


def test_gate1_refuses_eval_years():
    with pytest.raises(SystemExit):
        L.cmd_gate1(argparse.Namespace(years="2021,2022", variants="B0", force=False, smoke=True))
    with pytest.raises(SystemExit):
        L.cmd_gate1(argparse.Namespace(years="2019", variants="A1,B0", force=False, smoke=True))


def test_gate1_verdict_rule():
    pre = {"primary_target": "top3", "candidates": ["B1", "A1"]}
    s = {"top3": {"diff": {"B1-B0": {"mean": -0.001, "lo": -0.002, "hi": -0.0001, "years": {"2019": -1, "2020": -1, "2021": 1}},
                           "A1-B0": {"mean": 0.0, "lo": -0.001, "hi": 0.001, "years": {"2019": -1, "2020": 1, "2021": 1}},
                           "B1perm-B0": {"mean": 0.0, "lo": -0.001, "hi": 0.001, "years": {}}}}}
    v = L.gate1_verdict(s, pre)
    assert v["B1"]["signal"] and not v["A1"]["signal"] and v["any_signal"]
    s["top3"]["diff"]["B1perm-B0"]["hi"] = -0.0005                # 並べ替えでも良くなるなら、兆しとみなさない
    assert not L.gate1_verdict(s, pre)["any_signal"]
    del s["top3"]["diff"]["B1perm-B0"]                              # 対照が流れていなければ、兆しとみなさない
    assert not L.gate1_verdict(s, pre)["any_signal"]
