import numpy as np
import pandas as pd

from src.pipeline.features import v6_ext as vx


def test_asof_train_years_never_future_for_2022_plus():
    assert vx.asof_train_years(2024) == [2020, 2021, 2022, 2023]
    assert max(vx.asof_train_years(2022)) == 2021
    assert 2020 not in vx.asof_train_years(2020)


def test_gate_coeff_uses_only_earlier_years():
    rng = np.random.default_rng(0)
    rows = []
    for y, slope in ((2020, 0.1), (2021, 0.1), (2022, -0.5)):
        for r in range(120):
            rid = f"{y}{r:04d}"
            no = np.arange(1, 13)
            rel = (no - 1) / 11
            for n, rl in zip(no, rel):
                rows.append((rid, y, "A|芝|1600", n, 12, np.clip(0.5 + slope * (rl - 0.5) + rng.normal(0, 0.1), 0, 1)))
    h = pd.DataFrame(rows, columns=["race_id", "year", "course", "horse_number", "field_size", "early_norm"])
    st = vx.gate_year_stats(h)
    k21 = vx.gate_coeff_asof(st, 2021, {"A|芝|1600": "芝"})
    k23 = vx.gate_coeff_asof(st, 2023, {"A|芝|1600": "芝"})
    assert vx.gate_coeff_asof(st, 2020, {"A|芝|1600": "芝"}) == {}
    assert 0.0 < k21["A|芝|1600"] < 0.2
    # 2022 年の結果は 2022 年の係数には入らない
    k22 = vx.gate_coeff_asof(st, 2022, {"A|芝|1600": "芝"})
    assert abs(k22["A|芝|1600"] - k21["A|芝|1600"]) < 0.05
    assert k23["A|芝|1600"] < k22["A|芝|1600"]


def test_gate_features_small_field_zero_shift():
    df = pd.DataFrame({"race_id": ["r"] * 4, "course": ["c"] * 4, "surface": ["芝"] * 4, "horse_number": [1, 2, 3, 4], "field_size": [4] * 4,
                       "pf_early": [0.2, 0.4, 0.6, 0.8], "lead5": [0.5, 0.0, 0.0, 0.7], "n5": [5, 5, 5, 5]})
    g = vx.gate_features(df, {"c": 0.3})
    assert (g["gb_shift"] == 0).all()
    assert g["gb_outer_lead"].tolist() == [0, 0, 0, 1]


def test_same_condition_rules():
    assert vx.same_condition("芝", 2000, "芝", 1800)
    assert not vx.same_condition("芝", 2000, "ダ", 2000)
    assert not vx.same_condition("芝", 2000, "芝", 1700)
    assert vx.same_condition("芝", 2400, "芝", 2000)
    assert not vx.same_condition(None, 2000, None, 2000)


def test_prior_window_strictly_earlier_and_last5():
    d = pd.date_range("2024-01-01", periods=8, freq="7D")
    runs = pd.DataFrame({"horse_id": ["h"] * 8, "date": d, "idx": [50, 60, 55, 70, 40, 45, 52, np.nan], "surf": ["芝"] * 8, "dist": [2000] * 8})
    lines = pd.Series(50.0, index=runs.index)
    f = vx.prior_window_features(runs, lines)
    assert np.isnan(f.loc[0, "cb_max5"])
    assert f.loc[1, "cb_max5"] == 50
    assert f.loc[6, "cb_max5"] == 70                       # 直前 5 走 = 60,55,70,40,45
    assert f.loc[6, "cb_cnt_above5"] == 3
    assert f.loc[7, "cb_max5"] == 70 and f.loc[7, "cb_same_n"] == 5


def test_prior_window_same_date_not_used():
    d = pd.to_datetime(["2024-01-01", "2024-01-01"])
    runs = pd.DataFrame({"horse_id": ["h", "h"], "date": d, "idx": [80.0, 30.0], "surf": ["芝", "芝"], "dist": [2000, 2000]})
    f = vx.prior_window_features(runs, pd.Series(50.0, index=runs.index))
    assert f["cb_max5"].isna().all()


def test_expected_style_fit_matches_manual():
    L = np.ones((3, 3, 4))
    L[2, :, 0] = 0.6                                       # ハイだと逃げが不利
    pace = np.array([[0.2, 0.3, 0.5]])
    shape = np.array([[0.2, 0.6, 0.2]])
    f = vx.expected_style_fit(pace, shape, np.array([0]), L)
    assert abs(f.loc[0, "pt_style_lift"] - (0.5 * 0.6 + 0.5 * 1.0)) < 1e-9
    assert f.loc[0, "pt_style_sd"] > 0
    g = vx.expected_style_fit(pace, shape, np.array([-1]), L)
    assert g["pt_style_lift"].isna().all()


def test_normalize_owner_merges_corporate_forms():
    assert vx.normalize_owner("（株）サンデーレーシング") == vx.normalize_owner("株式会社サンデーレーシング")
    assert vx.normalize_owner("有限会社 ABC ファーム") == vx.normalize_owner("(有)ABCファーム")
    assert vx.normalize_owner(None) == ""
    assert vx.normalize_owner("山田太郎") != vx.normalize_owner("山田次郎")


def test_owner_asof_strictly_earlier_and_merges_spellings():
    d = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-02-01", "2024-03-01"])
    runs = pd.DataFrame({"race_id": list("abcd"), "horse_id": list("wxyz"), "date": d, "field_size": [10, 10, 10, 10],
                         "finish_position": [1, 5, 2, 9], "okey": ["O", "O", "O", "O"], "rkey": ["O", "O", "O2", "O"]})
    n = vx.owner_asof(runs, "okey").set_index("race_id")
    assert n.loc["a", "n"] == 0 and n.loc["b", "n"] == 0           # 同日は含めない
    assert n.loc["c", "n"] == 2 and n.loc["d", "n"] == 3
    r = vx.owner_asof(runs, "rkey").set_index("race_id")
    assert r.loc["c", "n"] == 0 and r.loc["d", "n"] == 2           # 表記が分かれるとレース c の履歴が別の馬主になる
