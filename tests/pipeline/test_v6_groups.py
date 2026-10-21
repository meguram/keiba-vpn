import numpy as np
import pandas as pd

from src.pipeline.features import v6_groups as vg


def _df():
    return pd.DataFrame({"race_id": ["r1", "r1", "r1", "r2", "r2"], "horse_id": list("abcde"),
                         "x": [1.0, 2.0, 3.0, 5.0, 7.0], "pf_early": [0.1, 0.5, 0.9, 0.2, np.nan],
                         "EASE2_drain": [0.1, 0.0, -0.1, 0.3, 0.2], "EASE2_capacity": [0.0, 0.1, 0.2, 0.1, 0.0],
                         "lp_p_slow": [0.2] * 5, "lp_p_mid": [0.5] * 5, "lp_p_high": [0.3] * 5, "lp_exp_f3_dev": [0.1] * 5, "lp_exp_l3_dev": [0.2] * 5})


def test_race_rel_is_within_race_only():
    d = _df()
    r = vg.race_rel(d, ["x"])
    assert r.loc[0, "v6r_x_rk"] == 0.0 and r.loc[2, "v6r_x_rk"] == 1.0 and r.loc[1, "v6r_x_rk"] == 0.5
    assert abs(r.loc[0, "v6r_x_gap"] - (1.0 - 2.0)) < 1e-6
    assert abs(r.loc[3, "v6r_x_gap"] - (5.0 - 6.0)) < 1e-6          # r1 の値はレース r2 の相対に影響しない
    d2 = d.copy()
    d2.loc[0, "x"] = 100.0
    r2 = vg.race_rel(d2, ["x"])
    assert (r.loc[3:, :] == r2.loc[3:, :]).all().all()                # 別レースの変更は影響しない


def test_lap_entropy_bounds():
    e = vg.lap_entropy(_df())
    assert (e["v6_lp_entropy"] > 0).all() and (e["v6_lp_entropy"] <= np.log(3) + 1e-6).all()
    assert np.allclose(e["v6_lp_hi_minus_slow"], 0.1)
    assert np.allclose(e["v6_lp_pmax"], 0.5)


def test_interactions_shape_and_nan():
    i = vg.interactions(_df())
    assert list(i.columns) == ["v6_front_x_pace", "v6_pf_x_f3", "v6_drain_x_hs", "v6_cap_x_hs", "v6_drain_x_l3"]
    assert np.isnan(i.loc[4, "v6_pf_x_f3"])                           # pf_early が欠損の馬は NaN
    assert i.loc[0, "v6_front_x_pace"] > i.loc[2, "v6_front_x_pace"]   # 先頭寄りほど値が大きい（0.5 - 順位）× (high−slow>0)
