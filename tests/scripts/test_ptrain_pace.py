import numpy as np
import pandas as pd

from src.scripts.data import ptrain_pace as pp


def _races(n=40):
    d = pd.date_range("2022-01-01", periods=n, freq="7D")
    g = pd.DataFrame({"race_id": [f"r{i}" for i in range(n)], "date": d, "ckey": "A", "bad": 0, "grade": "x",
                      "dist_band": 2, "surface": "芝", "f3": 35.0 + np.arange(n) % 3, "f5": 59.0 + np.arange(n) % 4})
    return g


def test_course_pace_excludes_same_and_future_races():
    g = _races()
    a = pp.course_pace(g).set_index("race_id")
    assert np.isnan(a.loc["r0", "P77_c3_mean"])
    assert abs(a.loc["r1", "P77_c3_mean"] - g.loc[0, "f3"]) < 1e-9
    g2 = g.copy()
    g2.loc[20:, ["f3", "f5"]] += 100          # 未来の値を変えても、それ以前の行は変わらない
    b = pp.course_pace(g2).set_index("race_id")
    assert a.loc["r0":"r20", "P77_c3_mean"].equals(b.loc["r0":"r20", "P77_c3_mean"])


def test_horse_fit_uses_only_past_runs():
    d = pd.date_range("2022-01-01", periods=6, freq="7D")
    r = pd.DataFrame({"race_id": [f"r{i}" for i in range(6)], "horse_id": "h", "date": d, "finish_position": [1, 2, 3, 1, 2, 3], "field_size": 10})
    g = pd.DataFrame({"race_id": r["race_id"], "z": [0.5, -0.5, 0.0, 1.0, 0.2, 0.3]})
    tg = pd.DataFrame({"race_id": ["t"], "horse_id": ["h"], "date": [d[3]], "P77_pred_z": [0.0]})
    a = pp.horse_fit(r, g, tg)
    r2, g2 = r.copy(), g.copy()
    g2.loc[3:, "z"] = 99.0                    # 対象日以降の走は結果に影響しない
    b = pp.horse_fit(r2, g2, tg)
    assert a.equals(b) and a.loc[0, "P77_R_n"] == 3


def test_horse_pass_uses_only_past_runs_and_field_relative():
    d = pd.date_range("2022-01-01", periods=6, freq="7D")
    est = pd.DataFrame({"race_id": [f"r{i}" for i in range(6)], "horse_id": "h", "date": d,
                        "est_lag3": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "est_lag5": [0.2] * 6})
    tg = pd.DataFrame({"race_id": ["t", "t"], "horse_id": ["h", "k"], "date": [d[3]] * 2,
                       "P77_c3_mean": 35.0, "P77_c3_sd": 0.5, "P77_c5_mean": 59.0, "P77_c5_sd": 0.7})
    a = pp.horse_pass(est, tg)
    est2 = est.copy()
    est2.loc[3:, ["est_lag3", "est_lag5"]] = 9.0       # 対象日以降の走は影響しない
    assert a.equals(pp.horse_pass(est2, tg))
    assert a.loc[0, "P77_E_n"] == 3 and abs(a.loc[0, "P77_E_lag3_last"] - 0.3) < 1e-6
    assert np.isnan(a.loc[1, "P77_E_lag3_ewm"])         # 履歴なしの馬は欠損
