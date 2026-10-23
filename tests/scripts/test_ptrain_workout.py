import numpy as np
import pandas as pd

from src.scripts.data import ptrain_workout as pw


def _R(n=240):
    d = pd.date_range("2022-01-01", periods=n, freq="7D")
    pat = np.where(np.arange(n) % 20 == 0, "slope|強め|速|多", "wood|馬也|中|並")
    return pd.DataFrame({"race_id": [f"r{i}" for i in range(n)], "horse_id": [f"h{i}" for i in range(n)], "date": d, "trainer_id": "t1",
                         "res": np.where(np.arange(n) % 20 == 0, 0.3, 0.0), "pat": pat, "n3w": 3})


def test_asof_workout_flags_rare_pattern_and_ignores_future():
    R = _R()
    a = pw.asof_workout(R, "pat", "P")
    assert (a.loc[100:, "P_flag"] == "勝負").any()
    R2 = R.copy()
    R2.loc[150:, "res"] = 0.5
    R2.loc[150:, "pat"] = "wood|馬也|中|並"                 # 未来の結果を変えても、それ以前の行は変わらない
    b = pw.asof_workout(R2, "pat", "P")
    cols = [c for c in a.columns if c != "P_flag"]
    assert a.loc[:149, cols].equals(b.loc[:149, cols])


def test_patterns_five_views():
    d = pd.DataFrame({"horse_id": "h", "race_date": "2022-01-15", "kind": ["slope", "wood", "slope"], "leg_color": ["強め", "馬也", "馬也"],
                      "l1": [12.0, 12.5, 12.8], "t1": [12.0, 12.5, 12.8], "t2": [24.5, 25.0, 25.9], "lmid": [12.5, 12.5, 13.0], "accel": [-0.5, 0.0, 0.5], "t4": [52.0, 53.0, 54.0], "days_before": [1, 4, 8]})
    p = pw.patterns(d)
    assert len(p) == 1 and p.loc[0, "v_prevday"].startswith("前日有") and p.loc[0, "v_route"] == "slope←slope"
    assert "加速" in p.loc[0, "v_shape"] and p.loc[0, "v_week1"].endswith("減速")


def test_lap_group_definitions():
    f = lambda a, b: pw.lap_group(pd.Series([b]), pd.Series([a + b]))[0]       # a=2F目(終いの前の1F), b=終い1F
    assert f(13.1, 12.5) == "A1" and f(12.8, 13.1) == "B1"
    assert f(12.5, 12.1) == "A2" and f(12.2, 12.6) == "B2"
    assert f(12.5, 11.9) == "A3" and f(11.8, 12.7) == "B3"
