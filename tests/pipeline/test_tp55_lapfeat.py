import numpy as np
import pandas as pd

from src.pipeline.features import tp55_lapfeat as lf


def _runs():
    rows = []
    d0 = pd.Timestamp("2024-01-06")
    for k in range(10):
        rows.append({"race_id": f"R{k}", "horse_id": "H1", "date": d0 + pd.Timedelta(days=14 * k), "early": 0.05 + 0.05 * k, "jockey_id": "J1"})
    rows.append({"race_id": "X", "horse_id": "H2", "date": d0, "early": 0.5, "jockey_id": "J2"})
    return pd.DataFrame(rows)


def test_parse_and_laps_meta():
    assert lf.parse_laps("[12.4, 10.7, 11.1, 11.4, 11.5, 11.8]") == [12.4, 10.7, 11.1, 11.4, 11.5, 11.8]
    assert lf.parse_laps("[1, 2]") is None and lf.parse_laps(None) is None
    assert lf.n_laps_of(1700) == 9 and lf.first_lap_meters(1700) == 100
    assert lf.n_laps_of(1200) == 6 and lf.first_lap_meters(1200) == 200
    assert lf.n_laps_of(1150) == 6 and lf.first_lap_meters(1150) == 150


def test_first_corner_norm():
    assert lf.first_corner_norm("1-1-2", 16) == 0.0
    assert abs(lf.first_corner_norm("9-8", 17) - 0.5) < 1e-9
    assert np.isnan(lf.first_corner_norm("", 16)) and np.isnan(lf.first_corner_norm("1-1", 3))


def test_asof_uses_only_strictly_earlier_runs():
    runs = _runs()
    ent = pd.DataFrame({"race_id": ["E1", "E2"], "horse_id": ["H1", "H1"], "date": [pd.Timestamp("2024-01-20"), pd.Timestamp("2024-01-06")]})
    f = lf.asof_horse_features(ent, runs)
    # E1 は 2024-01-20: 前の走は R0(01-06) と R1(01-20 は同日なので含めない) → R0 だけ
    assert f.loc[0, "h_n_runs"] == 1 and abs(f.loc[0, "h_early_w"] - 0.05) < 1e-9
    assert f.loc[1, "h_n_runs"] == 0 and np.isnan(f.loc[1, "h_early_w"])


def test_asof_exact_no_leak_when_future_changes():
    runs = _runs()
    ent = pd.DataFrame({"race_id": ["E1"], "horse_id": ["H1"], "date": [pd.Timestamp("2024-03-02")]})
    a = lf.asof_horse_features(ent, runs)
    r2 = runs.copy()
    r2.loc[r2["date"] >= pd.Timestamp("2024-03-02"), "early"] = 0.99          # 本走以降の結果を書き換える
    b = lf.asof_horse_features(ent, r2)
    pd.testing.assert_frame_equal(a, b)


def test_weighted_early_and_front_share():
    runs = _runs()
    ent = pd.DataFrame({"race_id": ["E"], "horse_id": ["H1"], "date": [pd.Timestamp("2024-07-01")]})
    f = lf.asof_horse_features(ent, runs).iloc[0]
    last5 = [0.05 + 0.05 * k for k in (5, 4, 3, 2, 1)]          # 直近が先頭: R5..R1 ではなく日付が前の 10 走のうち直近 5 つ
    assert f["h_n_known5"] == 5
    assert 0 < f["h_early_w"] < 1


def test_jockey_early_strictly_before():
    rows = []
    for k in range(40):
        rows.append({"race_id": f"J{k}", "horse_id": f"H{k}", "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=k), "early": 0.2, "jockey_id": "J1"})
    runs = pd.DataFrame(rows)
    ent = pd.DataFrame({"jockey_id": ["J1", "J1"], "date": [pd.Timestamp("2024-02-15"), pd.Timestamp("2024-01-10")]})
    s = lf.asof_jockey_early(ent, runs)
    assert abs(s.iloc[0] - 0.2) < 1e-9 and np.isnan(s.iloc[1])           # 2 つ目は前の走が 30 に満たない


def test_fit_par_shrinks_to_group():
    base = {"venue": "東京", "surface": "芝", "distance": 1600, "direction": "左"}
    rs = pd.DataFrame([dict(base, laps=[12.0] * 8) for _ in range(10)] + [dict(base, venue="中山", direction="右", laps=[13.0] * 8)])
    par = lf.fit_par(rs)
    v = par.vec("中山", "芝", 1600, "右")
    assert 12.0 < v[0] < 13.0          # 1 件だけのコースは距離×面の平均へ縮約
    assert abs(par.vec("東京", "芝", 1600, "左")[0] - 12.0) < 0.2
    assert par.vec("札幌", "芝", 1600, "右") is not None            # 未知のコースは距離×面の平均
    assert par.vec("札幌", "芝", 999, "右") is None


def test_race_features_one_basic():
    g = pd.DataFrame({
        "horse_number": [1, 2, 3, 4, 5, 6], "h_early_w": [0.05, 0.1, 0.3, 0.5, 0.8, np.nan], "h_p_front": [0.8, 0.6, 0.1, 0, 0, np.nan],
        "h_p_stalk": [1, 1, 0.6, 0.2, 0, np.nan], "h_front8": [0.7, 0.5, 0.1, 0, 0, np.nan], "h_n_known5": [5, 5, 5, 5, 5, 0],
        "ten1f": [12.6, 12.9, np.nan, 13.4, 13.6, np.nan], "ten1f_best": [12.4, 12.7, np.nan, 13.1, 13.5, np.nan],
        "jockey_early": [0.2, 0.3, np.nan, 0.5, 0.6, np.nan], "age": [3, 4, 3, 5, 4, 6], "is_female": [0, 1, 0, 0, 1, 0],
        "jockey_weight": [55, 56, 55, 57, 54, 55], "h_n_runs": [3, 8, 10, 4, 0, 2],
    })
    o = lf.race_features_one(g)
    assert o["n_entrants"] == 6 and o["n_known"] == 5 and o["n_front_hard"] == 2 and o["n_stalk_hard"] == 3
    assert o["no_leader"] == 0 and o["ten_cnt"] == 4 and abs(o["ten_min"] - 12.6) < 1e-9
    assert o["no_f1"] == 1 and o["no_f2"] == 2 and o["front_adjacent_pairs"] >= 1
    g2 = g.copy()
    g2["h_early_w"] = [0.5, 0.6, 0.7, 0.8, 0.9, np.nan]
    assert lf.race_features_one(g2)["no_leader"] == 1


def test_sameday_uses_only_earlier_rounds():
    par0 = lf.fit_par(pd.DataFrame([{"venue": "東京", "surface": "芝", "distance": 1600, "direction": "左", "laps": [12.0] * 8}] * 5))
    base = {"date": pd.Timestamp("2024-05-05"), "venue": "東京", "surface": "芝", "distance": 1600, "direction": "左"}
    rs = pd.DataFrame([
        dict(base, race_id="A1", round_no=1, laps=[12.0, 11.0, 11.0, 11.0, 11.0, 11.0, 12.0, 13.0]),
        dict(base, race_id="A2", round_no=2, laps=[12.0] * 8),
        dict(base, race_id="A3", round_no=3, laps=[12.0] * 8),
    ])
    sd = lf.sameday_features(rs, par0).set_index("race_id")
    assert sd.loc["A1", "sd_n"] == 0 and np.isnan(sd.loc["A1", "sd_q_mean"])
    assert sd.loc["A2", "sd_n"] == 1 and sd.loc["A3", "sd_n"] == 2
    rs2 = rs.copy()
    rs2.at[2, "laps"] = [9.0] * 8                    # 3R の結果を変えても 2R までの特徴は変わらない
    sd2 = lf.sameday_features(rs2, par0).set_index("race_id")
    assert sd2.loc["A2", "sd_q_mean"] == sd.loc["A2", "sd_q_mean"]
