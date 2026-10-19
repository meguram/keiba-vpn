"""tp-40: 調教の時計の出方（調子・成長）の特徴量の as-of 検査と値の確認。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.features import tp40_condition as C


def synth(n_horses=160, n_races=9, seed=0):
    rng = np.random.default_rng(seed)
    laps, runs = [], []
    rid = 0
    base = pd.Timestamp("2021-01-03")
    for h in range(n_horses):
        d = base + pd.Timedelta(days=int(rng.integers(0, 40)))
        sex_age = rng.choice(["牡3", "牝4", "セ5"])
        trainer = f"t{h % 7}"
        for _ in range(n_races):
            d = d + pd.Timedelta(days=int(rng.integers(14, 90)))
            rid += 1
            fs = 14
            fin = int(rng.integers(1, fs + 1))
            runs.append({"race_id": f"r{rid:05d}", "horse_id": f"h{h:04d}", "date": d.strftime("%Y-%m-%d"), "trainer_id": trainer, "finish_position": fin, "field_size": fs, "sex_age": sex_age,
                         "d_rest": float(rng.integers(14, 150))})
            for k in range(int(rng.integers(2, 6))):
                db = int(rng.integers(1, 40))
                td = d - pd.Timedelta(days=db)
                laps.append({"horse_id": f"h{h:04d}", "train_date": td.strftime("%Y-%m-%d"), "race_date": d.strftime("%Y-%m-%d"), "course": rng.choice(["栗坂", "美坂", "栗W"]),
                             "kind": "slope", "center": "栗", "cond": "良", "rider": "", "position": np.nan, "leg_color": rng.choice(["馬也", "強め", "一杯"]), "rank": "B",
                             "evaluation": "", "t1": 12.5 + rng.normal(0, 0.5), "t4": 53 + rng.normal(0, 2), "days_before": db,
                             "t4z": rng.normal(), "t1z": rng.normal(), "accelz": rng.normal()})
    return pd.DataFrame(laps), pd.DataFrame(runs)


def build(laps, runs):
    return C.build_features(runs, laps, with_ref=False)


def assert_same(a: dict, b: dict, race_ids: set):
    for g in a:
        x = a[g][a[g]["race_id"].isin(race_ids)].sort_values(["race_id", "horse_id"]).reset_index(drop=True)
        y = b[g][b[g]["race_id"].isin(race_ids)].sort_values(["race_id", "horse_id"]).reset_index(drop=True)
        assert list(x.columns) == list(y.columns) and len(x) == len(y), g
        for c in x.columns:
            if c in ("race_id", "horse_id"):
                assert (x[c] == y[c]).all()
                continue
            xv, yv = x[c].to_numpy(dtype=float), y[c].to_numpy(dtype=float)
            assert np.array_equal(np.isnan(xv), np.isnan(yv)), f"{g}.{c} 欠損の位置が違う"
            assert np.array_equal(xv[~np.isnan(xv)], yv[~np.isnan(yv)]), f"{g}.{c} 値が違う"          # 完全一致（端数も許さない）


def test_exact_asof_under_truncation():
    laps, runs = synth()
    cut = pd.Timestamp("2021-12-31")
    full = build(laps, runs)
    keep = runs[pd.to_datetime(runs["date"]) <= cut]
    lp = laps[pd.to_datetime(laps["race_date"]) <= cut]
    part = build(lp, keep)
    ids = set(keep["race_id"])
    assert len(ids) > 500
    assert_same(full, part, ids)
    nonnull = {g: int(part[g].drop(columns=["race_id", "horse_id"]).notna().sum().sum()) for g in part}
    assert all(v > 0 for v in nonnull.values()), nonnull               # 群がすべて値を持つ（空のまま通る検査にしない）


def test_future_results_and_workouts_do_not_change_past():
    laps, runs = synth(seed=1)
    cut = pd.Timestamp("2021-10-01")
    a = build(laps, runs)
    runs2 = runs.copy()
    late = pd.to_datetime(runs2["date"]) > cut
    runs2.loc[late, "finish_position"] = 1 + (runs2.loc[late, "finish_position"].to_numpy() * 7) % 14        # 未来の着順を変える
    laps2 = laps.copy()
    laps2.loc[pd.to_datetime(laps2["race_date"]) > cut, ["t4z", "t1z", "accelz"]] = 0.123                  # 未来の追い切りを変える
    b = build(laps2, runs2)
    past = set(runs.loc[~late, "race_id"])
    assert_same(a, b, past)


def test_cycle_values_by_hand():
    runs = pd.DataFrame([{"race_id": "R1", "horse_id": "H", "date": "2022-03-06", "trainer_id": "T", "finish_position": 3, "field_size": 10, "sex_age": "牡3", "d_rest": 30.0}])
    rows = []
    for db, t4z, t1z in ((21, 1.0, 0.0), (14, 0.0, 0.0), (7, -1.0, -2.0)):                                 # 古い順に速くなる
        rows.append({"horse_id": "H", "train_date": (pd.Timestamp("2022-03-06") - pd.Timedelta(days=db)).strftime("%Y-%m-%d"), "race_date": "2022-03-06", "course": "栗坂", "kind": "slope", "center": "栗",
                     "cond": "良", "rider": "", "position": np.nan, "leg_color": "強め", "rank": "B", "evaluation": "", "t1": 12.0, "t4": 52.0, "days_before": db, "t4z": t4z, "t1z": t1z, "accelz": 0.0})
    out = build(pd.DataFrame(rows), runs)["CY"].iloc[0]
    g = [-0.5, 0.0, 1.5]                                                                                    # g = -(t4z + t1z) / 2
    assert out["wc_cc_last"] == g[2] and abs(out["wc_cc_d1"] - (g[2] - g[1])) < 1e-12 and abs(out["wc_cc_dcyc"] - (g[2] - g[0])) < 1e-12
    assert out["wc_cc_n"] == 3 and out["wc_cc_best"] == 1.5
    t = np.array([0.0, 1.0, 2.0])
    assert abs(out["wc_cc_slope"] - np.polyfit(t, np.array(g), 1)[0]) < 1e-9                              # 1 週あたりの傾き
    assert abs(out["wc_cc_vsmean"] - (g[2] - np.mean(g[:2]))) < 1e-12


def test_self_group_uses_only_previous_cycles_and_keeps_old_hard_best():
    runs, laps = [], []
    # 3 サイクル: 1 つ目に強い追い切り（g 大）、2 つ目は馬なりだけ（強い追い切りなし）、3 つ目に強い追い切り
    spec = [("2022-01-09", [("強め", 2.0)]), ("2022-03-06", [("馬也", 0.5)]), ("2022-05-01", [("強め", 1.0)])]
    for i, (d, ws) in enumerate(spec):
        runs.append({"race_id": f"R{i}", "horse_id": "H", "date": d, "trainer_id": "T", "finish_position": 2, "field_size": 10, "sex_age": "牡3", "d_rest": 30.0})
        for j, (col, gv) in enumerate(ws):
            db = 7 + j
            laps.append({"horse_id": "H", "train_date": (pd.Timestamp(d) - pd.Timedelta(days=db)).strftime("%Y-%m-%d"), "race_date": d, "course": "栗坂", "kind": "slope", "center": "栗", "cond": "良",
                         "rider": "", "position": np.nan, "leg_color": col, "rank": "B", "evaluation": "", "t1": 12.0, "t4": 52.0, "days_before": db, "t4z": -gv, "t1z": -gv, "accelz": 0.0})
    sf = build(pd.DataFrame(laps), pd.DataFrame(runs))["SF"].set_index("race_id")
    assert np.isnan(sf.loc["R0", "wc_sf_hard_vs_best_hard"]) and sf.loc["R0", "wc_sf_n"] == 0
    assert np.isnan(sf.loc["R1", "wc_sf_hard_vs_best_hard"])                                              # 今回に強い追い切りが無い
    assert abs(sf.loc["R2", "wc_sf_hard_vs_best_hard"] - (1.0 - 2.0)) < 1e-12                           # 2 サイクル前の強い追い切りを覚えている
    assert abs(sf.loc["R2", "wc_sf_last_vs_mean"] - (1.0 - np.mean([2.0, 0.5]))) < 1e-12
    assert sf.loc["R2", "wc_sf_is_pr"] == 0.0 and sf.loc["R1", "wc_sf_is_pr"] == 0.0


def test_individual_slope_formula():
    # 同じ馬の 4 走。x1 は最終の良さ、y は着順パーセンタイル。3 走目の補正ポイントは 1〜2 走目だけで決まる
    runs, laps = [], []
    xs, fins = [0.0, 1.0, 2.0, 1.5], [9, 5, 1, 4]
    for i, (x, f) in enumerate(zip(xs, fins)):
        d = pd.Timestamp("2022-01-09") + pd.Timedelta(days=40 * i)
        runs.append({"race_id": f"R{i}", "horse_id": "H", "date": d.strftime("%Y-%m-%d"), "trainer_id": "T", "finish_position": f, "field_size": 10, "sex_age": "牡3", "d_rest": 30.0})
        laps.append({"horse_id": "H", "train_date": (d - pd.Timedelta(days=7)).strftime("%Y-%m-%d"), "race_date": d.strftime("%Y-%m-%d"), "course": "栗坂", "kind": "slope", "center": "栗", "cond": "良", "rider": "",
                     "position": np.nan, "leg_color": "強め", "rank": "B", "evaluation": "", "t1": 12.0, "t4": 52.0, "days_before": 7, "t4z": -x, "t1z": -x, "accelz": 0.0})
    ind = build(pd.DataFrame(laps), pd.DataFrame(runs))["IN"].set_index("race_id")
    y = [1 - (f - 1) / 9 for f in fins]
    assert np.isnan(ind.loc["R0", "wc_in1_cp"]) and np.isnan(ind.loc["R1", "wc_in1_cp"]) and ind.loc["R1", "wc_in1_n"] == 1
    x2, y2 = np.array(xs[:2]), np.array(y[:2])                                                              # R2 の補正ポイント: R0, R1 の 2 点から
    cov = ((x2 - x2.mean()) * (y2 - y2.mean())).sum()
    var = ((x2 - x2.mean()) ** 2).sum()
    beta = cov / (var + C.KAPPA)
    assert abs(ind.loc["R2", "wc_in1_cp"] - beta * (xs[2] - x2.mean())) < 1e-12
    assert abs(ind.loc["R2", "wc_in1_rel"] - var / (var + C.KAPPA)) < 1e-12
