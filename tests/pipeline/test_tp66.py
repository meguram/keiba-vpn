import numpy as np
import pandas as pd

from src.scripts.data import tp66_lib as L


def _runs(horse, rows):
    return pd.DataFrame(rows, columns=["date", "idx", "surf", "distance", "ok"]).assign(
        horse_id=horse, race_id=lambda d: [f"2024{i:08d}" for i in range(len(d))], year=2024)


def test_features_use_only_prior_runs():
    df = _runs("h", [("2024-01-01", 50, "芝", 1600, True), ("2024-02-01", 60, "芝", 1600, True),
                     ("2024-03-01", 70, "ダ", 1200, True), ("2024-04-01", 99, "芝", 1600, True)])
    df["date"] = pd.to_datetime(df["date"])
    f = L.compute_features(df, {}, min_year=2024)
    assert f["last1"].isna().iloc[0]                       # 初出走は履歴なし
    assert f["last1"].iloc[1] == 50 and f["max5"].iloc[2] == 60
    assert f["max5"].iloc[3] == 70                         # 今走の 99 は混ざらない


def test_same_day_not_used_and_low_confidence_skipped():
    df = _runs("h", [("2024-01-01", 50, "芝", 1600, True), ("2024-02-01", 80, "芝", 1600, False),
                     ("2024-03-01", 55, "芝", 1600, True), ("2024-03-01", 90, "芝", 1600, True)])
    df["date"] = pd.to_datetime(df["date"])
    f = L.compute_features(df, {}, min_year=2024)
    assert f["last1"].iloc[2] == 50                        # 低信頼(ok=False)の 80 は使わない
    assert f["last1"].iloc[3] == 50                        # 同じ日の走は使わない


def test_cond_and_margin_features():
    df = _runs("h", [("2024-01-01", 70, "芝", 1600, True), ("2024-02-01", 40, "ダ", 1200, True), ("2024-03-01", 50, "芝", 2000, True)])
    df["date"] = pd.to_datetime(df["date"])
    df.loc[2, "distance"] = 1700
    f = L.compute_features(df, {df.race_id.iloc[2]: 60.0}, min_year=2024)
    r = f.iloc[2]
    assert r["cond_max"] == 70 and r["cond_n"] == 1        # 同芝・距離差 100m の走だけ
    assert r["max5_margin"] == 10 and r["n_above5"] == 1


def test_dist_tol_and_winsor():
    assert L.dist_tol(2400) == 400 and L.dist_tol(1600) == 200
    assert L.winsor_mean(np.array([50.0, 51.0, 52.0, 120.0])) < np.mean([50, 51, 52, 120])


def test_mark_top_k_and_logit_fit():
    df = pd.DataFrame({"race_id": ["a"] * 6, "horse_id": list("abcdef"), "s": [5, 3, np.nan, 9, 1, 2]})
    m = L.mark_top_k(df, "s", 2)
    assert list(df.horse_id[m]) == ["d", "a"] or set(df.horse_id[m]) == {"a", "d"}
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 1))
    y = (rng.random(4000) < L.sigmoid(0.5 + 1.2 * X[:, 0])).astype(float)
    w = L.fit_logit(X, y)
    assert abs(w[1] - 1.2) < 0.2


def test_floor_and_weighted_huber():
    from src.pipeline.inference import megu_mark as M

    assert M.floored(10.0, 80.0) == 40.0 and M.floored(70.0, 80.0) == 70.0 and M.floored(5.0, None) == 5.0
    v = np.array([60.0, 62.0, 61.0, 20.0, 59.0])
    w = M.DECAY ** np.arange(5)
    assert M.weighted_huber(v, w) > float((v * w).sum() / w.sum())     # 外れ値(20)の影響を減らす


def test_floored_history_used():
    df = _runs("h", [("2024-01-01", 60, "芝", 1600, True), ("2024-02-01", 10, "芝", 1600, True), ("2024-03-01", 55, "芝", 1600, True)])
    df["date"] = pd.to_datetime(df["date"])
    df["idx_fl"] = [60.0, 40.0, 55.0]
    f = L.compute_features(df, {}, min_year=2024)
    assert f["ew5_fl"].iloc[2] > f["ew5"].iloc[2]


def test_v3b_feature_asof_and_low_weight():
    df = _runs("h", [("2024-01-01", 60, "芝", 1600, True), ("2024-02-01", 50, "芝", 1600, True), ("2024-03-01", 99, "芝", 1600, True)])
    df["date"] = pd.to_datetime(df["date"])
    df["valid"] = True
    df["low"] = [False, True, False]
    df["idx_fl"] = df["idx"].astype(float)
    f = L.v3b_feature(df)
    assert np.isnan(f.iloc[0])                       # 初出走は履歴なし
    assert f.iloc[1] == 60.0                         # 今走の 50 は使わない
    assert 50 < f.iloc[2] < 60                       # 3 走目の値は直前 2 走だけ（99 は混ざらない）


def test_mark_ties_broken_by_horse_number_not_row_order():
    df = pd.DataFrame({"race_id": ["a"] * 4, "horse_id": list("abcd"), "horse_number": [4, 3, 2, 1], "s": [2.0, 2.0, 2.0, 1.0]})
    m1 = L.mark_top_k(df, "s", 1, min_runners=1)
    m2 = L.mark_top_k(df.iloc[::-1].reset_index(drop=True), "s", 1, min_runners=1)
    assert list(df.horse_id[m1]) == ["c"] and list(df.iloc[::-1].reset_index(drop=True).horse_id[m2]) == ["c"]   # 同点は馬番が小さい馬
