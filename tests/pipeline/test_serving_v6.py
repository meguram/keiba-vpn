import numpy as np
import pandas as pd

from src.pipeline.features import serving_v6 as sv
from src.scripts.data import tp65_feat as tf


def _runs(seed=0, n_horses=60):
    rng = np.random.default_rng(seed)
    rows = []
    rid = 0
    dates = pd.date_range("2020-01-05", "2024-12-28", freq="7D")
    for h in range(n_horses):
        ds = rng.choice(dates, size=rng.integers(3, 14), replace=False)
        for d in sorted(ds):
            rows.append((f"r{rid}", f"h{h}", pd.Timestamp(d), np.nan, np.nan))
            rid += 1
    df = pd.DataFrame(rows, columns=["race_id", "horse_id", "date", "megu_index_v3a", "nk"])
    # 同じ日の走を同じレースにまとめる（日付ごとに 6 頭ずつ）
    df["race_id"] = df["date"].dt.strftime("%Y%m%d") + "_" + (df.groupby("date").cumcount() // 6).astype(str)
    df = df.drop_duplicates(["race_id", "horse_id"])
    df["finish_position"] = df.groupby("race_id").cumcount() + 1
    df["finish_position"] = df.groupby("race_id")["finish_position"].transform(lambda s: rng.permutation(s.to_numpy()))
    df["megu_index_v3a"] = rng.normal(50, 10, len(df)) + 8 * (df["finish_position"] == 1)
    df["nk"] = rng.normal(60, 12, len(df)) + 8 * (df["finish_position"] == 1)
    df.loc[rng.random(len(df)) < 0.15, "megu_index_v3a"] = np.nan
    df.loc[rng.random(len(df)) < 0.25, "nk"] = np.nan
    return df.reset_index(drop=True)


def test_rb_matches_training_asof_features():
    runs = _runs()
    ref = tf.asof_features(runs)
    params = sv.rb_params(runs)
    hist = sv.build_hist(runs, params)
    got = sv.rb_features(hist, runs[["race_id", "horse_id", "date"]], params)
    m = ref.merge(got, on=["race_id", "horse_id"], suffixes=("_t", "_s"))
    assert len(m) == len(runs)
    for c in sv.COLUMNS:
        a, b = m[f"{c}_t"].to_numpy(float), m[f"{c}_s"].to_numpy(float)
        assert np.array_equal(np.isnan(a), np.isnan(b)), c
        assert np.allclose(a[~np.isnan(a)], b[~np.isnan(b)], atol=1e-4, rtol=1e-5), c


def test_rb_does_not_use_future_or_same_day_runs():
    runs = _runs(1)
    params = sv.rb_params(runs)
    hist = sv.build_hist(runs, params)
    t = runs[["race_id", "horse_id", "date"]].iloc[:50]
    a = sv.rb_features(hist, t, params)
    cut = pd.Timestamp("2022-06-01")
    # 切断日以降の履歴を消しても、切断日より前の行は変わらない
    b = sv.rb_features(hist[hist["date"] < cut], t[t["date"] < cut], params)
    aa = a.merge(b, on=["race_id", "horse_id"], suffixes=("_a", "_b"))
    for c in sv.COLUMNS:
        x, y = aa[f"{c}_a"].to_numpy(float), aa[f"{c}_b"].to_numpy(float)
        assert np.allclose(x, y, equal_nan=True), c
