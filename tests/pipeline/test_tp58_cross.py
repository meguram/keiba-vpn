import numpy as np
import pandas as pd

from src.pipeline.features.tp58_cross import asof_sum_count, shrunk_cross


def _frame(seed=0, n=400):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "date": pd.to_datetime("2022-01-01") + pd.to_timedelta(rng.integers(0, 60, n), unit="D"),
        "j": rng.integers(0, 5, n).astype(str),
        "c": rng.integers(0, 3, n).astype(str),
        "y": rng.integers(0, 2, n).astype(float),
    })


def test_asof_matches_brute_force_and_excludes_same_day():
    df = _frame()
    date = df["date"].to_numpy("datetime64[D]").astype("int64")
    keys = pd.factorize(df["j"] + df["c"])[0]
    valid = np.ones(len(df), bool)
    sy, sc = asof_sum_count(keys, date, df["y"].to_numpy(), valid)
    for i in range(len(df)):
        prev = (keys == keys[i]) & (date < date[i])
        assert sc[i] == prev.sum()
        assert sy[i] == df["y"].to_numpy()[prev].sum()


def test_invalid_rows_not_aggregated():
    df = _frame(1)
    date = df["date"].to_numpy("datetime64[D]").astype("int64")
    keys = pd.factorize(df["j"])[0]
    valid = (np.arange(len(df)) % 3) != 0
    sy, sc = asof_sum_count(keys, date, df["y"].to_numpy(), valid)
    for i in range(0, len(df), 7):
        prev = (keys == keys[i]) & (date < date[i]) & valid
        assert sc[i] == prev.sum() and sy[i] == df["y"].to_numpy()[prev].sum()


def test_shrunk_cross_exact_and_lookahead_free():
    df = _frame(2)
    y = df["y"].to_numpy()
    valid = np.ones(len(df), bool)
    out = shrunk_cross(df, ["j", "c"], ["j"], y, valid, m=5.0, m_parent=10.0)
    d = df["date"].to_numpy("datetime64[D]").astype("int64")
    for i in range(0, len(df), 11):
        prev_all = d < d[i]
        root = y[prev_all].mean() if prev_all.any() else 0.1
        pj = prev_all & (df["j"].to_numpy() == df["j"].iloc[i])
        parent = (y[pj].sum() + 10 * root) / (pj.sum() + 10) if not np.isnan(root) else np.nan
        pc = pj & (df["c"].to_numpy() == df["c"].iloc[i])
        child = (y[pc].sum() + 5 * parent) / (pc.sum() + 5)
        if not np.isnan(child):
            assert abs(out["m"][i] - child) < 1e-9
            assert abs(out["dev"][i] - (child - parent)) < 1e-9
    # 将来の y を書き換えても過去行の値は変わらない
    y2 = y.copy()
    y2[d > np.median(d)] = 1 - y2[d > np.median(d)]
    out2 = shrunk_cross(df, ["j", "c"], ["j"], y2, valid, m=5.0, m_parent=10.0)
    early = d <= np.median(d)
    np.testing.assert_allclose(out["m"][early], out2["m"][early], atol=1e-12)
