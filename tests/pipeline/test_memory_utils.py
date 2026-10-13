import numpy as np
import pandas as pd
import pytest

from src.pipeline.models.memory_utils import batched_predict, downcast_frame, frame_mb, read_parquet_columns


def _df(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "race_id": [f"2024{i:08d}" for i in range(n)],
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "cnt": rng.integers(0, 100, size=n).astype(np.int64),
            "venue": rng.choice(["東京", "中山", "京都"], size=n),
            "past": [[1, 2]] * n,
            "flag": rng.integers(0, 2, size=n).astype(bool),
        }
    )


def test_downcast_shrinks_and_keeps_protected_columns():
    df = _df()
    out = downcast_frame(df, keep=["race_id"])
    assert out["x1"].dtype == np.float32
    assert out["cnt"].dtype == np.int8
    assert out["venue"].dtype.name == "category"
    assert out["race_id"].dtype == df["race_id"].dtype  # keep 列は変えない（pandas 2 は object / 3 は str）
    assert out["past"].dtype == object  # リスト列は触らない
    assert out["flag"].dtype == bool
    assert frame_mb(out) < frame_mb(df)
    assert np.allclose(out["x1"], df["x1"], rtol=1e-6)
    assert (out["cnt"] == df["cnt"]).all()


def test_float32_does_not_change_lightgbm_predictions_materially():
    lgb = pytest.importorskip("lightgbm")
    df = _df(20000)
    y = ((df["x1"] + 0.5 * df["x2"] + np.random.default_rng(1).normal(scale=0.5, size=len(df))) > 0).astype(int)
    feats = ["x1", "x2", "cnt"]
    params = dict(n_estimators=100, learning_rate=0.1, num_leaves=31, random_state=0, verbose=-1, deterministic=True, n_jobs=1)
    p64 = lgb.LGBMClassifier(**params).fit(df[feats], y).predict_proba(df[feats])[:, 1]
    d32 = downcast_frame(df[feats])
    p32 = lgb.LGBMClassifier(**params).fit(d32, y).predict_proba(d32)[:, 1]
    assert np.abs(p64 - p32).max() < 1e-2


def test_batched_predict_equals_full_predict():
    df = _df(1234)

    def f(x):
        return (x["x1"] * 2 + x["x2"]).to_numpy()

    assert np.array_equal(batched_predict(f, df, batch_size=100), f(df))


def test_read_parquet_columns_projects_and_filters(tmp_path):
    pl = pytest.importorskip("polars")
    df = _df(300)
    p = tmp_path / "a.parquet"
    df[["race_id", "x1", "x2", "cnt", "venue"]].to_parquet(p, index=False)
    out = read_parquet_columns(p, ["race_id", "x1"], filter_expr=pl.col("cnt") < 50)
    assert list(out.columns) == ["race_id", "x1"]
    assert len(out) == int((df["cnt"] < 50).sum())
    assert out["x1"].dtype == np.float32
