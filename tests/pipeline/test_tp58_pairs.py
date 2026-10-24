import numpy as np
import pandas as pd

from src.pipeline.features.tp58_pairs import column_name, fit_constants, pair_column


def _df():
    rng = np.random.default_rng(0)
    n = 60
    return pd.DataFrame({
        "race_id": np.repeat(np.arange(6), 10).astype(str),
        "a": rng.normal(5, 2, n), "b": rng.uniform(1, 9, n), "c": rng.normal(0, 1, n),
        "year": np.repeat([2020, 2020, 2021, 2022, 2022, 2023], 10),
    })


def test_constants_use_only_train_rows():
    df = _df()
    m = (df["year"] <= 2021).to_numpy()
    c1 = fit_constants(df, ["a", "rel(a)", "b"], m)
    df2 = df.copy()
    df2.loc[~m, "a"] += 1000.0
    c2 = fit_constants(df2, ["a", "b"], m)
    assert c1["a"] == c2["a"] and set(c1) == {"a", "b"}


def test_prod_diff_and_relative_definition():
    df = _df()
    c = fit_constants(df, ["a", "b"], (df["year"] <= 2021).to_numpy())
    za = np.clip((df["a"] - c["a"]["mu"]) / c["a"]["sd"], -5, 5)
    zb = np.clip((df["b"] - c["b"]["mu"]) / c["b"]["sd"], -5, 5)
    np.testing.assert_allclose(pair_column(df, "prod", "a", "b", c), za * zb)
    np.testing.assert_allclose(pair_column(df, "diff", "a", "b", c), za - zb)
    rel = pair_column(df, "diff", "rel(a)", "b", c)
    za_rel = za - za.groupby(df["race_id"]).transform("mean")
    np.testing.assert_allclose(rel, za_rel - zb)
    # レース内偏差は同じレースの中で平均 0
    r = pair_column(df, "prod", "rel(a)", "rel(b)", c)
    assert np.isfinite(r).all()


def test_logratio_and_name():
    df = _df()
    c = fit_constants(df, ["a", "b"], np.ones(len(df), bool))
    out = pair_column(df, "logratio", "a", "b", c)
    assert out.shape == (60,) and np.isfinite(out).all()
    assert column_name("prod", "x", "rel(y)") == "x58_prod__x__rel(y)"
