"""v4 の推論用（serving_v4）: 当日専用の高速な as-of が学習側の asof_group_mean と同じ値になること、列の一覧と必要な群の判定。"""

import numpy as np
import pandas as pd

from src.pipeline.features import serving_v4 as sv
from src.pipeline.models.stage_models import asof_group_mean


def _toy(n_days: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        date = pd.Timestamp("2024-01-06") + pd.Timedelta(days=7 * d)
        for r in range(6):
            rows.append({"date": date, "k": f"k{int(rng.integers(0, 5))}", "v": float(rng.random()) if rng.random() > 0.1 else np.nan})
    return pd.DataFrame(rows)


def test_fast_asof_equals_training_function_for_the_last_day():
    df = _toy()
    day = df["date"].max()
    fast = sv.make_fast_asof(day)(df, "k", "v", k=7.0, name="x")
    ref = asof_group_mean(df, "k", "v", k=7.0, name="x")
    sel = (df["date"] >= day).to_numpy()
    np.testing.assert_allclose(fast.loc[sel, "x_mean"].to_numpy(), ref.loc[sel, "x_mean"].to_numpy(), rtol=1e-5)
    np.testing.assert_allclose(fast.loc[sel, "x_n"].to_numpy(), ref.loc[sel, "x_n"].to_numpy())


def test_fast_asof_ignores_results_of_the_day_itself():
    df = _toy()
    day = df["date"].max()
    a = sv.make_fast_asof(day)(df, "k", "v", k=7.0, name="x")
    df2 = df.copy()
    df2.loc[df2["date"] >= day, "v"] = 99.0                   # 当日の結果を変えても、当日の行の値は変わらない
    b = sv.make_fast_asof(day)(df2, "k", "v", k=7.0, name="x")
    sel = (df["date"] >= day).to_numpy()
    np.testing.assert_allclose(a.loc[sel].to_numpy(), b.loc[sel].to_numpy())


def test_context_manager_restores_original_function():
    from src.pipeline.features import factory_field as ff
    from src.pipeline.models import stage_models as sm

    o1, o2 = sm.asof_group_mean, ff.asof_group_mean
    with sv.fast_asof(pd.Timestamp("2024-06-01")):
        assert sm.asof_group_mean is not o1
    assert sm.asof_group_mean is o1 and ff.asof_group_mean is o2


def test_group_columns_cover_the_adopted_groups_and_pick_needed_groups():
    cols = sv.group_columns()
    assert set(cols) == {"owner", "self8", "field7", "ped"}
    assert (len(cols["owner"]), len(cols["self8"]), len(cols["field7"]), len(cols["ped"])) == (125, 199, 451, 32)
    assert sv.needed_groups(["o_win", "foo"]) == {"owner"}
    assert sv.needed_groups(["foo", "bar"]) == set()
