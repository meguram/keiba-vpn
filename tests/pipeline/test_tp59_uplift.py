"""tp-59 上積み特徴量: 先読みなし（その日以降の結果を変えても値が変わらない）、同日の結果を使わない、期待値は 2 走以上から。"""

import numpy as np
import pandas as pd
import pytest

from src.pipeline.features import tp59_uplift as U


def _history(seed=0, n_days=60, n_races_per_day=3, field=8, n_horses=30):
    rng = np.random.default_rng(seed)
    rows = []
    d0 = pd.Timestamp("2022-01-01")
    for d in range(n_days):
        for r in range(n_races_per_day):
            hs = rng.choice(n_horses, size=field, replace=False)
            fin = rng.permutation(field) + 1
            for k, h in enumerate(hs):
                rows.append({"race_id": f"2022{d:03d}{r:02d}", "date": d0 + pd.Timedelta(days=7 * d), "venue_code": "05", "surface": "芝", "distance": 1600,
                             "finish_position": int(fin[k]), "horse_number": k + 1, "horse_id": f"h{h}", "jockey_id": f"j{rng.integers(5)}", "trainer_id": f"t{h % 4}"})
    return pd.DataFrame(rows)


def test_features_do_not_change_when_future_results_change():
    h = _history()
    base = U.build_features(U.prepare_history(h))
    cut = pd.Timestamp("2022-05-01")
    h2 = h.copy()
    m = h2["date"] >= cut
    h2.loc[m, "finish_position"] = h2.loc[m, "finish_position"].sample(frac=1.0, random_state=1).to_numpy()
    alt = U.build_features(U.prepare_history(h2))
    before = (h["date"] <= cut).to_numpy()   # 切断日当日も、同日の結果は使わないので変わらない
    for c in base.columns:
        np.testing.assert_allclose(base.loc[before, c].to_numpy(dtype=float), alt.loc[before, c].to_numpy(dtype=float), err_msg=c, equal_nan=True)
    assert not np.allclose(base.loc[~before, "up_j_res"].to_numpy(dtype=float), alt.loc[~before, "up_j_res"].to_numpy(dtype=float), equal_nan=True)


def test_horse_expectation_needs_two_prior_runs():
    h = U.prepare_history(_history())
    k = h.sort_values(["horse_id", "date", "race_id"]).groupby("horse_id").cumcount()
    srt = h.sort_values(["horse_id", "date", "race_id"])
    first = srt[k < 2]
    assert first["d_res"].isna().all()
    later = srt[k >= 2]
    assert later["d_res"].notna().mean() > 0.95


def test_entity_uplift_is_shrunk_toward_zero_and_first_rows_are_zero():
    h = U.prepare_history(_history())
    F = U.build_features(h)
    first_day = h["date"] == h["date"].min()
    assert (F.loc[first_day, "up_j_res"] == 0).all()
    assert F["up_j_res"].abs().max() < 0.5
