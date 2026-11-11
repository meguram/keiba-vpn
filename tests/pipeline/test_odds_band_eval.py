import numpy as np
import pandas as pd

from src.scripts.data import odds_band_eval as ob


def _frame(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for year in (2022, 2023, 2024, 2025):
        for r in range(150):
            pred = np.sort(rng.uniform(1.5, 40, 12))
            act = pred * np.exp(rng.normal(0.05, 0.3, 12))
            for i in range(12):
                rows.append({"year": year, "race_id": f"{year}{r:04d}", "pred": pred[i], "win_odds": act[i]})
    return pd.DataFrame(rows)


def test_interval_coverage_near_80pct():
    d = _frame()
    out = ob.interval_and_rank(d, 2025, (2022, 2024))
    assert 0.7 < out["coverage_overall"] < 0.9
    assert 0 <= out["fav_is_pred_top1"] <= 1


def test_mode_shift_does_not_use_eval_year_and_not_worse():
    d = _frame()
    out = ob.mode_shift(d, 2025, (2022, 2024))
    assert 0.0 <= out["k"] <= 2.0
    base = ((d[d.year == 2025]["pred"] - d[d.year == 2025]["win_odds"]).abs() <= 1.0)
    o = d[d.year == 2025]["win_odds"]
    assert out["le20_all"] >= base[o <= 20].mean() - 0.02


def test_eval_year_changes_do_not_move_fitted_values():
    d = _frame()
    a = ob.mode_shift(d, 2025, (2022, 2024))
    d2 = d.copy()
    d2.loc[d2.year == 2025, "win_odds"] *= 3.0
    b = ob.mode_shift(d2, 2025, (2022, 2024))
    assert a["k"] == b["k"] and a["sigma_by_band"] == b["sigma_by_band"]


def test_tied_favorites_both_count():
    d = pd.DataFrame({"year": [2025] * 4, "race_id": ["r"] * 4, "pred": [2.0, 2.8, 5.0, 9.0], "win_odds": [2.8, 2.8, 5.0, 9.0]})
    tr = pd.DataFrame({"year": [2024] * 4, "race_id": ["q"] * 4, "pred": [2.0, 2.8, 5.0, 9.0], "win_odds": [2.1, 2.9, 5.1, 9.2]})
    out = ob.interval_and_rank(pd.concat([tr, d]), 2025, (2024, 2024))
    assert out["fav_in_pred_top3"] == 1.0
