import numpy as np
import pytest

from src.scripts.data import calibrate_ev as ce


def test_pava_pools_violations():
    x, y = ce.pava(np.array([1, 2, 3, 4, 5.0]), np.array([1, 3, 2, 4, 5.0]))
    assert list(x) == [1.0, 2.5, 4.0, 5.0] and list(y) == [1.0, 2.5, 4.0, 5.0]
    x, y = ce.pava(np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0]))
    assert len(y) == 1 and y[0] == pytest.approx(2.0)


def test_fit_isotonic_is_monotone_and_not_extrapolated():
    rng = np.random.default_rng(0)
    ev = rng.uniform(0.2, 3.0, 20000)
    ret = rng.exponential(0.5 + 0.2 * ev)          # 真の期待回収率は 0.5 + 0.2×EV
    spec = ce.fit_isotonic_knots(ev, ret)
    assert np.all(np.diff(spec["y"]) >= -1e-9) and np.all(np.diff(spec["x"]) > 0)
    lo, hi = ce.apply_spec(spec, 0.0), ce.apply_spec(spec, 100.0)
    assert lo == pytest.approx(spec["y"][0]) and hi == pytest.approx(spec["y"][-1])
    assert abs(ce.apply_spec(spec, 1.5) - (0.5 + 0.2 * 1.5)) < 0.08


def test_linear_shrink_const_fits():
    ev = np.linspace(0.5, 2.5, 1000)
    ret = 0.7 + 0.1 * ev
    lin = ce.fit_linear(ev, ret)
    assert lin["b"] == pytest.approx(0.1, abs=1e-3) and lin["a"] == pytest.approx(0.7, abs=1e-3)
    sh = ce.fit_shrink(ev, 0.8 + 0.3 * (ev - 0.8), 0.8)
    assert sh["k"] == pytest.approx(0.3, abs=1e-3)
    assert ce.fit_const(np.array([0.5, 1.5, 1.0]))["value"] == pytest.approx(1.0)
    assert ce.fit_linear(ev, 2.0 - ev)["b"] == 0.0                         # 負の傾きは 0 に丸める（EV が高いほど悪いと表示しない）


def test_calib_metrics_detects_overclaiming_and_accepts_calibrated():
    rng = np.random.default_rng(1)
    n = 6000
    race = np.repeat(np.arange(n // 3), 3)
    true = rng.uniform(0.6, 1.2, n)
    ret = rng.exponential(true)
    good = ce.calib_metrics(true, ret, race)
    assert abs(good["gap"]) < 0.05 and good["ece"] < 0.1
    over = ce.calib_metrics(true + 0.8, ret, race)
    assert over["gap"] == pytest.approx(0.8, abs=0.05) and over["ece"] > 0.7


def _wf(over_by_year: dict[int, float]):
    import pandas as pd

    rng = np.random.default_rng(2)
    rows = []
    for y, shift in over_by_year.items():
        ev = rng.uniform(0.5, 2.0, 4000)
        rows.append(pd.DataFrame({"year": y, "race_id": [f"{y}{i // 4}" for i in range(4000)], "ev": ev, "ret": rng.exponential(0.8 + 0.1 * ev + shift)}))
    return pd.concat(rows, ignore_index=True)


def test_walk_forward_uses_only_earlier_years_and_calibrated_beats_raw():
    df = _wf({2022: 0, 2023: 0, 2024: 0, 2025: 0})
    res = ce.walk_forward(df, "ev", "ret", 0.8, years=(2023, 2024, 2025))
    assert res["raw"]["pooled"]["gap"] > 0.3
    for m in ("shrink", "linear", "isotonic", "const"):
        assert abs(res[m]["pooled"]["gap"]) < 0.08
    only_one_year = ce.walk_forward(df[df["year"] == 2022], "ev", "ret", 0.8, years=(2022,))
    assert "2022" not in only_one_year["linear"]                           # 当てはめる過去の年が無ければ評価しない


def test_choose_method_prefers_passing_nonconstant_and_falls_back_to_const():
    df = _wf({2022: 0, 2023: 0, 2024: 0, 2025: 0})
    res = ce.walk_forward(df, "ev", "ret", 0.8)
    m, info = ce.choose_method(res)
    assert m in ("shrink", "linear", "isotonic") and info["ok"]
    drift = ce.walk_forward(_wf({2022: 0, 2023: 0, 2024: 0.5, 2025: 0.9}), "ev", "ret", 0.8)
    m2, info2 = ce.choose_method(drift)
    assert m2 == "const" and info2["ok"] is False                           # 最近の年で急にずれる（基準を満たさない）なら、収束すると言わない
