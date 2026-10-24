"""暦の特徴量の介入実験（drift_calendar_experiment）の純関数のテスト（合成データ）。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.scripts.data import drift_calendar_experiment as C


def test_train_end_year_frac_reads_year_from_bundle_name():
    assert C.train_end_year_frac(Path("x/bundle_v5_2024")) == pytest.approx(2024 + 364 / 365.25)
    assert C.train_end_year_frac(Path("bundle_v5_2025")) > C.train_end_year_frac(Path("bundle_v5_2024"))
    with pytest.raises(ValueError):
        C.train_end_year_frac(Path("bundle_without_year"))


def test_scenario_values_use_training_mean_of_linear_prep():
    cols = C.CAL_ALL + ["other"]
    prep = SimpleNamespace(num_cols=cols, mu=np.arange(len(cols), dtype=float) + 100.0)
    v = C.scenario_values(prep, 2025.0)
    assert v["a_current"] == {}
    assert v["b_yf_to_train_max"] == {"fe_year_frac": 2025.0}
    assert v["c_yf_to_train_mean"]["fe_year_frac"] == 100.0
    assert set(v["d_all_calendar_to_mean"]) == set(C.CAL_ALL)
    assert set(v["e_season_to_mean_keep_yf"]) == set(C.CAL_SEASON)          # 季節性だけ。トレンド列は現状のまま
    assert "fe_year_frac" not in v["e_season_to_mean_keep_yf"]


def test_race_bootstrap_interval_contains_mean_and_is_ordered():
    rng = np.random.default_rng(0)
    race = np.repeat(np.arange(60), 10)
    delta = rng.normal(0.002, 0.01, len(race))
    m, lo, hi = C.race_bootstrap(race, delta, n=400)
    assert lo <= m <= hi
    z, zl, zh = C.race_bootstrap(race, np.zeros(len(race)), n=200)
    assert (z, zl, zh) == (0.0, 0.0, 0.0)


def test_bin_ll_matches_definition_and_logit_is_clipped():
    y = np.array([1.0, 0.0])
    p = np.array([0.25, 0.25])
    assert C._bin_ll(y, p)[0] == pytest.approx(-np.log(0.25))
    assert C._bin_ll(y, p)[1] == pytest.approx(-np.log(0.75))
    assert np.isfinite(C._logit(np.array([0.0, 1.0]))).all()
