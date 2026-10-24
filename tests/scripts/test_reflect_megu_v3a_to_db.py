import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import reflect_megu_v3a_to_db as R

MODEL = Path("config/megu_v3a")


def test_param_rows_have_new_beta_weight_and_copied_params():
    spec = json.loads((MODEL / "spec.json").read_text())
    centers = json.loads((MODEL / "centers.json").read_text())
    v2 = {"beta_pace": (0.79, 0.017), "beta_track": (0.19, 0.01), "tsi_mean": (3.3, 0.0), "beta_level": (-0.93, 0.0), "r2_adj": (0.94, 0.0)}
    rows = {r["param_name"]: r for r in R.build_param_rows(spec, centers, v2)}
    assert rows["beta_weight"]["param_value"] == 0.05 and rows["v3a_base_kg"]["param_value"] == 55.0
    assert rows["beta_pace"]["param_value"] == 0.79 and "r2_adj" not in rows
    assert rows["v3a_center_abs_turf"]["param_value"] == centers["abs"]["芝"]
    assert all(len(r["param_name"]) <= 50 for r in rows.values())


def test_par_rows_fit_column_widths_and_are_unique():
    rows = R.build_par_rows(pd.read_csv(MODEL / "par.csv"))
    assert rows and all(len(r["track_condition"]) <= 10 and len(r["class_bucket"]) <= 10 and len(r["course"]) <= 20 for r in rows)
    keys = [(r["distance"], r["course"], r["surface"], r["track_condition"], r["class_bucket"]) for r in rows]
    assert len(keys) == len(set(keys))


def test_index_rows_use_abs_index_and_skip_unusable_and_leave_legacy_deltas_null():
    df = pd.DataFrame({
        "race_id": ["r1", "r1", "r2"], "horse_id": ["a", "b", "c"], "finish_time_sec": [60.1, 61.0, 70.0],
        "megu_v3a_status": ["ok", "restored_out_of_range", None],
        "megu_index_v3a_abs_robust": [52.34, -12.06, np.nan], "v3a_par_sec": [60.0, 60.0, np.nan],
        "v3a_corrected_sec": [59.9, 62.0, np.nan], "v3a_track_sec": [0.5, 0.5, np.nan],
    })
    out = R.build_index_rows(df)
    assert list(out["horse_id"]) == ["a", "b"] and list(out["megu_index"]) == [52.3, -12.1]
    assert out["delta_pace_sec"].isna().all() and out["delta_weight_sec"].isna().all() and (out["computation_status"] == "valid").all()
    assert R.check_ranges(out) == []


def test_range_check_detects_overflow():
    rows = pd.DataFrame({"megu_index": [1e6], "par_time_sec": [60.0]})
    assert R.check_ranges(rows)
