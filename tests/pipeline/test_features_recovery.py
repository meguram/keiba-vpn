"""回収率・クロス特徴量のユニットテスト。"""

import pandas as pd

from src.pipeline.features.cross_features import add_cross_features, running_style_label
from src.pipeline.recovery import calculate_recovery_rate


def test_recovery_rate_precision():
    r = calculate_recovery_rate(0.1, 10.0, 0.3, 2.0)
    assert r["win_roi"] == 100.0
    assert r["show_roi"] == 60.0


def test_cross_features_pace_prior():
    df = pd.DataFrame({
        "race_id": ["r1", "r1", "r1"],
        "running_style_score": [-4, -3, 2],
        "field_size": [3, 3, 3],
        "speed_index_avg": [100, 110, 90],
        "days_since_last": [10, 20, 30],
    })
    out = add_cross_features(df)
    assert "front_runner_count" in out.columns
    assert out["front_runner_count"].iloc[0] >= 1
    assert running_style_label(-4) == "FRONT"
