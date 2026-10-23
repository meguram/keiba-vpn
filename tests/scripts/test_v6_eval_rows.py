import numpy as np
import pandas as pd

from src.pipeline.models.time_split import Fold
from src.scripts.data.v6_eval import _restrict_folds, _row_mask


def test_row_mask_and_restrict_folds():
    df = pd.DataFrame({"surface": ["芝", "ダート", "障", "芝", "障", "ダート"]})
    assert _row_mask(df, "all") is None
    assert _row_mask(df, "hurdle").tolist() == [False, False, True, False, True, False]
    assert _row_mask(df, "flat").tolist() == [True, True, False, True, False, True]
    ts = pd.Timestamp("2022-01-01")
    f = Fold("f", "2022", np.array([0, 1, 2]), np.array([3, 4, 5]), ts, ts)
    flat = _restrict_folds([f], _row_mask(df, "flat"))[0]
    assert flat.train_index.tolist() == [0, 1] and flat.valid_index.tolist() == [3, 5]
    hur = _restrict_folds([f], _row_mask(df, "hurdle"))[0]
    assert hur.train_index.tolist() == [2] and hur.valid_index.tolist() == [4]
    assert _restrict_folds([f], np.zeros(6, dtype=bool)) == []        # 検証行が無い fold は落とす
