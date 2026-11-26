"""学習と推論の X の欠けの率のふるい（skew_sieve）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.scripts.data import skew_sieve as ss


def test_group_of():
    assert ss.group_of("tl_w1_t1") == "tl_w1" and ss.group_of("tl_f_pos") == "tl" and ss.group_of("wc_cc_gap") == "wc"
    assert ss.group_of("fe_prev_day_going") == "prev_day" and ss.group_of("weight") == "weight" and ss.group_of("d_wt_dev") == "weight"


def test_stratum_jump_needs_blank_class_and_jump_surface():
    k = ss.stratum(pd.Series(["サラ系２歳 新馬", "サラ系２歳 未勝利", "", "", "サラ系３歳以上 １勝クラス"]),
                   pd.Series(["芝", "ダート", "障", "芝", "芝"]))
    assert k.tolist() == ["shinba", "misho", "jump", "other", "other"]


def test_compare_weights_by_served_strata(tmp_path):
    n = 8
    meta = pd.DataFrame({"race_id": [f"r{i}" for i in range(n)], "date": ["2025-11-01"] * 6 + ["2023-11-01"] * 2,
                         "race_class": ["新馬", "新馬", "未勝利", "未勝利", "未勝利", "未勝利", "新馬", "未勝利"]})
    # 学習: 新馬は a が埋まる・未勝利は a が半分空。2023 年の行（窓の外）は全部空で、数えないこと
    X = pd.DataFrame({"a": [1.0, 2.0, np.nan, 1.0, np.nan, 1.0, np.nan, np.nan], "b": [1.0] * n})
    X.to_parquet(tmp_path / "X.parquet")
    served = pd.DataFrame({"race_id": ["s"] * 4, "horse_id": list("wxyz"), "race_class": ["新馬", "未勝利", "未勝利", "未勝利"],
                           "a": [1.0, np.nan, np.nan, np.nan], "b": [1.0, 1.0, 1.0, 1.0]})
    o = ss.compare(served, tmp_path / "X.parquet", meta, [2025], [11]).set_index("col")
    assert o.loc["a", "na_train_shinba"] == 0.0 and o.loc["a", "na_train_misho"] == 0.5
    assert o.loc["a", "na_served_misho"] == 1.0
    assert abs(o.loc["a", "na_train"] - 0.375) < 1e-9          # 0.25×0 + 0.75×0.5
    assert abs(o.loc["a", "d_pt"] - 37.5) < 1e-9
    assert o.loc["b", "d_pt"] == 0.0
    s = ss.summary(o.reset_index())
    assert s["over"] == 1 and s["by_group"] == {"other": {"n": 1, "max_d_pt": 37.5}}      # 接頭辞の無い列は other


def _train(tmp_path, X, dates, classes):
    X.to_parquet(tmp_path / "X.parquet")
    return pd.DataFrame({"race_id": [f"r{i}" for i in range(len(X))], "date": dates, "race_class": classes})


def test_blank_strings_count_as_missing_and_one_sided_columns_are_listed(tmp_path):
    meta = _train(tmp_path, pd.DataFrame({"going": ["良", "稍"], "only_train": [1.0, 2.0]}), ["2025-11-01"] * 2, ["未勝利"] * 2)
    served = pd.DataFrame({"race_id": ["s", "s"], "horse_id": ["a", "b"], "race_class": ["未勝利"] * 2,
                           "going": ["", "None"], "only_served": [1.0, np.nan]})
    o = ss.compare(served, tmp_path / "X.parquet", meta, [2025], [11]).set_index("col")
    assert o.loc["going", "na_served"] == 1.0 and o.loc["going", "na_train"] == 0.0
    assert o.loc["only_train", "side"] == "train_only" and o.loc["only_train", "na_served"] == 1.0
    assert o.loc["only_served", "side"] == "served_only"
    s = ss.summary(o.reset_index())
    assert s["train_only"] == ["only_train"] and s["served_only"] == ["only_served", "race_class"] and s["cols"] == 1


def test_row_count_mismatch_is_an_error(tmp_path):
    meta = _train(tmp_path, pd.DataFrame({"a": [1.0, 2.0]}), ["2025-11-01"] * 2, ["未勝利"] * 2).iloc[:1]
    served = pd.DataFrame({"race_id": ["s"], "horse_id": ["a"], "race_class": ["未勝利"], "a": [1.0]})
    with pytest.raises(ValueError):
        ss.compare(served, tmp_path / "X.parquet", meta, [2025], [11])


def test_stratum_absent_in_train_does_not_count_as_skew(tmp_path):
    meta = _train(tmp_path, pd.DataFrame({"a": [1.0, 1.0]}), ["2025-11-01"] * 2, ["未勝利"] * 2)
    served = pd.DataFrame({"race_id": ["s", "s"], "horse_id": ["a", "b"], "race_class": ["未勝利", "新馬"], "a": [1.0, np.nan]})
    o = ss.compare(served, tmp_path / "X.parquet", meta, [2025], [11]).set_index("col")
    assert np.isnan(o.loc["a", "na_train_shinba"]) and o.loc["a", "d_pt"] == 0.0


def test_months_forms():
    assert ss._months("10-12") == [10, 11, 12] and ss._months("10,12") == [10, 12] and ss._months("1-2,11") == [1, 2, 11]
