"""tp-65: めぐ指数と netkeiba タイム指数の比較（走の表の作成・分散の要約・ロバスト特徴量・先読み）のテスト。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.scripts.data import tp65_feat as F
from src.scripts.data import tp65_prep as P
from src.scripts.data import tp65_var as V


def test_mad_sd_ignores_outliers():
    x = pd.Series([50.0] * 50 + [48.0, 52.0] * 25 + [-800.0])
    assert V.mad_sd(x) < 4.0
    assert x.std() > 50


def test_class_bucket_basic():
    rc = pd.Series(["サラ系２歳 未勝利", "サラ系３歳以上 1勝クラス", "サラ系２歳 新馬", "サラ系３歳以上 オープン", "サラ系３歳以上 2勝クラス"])
    gr = pd.Series(["未勝利", "1勝", "新馬", "G3", "2勝"])
    out = V.class_bucket(rc, gr).tolist()
    assert out == ["未勝利", "1勝", "新馬", "重賞", "2勝"]


def test_huber_resists_outlier_but_follows_level():
    M = np.array([[60.0, 61.0, 59.0, 60.0, -200.0], [70.0, 70.0, 70.0, 70.0, 70.0]])
    h = V.huber_loc(M, 10.0)
    assert 55 <= h[0] <= 62
    assert abs(h[1] - 70) < 1e-6
    assert np.nanmean(M[0]) < 30


def test_feature_variants_handle_missing_and_order():
    M = np.array([[80.0, 70.0, np.nan, np.nan, np.nan], [np.nan] * 5, [50.0, 60.0, 70.0, 80.0, 90.0]])
    f = V.feature_variants(M, 10.0)
    assert f["last1"][0] == 80 and np.isnan(f["last1"][1])
    assert f["max5"][2] == 90 and f["median5"][2] == 70
    assert np.isnan(f["ewm5"][1])
    # ewm は新しい走ほど重い: [50(直近), 60, ...] の加重平均は単純平均 70 より小さい
    assert f["ewm5"][2] < 70
    assert f["trimmed5"][2] == 70


def test_past_matrix_uses_only_previous_rows():
    v = pd.Series([1.0, 2.0, 3.0, 4.0], index=range(4))
    g = pd.Series(["a", "a", "b", "a"])
    M = V.past_matrix(v, g, 2)
    assert np.isnan(M[0]).all()
    assert M[1, 0] == 1.0
    assert np.isnan(M[2]).all()
    assert M[3, 0] == 2.0 and M[3, 1] == 1.0


def test_race_softmax_ll_uniform_when_beta_zero():
    ridx = np.array([0, 0, 0, 1, 1])
    y = np.array([1, 0, 0, 0, 1])
    sc = np.array([1.0, 2.0, 3.0, 0.0, 5.0])
    ll = V.race_softmax_ll(sc, ridx, y, 0.0)
    assert np.allclose(sorted(ll.to_numpy()), sorted([-np.log(3), -np.log(2)]))
    ll1 = V.race_softmax_ll(sc, ridx, y, 1.0)
    assert ll1.loc[1] > ll.loc[1]


def test_race_spearman_sign():
    b = pd.DataFrame({"ridx": [0] * 4, "s": [4.0, 3.0, 2.0, 1.0], "fp": [1, 2, 3, 4]})
    assert V.race_spearman(b).iloc[0] > 0.99


def test_icc_oneway_detects_horse_effect():
    rng = np.random.default_rng(0)
    h = np.repeat(np.arange(300), 5)
    x = pd.Series(rng.normal(0, 10, 300).repeat(5) + rng.normal(0, 5, 1500))
    r = V.icc_oneway(x, pd.Series(h))
    assert 0.7 < r["icc1"] < 0.9
    assert 4 < r["within_sd"] < 6


def _runs():
    d = pd.to_datetime(["2022-01-10", "2022-02-10", "2022-03-10", "2022-04-10", "2022-05-10", "2022-06-10", "2022-07-10"])
    rows = []
    for i, dt in enumerate(d):
        rows.append({"race_id": f"R{i}", "horse_id": "H", "date": dt, "megu_index_v3a": 40.0 + i, "nk": 70.0 + 2 * i, "finish_position": 1})
    return pd.DataFrame(rows)


def test_asof_features_do_not_use_same_day_or_future():
    runs = _runs()
    a = F.asof_features(runs).set_index("race_id")
    assert np.isnan(a.loc["R0", "rb_mg_ewm5"])
    assert a.loc["R1", "rb_mg_median5"] == 40.0
    assert a.loc["R3", "rb_nk_max5"] == 74.0
    assert a.loc["R3", "rb_nk_fl_ewm"] > 70.0
    b = F.asof_features(runs.assign(megu_index_v3a=lambda x: np.where(x["race_id"] == "R6", 999.0, x["megu_index_v3a"]), nk=lambda x: np.where(x["race_id"] == "R6", 999.0, x["nk"]))).set_index("race_id")
    pd.testing.assert_frame_equal(a.loc[["R0", "R1", "R2", "R3", "R4", "R5", "R6"]].drop(columns=[]), b.loc[["R0", "R1", "R2", "R3", "R4", "R5", "R6"]])


def test_asof_features_truncation_exact_equality():
    runs = _runs()
    full = F.asof_features(runs).set_index("race_id")
    cut = F.asof_features(runs, cut="2022-05-01").set_index("race_id")
    pd.testing.assert_frame_equal(full.loc[cut.index], cut)


def test_observations_align_speed_recent_with_past_dates(tmp_path, monkeypatch):
    y = tmp_path / "2024"
    y.mkdir()
    pd.DataFrame({"race_id": ["R9"], "horse_id": ["H"], "date": ["2024-07-21"], "speed_recent": [json.dumps([57, 86, 0])]}).to_parquet(y / "race_index_flat.parquet")
    past = [{"date": "2024.05.04"}, {"date": "2024.02.17"}, {"date": "2023.12.28"}]
    pd.DataFrame({"race_id": ["R9"], "horse_id": ["H"], "past_races": [json.dumps(past)]}).to_parquet(y / "race_shutuba_past_flat.parquet")
    monkeypatch.setattr(P, "TABLES", tmp_path)
    o = P.observations([2024])
    assert o.loc[o["lag"] == 0, "run_date"].iloc[0] == "2024-05-04" and o.loc[o["lag"] == 0, "val"].iloc[0] == 57.0
    assert o.loc[o["lag"] == 1, "run_date"].iloc[0] == "2024-02-17" and o.loc[o["lag"] == 1, "val"].iloc[0] == 86.0
    # 0 は欠損として build 側で NaN にされる（観察の段階では値のまま残す）
    assert o.loc[o["lag"] == 2, "val"].iloc[0] == 0.0
