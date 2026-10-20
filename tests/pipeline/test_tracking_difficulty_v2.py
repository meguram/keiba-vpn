"""追走難度 v2（tp-54）: 先読みなしの確認（as-of の集計が対象の行・未来の行を含まない）、特徴量の列、分位への変換、フラグ既定オフ。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.models import tracking_difficulty_v2 as td2
from src.scripts.data import tp54_dataset as ds
from src.scripts.data import tp54_pace as pc


def _frame() -> pd.DataFrame:
    rows = []
    for hid, vals in {"a": [0.5, -0.2, 1.0, 0.3, -0.7], "b": [0.1, 0.4, -0.3, 0.9, 0.2]}.items():
        for i, v in enumerate(vals):
            rows.append({"horse_id": hid, "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=7 * i), "v": v, "early": 0.2 + 0.15 * i})
    return pd.DataFrame(rows).sort_values(["horse_id", "date"]).reset_index(drop=True)


def test_prior_stats_exclude_self_and_future():
    f = _frame()
    p = ds.prior_stats(f, {"v": f["v"]})
    a = f[f["horse_id"] == "a"].index
    # 通算平均: k 行目は 0..k-1 行目の平均（k=0 は NaN）
    for k in range(len(a)):
        exp = f.loc[a[:k], "v"].mean() if k else np.nan
        got = p.loc[a[k], "prior_v"]
        assert (np.isnan(exp) and np.isnan(got)) or abs(exp - got) < 1e-12
    # 直近 5 走の平均も、自分より前だけ
    assert abs(p.loc[a[3], "prior5_v"] - f.loc[a[:3], "v"].mean()) < 1e-12


def test_prior_stats_unchanged_when_future_row_changes():
    f = _frame()
    base = ds.prior_stats(f, {"v": f["v"]})
    g = f.copy()
    last = g[g["horse_id"] == "a"].index[-1]
    g.loc[last, "v"] = 99.0
    new = ds.prior_stats(g, {"v": g["v"]})
    other = g.index[g.index != last]
    pd.testing.assert_frame_equal(base.loc[other[other < last]], new.loc[other[other < last]])       # 最後の行を変えても、それ以前の行の集計は完全一致
    assert (base.loc[g[g["horse_id"] == "b"].index].fillna(-1) == new.loc[g[g["horse_id"] == "b"].index].fillna(-1)).all().all()


def test_conditional_prior_mean_and_count_use_only_past():
    f = _frame()
    mask = f["early"] < 0.5
    m = ds.conditional_prior_mean(f, f["v"], mask)
    c = ds.conditional_prior_count(f, mask)
    a = f[f["horse_id"] == "a"].index
    for k in range(len(a)):
        past = f.loc[a[:k]]
        sel = past[past["early"] < 0.5]
        assert c.loc[a[k]] == len(sel)
        if len(sel):
            assert abs(m.loc[a[k]] - sel["v"].mean()) < 1e-12
        else:
            assert np.isnan(m.loc[a[k]])


def test_to_ease_pct_is_monotone_and_bounded():
    q = np.linspace(-1.0, 1.0, 101).tolist()
    drain = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
    e = td2.to_ease_pct(drain, q)
    assert (np.diff(e) <= 0).all()                 # Y が大きい（大変）ほど ease は小さい
    assert e[0] == 100.0 and e[-1] == 0.0
    assert abs(e[2] - 50.0) <= 1.0


def test_ease_label_thresholds():
    assert td2.ease_label(80) == "非常に楽" and td2.ease_label(60) == "楽" and td2.ease_label(45) == "普通" and td2.ease_label(30) == "やや大変" and td2.ease_label(29.9) == "大変"


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv(td2.ENV_FLAG, raising=False)
    assert td2.enabled() is False
    monkeypatch.setenv(td2.ENV_FLAG, "1")
    assert td2.enabled() is True


def _race_rows(n: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    d = pd.DataFrame({
        "pf_early": np.linspace(0.0, 1.0, n), "pf_rank": np.arange(1, n + 1), "field_size": n, "horse_number": np.arange(1, n + 1), "z_hat": -0.5, "pf_front": 1, "pf_stalker": 3,
        "pf_pressure": 0.3, "pf_known": n, "surface": "芝", "distance": 1600, "track_condition": "良", "venue": "東京",
        "prev_act_early_norm": rng.random(n), "prev_act_l3f_dev": rng.normal(size=n), "prev_z_act": rng.normal(size=n), "prev_distance": 1400,
    })
    for c in ("ability_adj", "prior_l3f_adj", "prior_l3f_dev_front", "prior_l3f_dev_rear", "prior_l3f_adj_fast", "prior_early", "prior_tpre_dev", "prior_finq",
              "n_prior", "n_prior_front", "n_prior_fast", "age", "weight", "weight_change", "rest_days"):
        d[c] = rng.normal(size=n)
    return d


def test_make_features_has_all_columns_and_no_result_columns():
    f = td2.make_features(_race_rows())
    assert list(f.columns) == td2.FEATURES
    assert not [c for c in f.columns if c.startswith(("act_", "finish", "odds", "popularity"))]      # 結果・市場の列を使わない
    assert f["demand_idx"].iloc[0] > f["demand_idx"].iloc[-1]                                         # 前の馬ほど要求が大きい


def test_make_features_independent_of_other_rows_results():
    d = _race_rows()
    f1 = td2.make_features(d)
    d2 = d.copy()
    d2["prev_act_l3f_dev"] = 123.0         # 前走の値は特徴量の入力だが、今回のレース内の他馬の結果とは無関係
    f2 = td2.make_features(d2)
    assert f1.drop(columns=["prev_l3f_dev"]).equals(f2.drop(columns=["prev_l3f_dev"]))


def test_pace_design_uses_prerace_columns_only():
    r = pd.DataFrame({"pf_front": [0, 2], "pf_stalker": [6, 3], "pf_pressure": [0.3, 0.5], "pf_known": [14, 16], "field_size": [14, 16], "surface": ["芝", "ダ"],
                      "distance": [1600, 1200], "going_group": ["良", "道悪"]})
    X = pc.design(r)
    assert list(X.columns) == pc.FEATURES
    assert X.loc[0, "f0"] == 1.0 and X.loc[1, "f0"] == 0.0 and X.loc[0, "f0_stalk"] > 0
    m = pc.fit_ridge(pd.concat([X] * 20, ignore_index=True), np.tile([0.5, -0.2], 20))
    p = pc.predict_ridge(m, X)
    assert p.shape == (2,) and np.isfinite(p).all()
