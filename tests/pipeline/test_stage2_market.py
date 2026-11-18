"""第 2 段のモデルの当日の推論 stage2_market: 列と型のそろえ方、q と offset、合計 1・2・3 へのそろえ、オッズの崩れの検出。
小さな xgb を一時フォルダに作って当てる（本物のモデルはリポジトリの外）。外部接続なし。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

xgb = pytest.importorskip("xgboost")

from src.pipeline.inference import stage2_market as sm  # noqa: E402


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("noise_fit")
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "surface": pd.Categorical(rng.choice(["芝", "ダート"], n), categories=["ダート", "芝", "障"])})
    y = (X["a"] + rng.normal(size=n) > 1).astype(int)
    for t in ("win", "top2", "top3"):
        for s in range(2):
            m = xgb.XGBClassifier(n_estimators=5, max_depth=2, enable_categorical=True, tree_method="hist", random_state=s)
            m.fit(X, y, base_margin=np.zeros(n))
            m.save_model(str(d / f"{t}_s{s}.json"))
    (d / "meta.json").write_text(json.dumps({"name": "tiny", "spec": {"offset": "raw"}, "seeds": 2, "feature_cols": ["a", "b", "surface"],
                                             "cat_levels": {"surface": ["ダート", "芝", "障"]}}, ensure_ascii=False), encoding="utf-8")
    return d


def _race(n=8, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"race_id": "R", "horse_id": [f"h{i}" for i in range(n)], "horse_number": np.arange(1, n + 1),
                         "a": rng.normal(size=n), "b": rng.normal(size=n), "surface": "芝", "extra": 1.0})


def test_predict_race_sums_and_r(model_dir):
    m = sm.Stage2Model(model_dir)
    X = _race()
    odds = np.array([2.0, 3.0, 5.0, 7.0, 8.0, 12.0, 20.0, 40.0])
    P = m.predict_race(X, odds)
    assert np.isclose(P["p_win"].sum(), 1.0) and np.isclose(P["p_top2"].sum(), 2.0) and np.isclose(P["p_top3"].sum(), 3.0)
    q = (1 / odds) / (1 / odds).sum()
    assert np.allclose(P["q"], q)
    assert np.allclose(P["r"], np.log(P["p_win"] / q))


def test_missing_odds_horse_gets_nan_q(model_dir):
    m = sm.Stage2Model(model_dir)
    odds = np.array([2.0, 3.0, 5.0, np.nan, 8.0, 12.0, 20.0, 40.0])
    P = m.predict_race(_race(), odds)
    assert np.isnan(P["q"][3]) and np.isnan(P["r"][3]) and np.isclose(np.nansum(P["q"]), 1.0)
    assert np.isfinite(P["p_win"]).all()


def test_missing_column_stops(model_dir):
    m = sm.Stage2Model(model_dir)
    with pytest.raises(KeyError):
        m.predict_race(_race().drop(columns=["b"]), np.full(8, 8.0))


def test_cast_uses_cat_levels(model_dir):
    m = sm.Stage2Model(model_dir)
    X = _race()
    X.loc[0, "surface"] = None
    c = m.cast(X)
    assert list(c.columns) == ["a", "b", "surface"] and list(c["surface"].cat.categories) == ["ダート", "芝", "障"]


@pytest.mark.parametrize("odds", [[1.2, 1.3, 1.4, 2.0], [50.0, 60.0, 70.0, 80.0], [3.0, np.nan, np.nan, np.nan]])
def test_broken_odds(odds):
    with pytest.raises(sm.OddsBroken):
        sm.check_odds(odds)
