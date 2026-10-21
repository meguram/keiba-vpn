"""tp-63: アンサンブルの幅出し（ランダムフォレスト・決定木の袋詰め）のメンバー作成と、追加メンバーを足した合成の評価の部品。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.scripts.data import tp63_eval as ev
from src.scripts.data import tp63_members as mem


def _xy(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    X.loc[::7, "a"] = np.nan
    X["c"] = pd.Categorical(rng.choice(["x", "y", None], size=n))
    y = (X["b"].fillna(0) + rng.normal(size=n) * 0.5 > 0.8).astype(int).to_numpy()
    return X, {"win": y, "top2": y, "top3": y}


def test_to_numeric_fills_missing_with_sentinel_and_codes_categories():
    X, _ = _xy()
    Xn = mem.to_numeric(X)
    assert all(str(t) == "float32" for t in Xn.dtypes)
    assert Xn["a"].min() == mem.SENTINEL
    assert Xn["c"].min() >= -1


def test_tree_models_return_probabilities_for_three_targets():
    X, y = _xy()
    Xn = mem.to_numeric(X)
    ytr = {k: v[:1000] for k, v in y.items()}
    for kind, prm in (("rf", dict(n_estimators=8, max_depth=4, max_features=0.5, cal="platt")), ("et", dict(n_estimators=8, max_depth=4)), ("tree", dict(max_depth=3, min_samples_leaf=20))):
        out = mem.fit_sk_trees(kind, Xn.iloc[:1000], ytr, Xn.iloc[1000:], 0, 1, dict(kind=kind, **prm))
        assert set(out) == {"win", "top2", "top3"}
        for p in out.values():
            assert len(p) == 500 and np.all((p >= 0) & (p <= 1))


def test_platt_calibration_is_monotone_and_does_not_use_validation_labels():
    X, y = _xy()
    Xn = mem.to_numeric(X)
    ytr = {k: v[:1000] for k, v in y.items()}
    a = mem.fit_sk_trees("rf", Xn.iloc[:1000], ytr, Xn.iloc[1000:], 0, 1, dict(kind="rf", n_estimators=8, max_depth=4, cal="platt"))
    y2 = {k: v.copy() for k, v in y.items()}
    for k in y2:
        y2[k][1000:] = 1 - y2[k][1000:]          # 検証側のラベルを変えても予測は変わらない
    b = mem.fit_sk_trees("rf", Xn.iloc[:1000], {k: v[:1000] for k, v in y2.items()}, Xn.iloc[1000:], 0, 1, dict(kind="rf", n_estimators=8, max_depth=4, cal="platt"))
    assert np.allclose(a["win"], b["win"])


def _frame(extra=("rf_x",), n_races=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for year in (2022, 2023, 2024):
        for r in range(n_races):
            s = rng.normal(size=8)
            fin = np.argsort(np.argsort(-s)) + 1
            for h in range(8):
                rows.append({"race_id": f"{year}{r:04d}", "horse_id": f"h{h}", "year": year, "finish_position": int(fin[h]), "base": s[h]})
    df = pd.DataFrame(rows)
    for m in list(ev.MEMBERS) + list(extra):
        noise = rng.normal(scale=0.7, size=len(df))
        z = df["base"].to_numpy() + noise
        for t, k in (("win", 1.0), ("top2", 2.0), ("top3", 3.0)):
            df[f"{m}__{t}"] = 1 / (1 + np.exp(-(z - 2.2 + 0.6 * k)))
    return df


def test_added_member_schemes_run_and_return_normalised_probabilities():
    df = _frame()
    sc = ev.schemes_for(list(ev.MEMBERS) + ["rf_x"])
    for nm in ("P", "L3win2"):
        oof = ev.S.run_scheme(df, sc[nm], "forward")
        assert {"p_win", "p_top2", "p_top3"} <= set(oof.columns)
        assert abs(oof.groupby("race_id")["p_win"].sum() - 1).max() < 1e-6
        assert abs(oof.groupby("race_id")["p_top3"].sum() - 3).max() < 1e-6


def _small_frame(n=1800, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    race = np.array([f"r{i // 9:04d}" for i in range(n)])
    y = (X["a"] + rng.normal(size=n) > 1.0).astype(int).to_numpy()
    return X, race, {"win": y, "top2": y, "top3": y, "finish": np.where(y == 1, 1, 5)}


def test_condition_split_and_quality_specialists_return_three_targets_and_oracle():
    from src.pipeline.models import zoo

    orig = zoo.fit_predict
    mem.patch_zoo({})
    try:
        X, race, y = _small_frame()
        tr, va = np.arange(1200), np.arange(1200, 1800)
        mem.LABELS["pace"] = pd.Series(np.where(np.arange(1800) % 27 < 9, "slow", np.where(np.arange(1800) % 27 < 18, "mid", "high")), index=X.index)
        mem.LABELS["surface"] = pd.Series(np.where(np.arange(1800) % 2 == 0, "芝", "ダート"), index=X.index)
        mem.QUALITY["pace"] = pd.Series(mem.LABELS["pace"].to_numpy(), index=race).groupby(level=0).first()
        mem.QUALITY["pi"] = pd.DataFrame(1 / 3, index=mem.QUALITY["pace"].index, columns=["slow", "mid", "high"])
        prm = {"lgb": {"learning_rate": 0.2, "num_leaves": 4, "min_data_in_leaf": 5, "feature_fraction": 1.0, "bagging_fraction": 1.0, "lambda_l2": 1.0}, "n_rounds": 5}
        ytr = {k: v[tr] for k, v in y.items()}
        mem.ORACLE.clear()
        out = zoo.fit_predict("lgb_quality", X.iloc[tr], ytr, race[tr], X.iloc[va], race[va], 1, params=prm)
        assert set(out) >= {"win", "top2", "top3"} and np.all((out["win"] >= 0) & (out["win"] <= 1))
        assert len(mem.ORACLE) == 1 and len(mem.ORACLE[0][0]) == 600
        ls = dict(prm, by="surface")
        mem.LABELS["surface"] = pd.Series(np.where(np.arange(1800) % 2 == 0, "芝", "ダート"), index=X.index)
        # 学習行が 8000 未満の水準は全体モデルで補う（ここでは小さいので全部そうなる）
        out2 = zoo.fit_predict("lgb_split", X.iloc[tr], ytr, race[tr], X.iloc[va], race[va], 1, params=ls)
        assert not np.isnan(out2["win"]).any()
    finally:
        zoo.fit_predict = orig
