import numpy as np
import pandas as pd
import pytest

from src.pipeline.models import ablation as ab


def _synthetic(n_days=900, per_day=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        date = pd.Timestamp("2020-01-01") + pd.Timedelta(days=d)
        for r in range(per_day // 10):
            rid = f"{date:%Y%m%d}{r:02d}"
            skill = rng.normal(size=10)
            order = (-(skill + rng.normal(scale=1.0, size=10))).argsort().argsort() + 1
            for k in range(10):
                rows.append({"race_id": rid, "date": date, "finish_position": int(order[k]), "x_good": skill[k], "x_noise": rng.normal(),
                             "x_drift": skill[k] * 0.1 + 0.002 * d})
    df = pd.DataFrame(rows)
    df["year"] = df["date"].dt.year
    for name, col in ab.TARGETS.items():
        df[col] = (df["finish_position"] <= {"win": 1, "top2": 2, "top3": 3}[name]).astype("int8")
    return df


def test_asof_standardize_matches_brute_force_and_ignores_same_day_and_future():
    df = _synthetic(n_days=500, per_day=20)
    z = ab.asof_standardize(df, ["x_drift"], window_days=100, min_n=50)["x_drift"].to_numpy()
    i = 400 * 20 + 3  # 400 日目の行
    d = df.loc[i, "date"]
    w = df[(df["date"] >= d - pd.Timedelta(days=100)) & (df["date"] < d)]["x_drift"]
    expect = (df.loc[i, "x_drift"] - w.mean()) / w.std(ddof=0)
    assert z[i] == pytest.approx(expect, rel=1e-4)
    assert np.isnan(z[:5]).all()  # 窓の行数が足りない最初の日は欠損
    # 将来の値を変えても、過去の行は変わらない
    df2 = df.copy()
    df2.loc[df2["date"] > d, "x_drift"] += 100.0
    z2 = ab.asof_standardize(df2, ["x_drift"], window_days=100, min_n=50)["x_drift"].to_numpy()
    assert np.allclose(z[: i + 1], z2[: i + 1], equal_nan=True)


def test_race_nll_is_lower_for_better_scores():
    rid = np.array(["a"] * 3 + ["b"] * 3)
    y = np.array([1, 0, 0, 0, 1, 0])
    good = np.array([3.0, 0.0, 0.0, 0.0, 3.0, 0.0])
    bad = np.array([0.0, 3.0, 0.0, 3.0, 0.0, 0.0])
    assert ab.race_nll(rid, good, y) < ab.race_nll(rid, bad, y)


def test_evaluate_prefers_informative_feature_and_summary_decision(monkeypatch):
    monkeypatch.setattr(ab, "N_ROUNDS", 30)
    from src.pipeline.models.time_split import Fold

    df = _synthetic()
    y = df["year"].to_numpy()
    folds = []
    for vy in (2021, 2022):
        tr = np.where(y < vy)[0]
        va = np.where(y == vy)[0]
        folds.append(Fold(f"valid_{vy}", str(vy), tr, va, pd.Timestamp(f"{vy}-01-01"), pd.Timestamp(f"{vy}-01-01")))
    base = ab.evaluate(df, folds, ab.Spec("base", ["x_noise"]), seeds=[0, 1], threads=2)
    good = ab.evaluate(df, folds, ab.Spec("good", ["x_noise", "x_good"]), seeds=[0, 1], threads=2)
    res = pd.DataFrame(base + good)
    s = ab.summarize(res)
    assert s.loc["good", "mean"] < s.loc["base", "mean"]
    cmp = ab.compare_to_base(s, "base")
    assert bool(cmp.loc["good", "採用基準を満たす"]) and bool(cmp.loc["good", "全foldで改善"])
    # 学習の開始年を絞ると、学習行が空になる fold は評価されない
    r = ab.evaluate(df, folds, ab.Spec("late", ["x_good"], min_train_year=2021), seeds=[0], threads=2)
    assert {x["fold"] for x in r} == {"valid_2022"}


def test_drop_correlated_keeps_the_more_important_one():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    df = pd.DataFrame({"a": a, "a2": a * 2 + rng.normal(scale=0.01, size=5000), "b": rng.normal(size=5000)})
    kept, dropped = ab.drop_correlated(df, ["a", "a2", "b"], order=["a2", "a", "b"], threshold=0.9)
    assert "a2" in kept and "a" not in kept and "b" in kept and dropped["a"]["代表"] == "a2"
