"""tp-05a の部品（確率の整合・評価指標・レース内 softmax の目的関数）のテスト。"""

import numpy as np
import pandas as pd

from src.pipeline.models import win_models as wm


def _races():
    race = np.array(["a"] * 4 + ["b"] * 3)
    finish = np.array([1, 2, 3, 4, 2, 1, 3])
    return race, finish


def test_normalize_to_total_sums_and_keeps_order():
    race, _ = _races()
    p = np.array([0.5, 0.3, 0.2, 0.1, 0.4, 0.6, 0.1])
    for total in (1.0, 2.0):
        q = wm.normalize_to_total(race, p, total)
        assert abs(q[race == "a"].sum() - total) < 1e-6 and abs(q[race == "b"].sum() - total) < 1e-6
        assert np.all(np.argsort(q[race == "a"]) == np.argsort(p[race == "a"]))
        assert np.all((q > 0) & (q < 1 + 1e-9))


def test_small_field_gets_all_ones_when_total_exceeds_field():
    race = np.array(["x", "x"])
    assert np.allclose(wm.normalize_to_total(race, np.array([0.3, 0.2]), 3.0), 1.0)


def test_softmax_by_race_sums_to_one():
    race, _ = _races()
    p = wm.softmax_by_race(race, np.array([1.0, 2.0, 0.5, 0.0, 3.0, 1.0, 2.0]))
    assert abs(p[race == "a"].sum() - 1) < 1e-9 and abs(p[race == "b"].sum() - 1) < 1e-9
    assert p[race == "a"].argmax() == 1


def test_metrics_perfect_and_uniform():
    race, finish = _races()
    y = (finish == 1).astype(float)
    uni = wm.normalize_to_total(race, np.full(7, 0.3), 1.0)
    m = wm.evaluate_probs(race, uni, None, None, finish)
    assert abs(m["race_nll"] - (np.log(4) + np.log(3)) / 2) < 1e-6
    good = wm.softmax_by_race(race, y * 10)
    mg = wm.evaluate_probs(race, good, None, None, finish)
    assert mg["race_nll"] < 0.01 and mg["top1_hit"] == 1.0 and mg["ndcg3"] > 0.7


def test_ndcg_prefers_correct_order():
    race, finish = _races()
    best = wm.ndcg_at_k(race, -finish.astype(float), finish)
    worst = wm.ndcg_at_k(race, finish.astype(float), finish)
    assert best == 1.0 and worst < best


def test_softmax_objective_gradient_matches_numeric():
    race, finish = _races()
    y = (finish == 1).astype(float)
    fobj = wm.softmax_objective(race, y)
    s = np.array([0.2, -0.1, 0.4, 0.0, 0.3, 0.1, -0.2])
    g, h = fobj(s, None)

    def loss(sc):
        p = wm.softmax_by_race(race, sc)
        return -np.log(p[y == 1]).sum()

    num = np.array([(loss(s + np.eye(7)[i] * 1e-5) - loss(s - np.eye(7)[i] * 1e-5)) / 2e-5 for i in range(7)])
    assert np.allclose(g, num, atol=1e-5) and (h > 0).all()


def test_softmax_objective_skips_races_without_single_winner():
    race = np.array(["a", "a", "b", "b"])
    y = np.array([1.0, 1.0, 0.0, 1.0])         # a は勝ち馬が 2 頭
    g, _ = wm.softmax_objective(race, y)(np.zeros(4), None)
    assert np.allclose(g[:2], 0) and not np.allclose(g[2:], 0)


def test_paired_bootstrap_detects_improvement():
    a = pd.Series(np.full(200, 0.30), index=[f"r{i}" for i in range(200)])
    b = a - 0.01
    r = wm.paired_bootstrap_diff(a, b, n_boot=200)
    assert r["hi"] < 0 and abs(r["diff"] + 0.01) < 1e-9


def test_add_race_relative_columns():
    df = pd.DataFrame({"race_id": ["a", "a", "a", "b", "b"], "x": [1.0, 2.0, 3.0, 5.0, 7.0]})
    r = wm.add_race_relative(df, ["x"])
    assert list(r["x_rrank"]) == [3, 2, 1, 2, 1] and list(r["x_rgap"]) == [-2, -1, 0, -2, 0]
    assert abs(r.loc[0, "x_rz"] + 1) < 1e-6


def test_fit_predict_runs_all_kinds_on_small_data():
    rng = np.random.default_rng(0)
    n_r, n_h = 60, 8
    race = np.repeat([f"r{i:03d}" for i in range(n_r)], n_h)
    X = pd.DataFrame({"a": rng.normal(size=n_r * n_h), "b": rng.normal(size=n_r * n_h)})
    finish = np.concatenate([np.argsort(np.argsort(-(X["a"].to_numpy()[i * n_h:(i + 1) * n_h] + rng.normal(0, 0.5, n_h)))) + 1 for i in range(n_r)])
    y = {"win": (finish == 1).astype(int), "top2": (finish <= 2).astype(int), "top3": (finish <= 3).astype(int), "finish": finish}
    for kind in ("binary", "softmax", "lambdarank", "xendcg"):
        out = wm.fit_predict(kind, X, y, race, X, race, threads=2, n_rounds=20, params={"min_data_in_leaf": 5})
        assert all(len(v) == len(X) for v in out.values()), kind
    sm = wm.softmax_by_race(race, out["win_score"])
    assert wm.top1_hit_rate(race, sm, y["win"].astype(float)) > 0.3   # 乱択(1/8)より明らかに良い
