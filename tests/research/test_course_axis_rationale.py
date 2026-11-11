"""コース別の走法軸の優位の根拠づけ（形状での再現判定・実力の除去・形状ベクトル）の確認。"""
from __future__ import annotations

import numpy as np

from src.research.race import course_axis_rationale as car

AXES = ["gear", "ts"]


def _feats(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    keys = [f"c{i}" for i in range(n)]
    X = np.column_stack([np.ones(n), rng.integers(-2, 3, n), rng.integers(0, 2, n), rng.integers(0, 2, n), rng.integers(0, 2, n), rng.integers(-2, 3, n)]).astype(float)
    return keys, {k: list(x) for k, x in zip(keys, X)}, X[:, 5]


def _stats(keys, dist, slope_fit: float, slope_chk: float, se: float = 0.01, seed: int = 1):
    rng = np.random.default_rng(seed)
    out = {}
    for k, d in zip(keys, dist):
        out[k] = {p: {"b": [float(rng.normal(0, se)), float(slope * d + rng.normal(0, se))], "se": [se, se]} for p, slope in (("fit", slope_fit), ("y26", slope_chk))}
    return out


def test_same_sign_in_both_periods_is_replicated():
    keys, feats, dist = _feats(60)
    ev = car.shape_effects(_stats(keys, dist, 0.01, 0.01), feats, AXES)
    e = next(x for x in ev if x["axis"] == "ts" and x["feature"] == "距離")
    assert e["verdict"] == "根拠あり" and e["b_fit"] > 0 and e["b_chk"] > 0


def test_one_sided_effect_is_not_replicated():
    keys, feats, dist = _feats(60)
    ev = car.shape_effects(_stats(keys, dist, 0.01, 0.0), feats, AXES)
    e = next(x for x in ev if x["axis"] == "ts" and x["feature"] == "距離")
    assert e["verdict"] == "片方の期間のみ"


def test_opposite_sign_is_not_replicated():
    keys, feats, dist = _feats(60)
    ev = car.shape_effects(_stats(keys, dist, 0.01, -0.01), feats, AXES)
    e = next(x for x in ev if x["axis"] == "ts" and x["feature"] == "距離")
    assert e["verdict"] == "片方の期間のみ" and not e["same_sign"]


def test_pure_noise_has_no_replication():
    keys, feats, dist = _feats(60)
    ev = car.shape_effects(_stats(keys, dist, 0.0, 0.0), feats, AXES)
    assert all(x["verdict"] != "根拠あり" for x in ev)


def test_shape_features_none_when_straight_differs():
    items = [{"final_straight_m": 300}, {"final_straight_m": 500}]
    assert car.shape_features(items, "芝", 1600) is None


def test_shape_features_values():
    items = [{"final_straight_m": 525, "corner_radius_class_A": "大回り", "goal_slope_exists": True}]
    f = car.shape_features(items, "ダート", 2000)
    assert f == [1.0, 1.25, 0.0, 1.0, 1.0, 1.0]
    small = car.shape_features([{"final_straight_m": 300, "corner_radius_class_A": "小回り（急）"}], "芝", 1200)
    assert small[2] == 1.0 and small[5] == -1.0


def test_bh_is_monotone_and_bounded():
    p = [0.001, 0.01, 0.04, 0.5, 0.9]
    q = car.bh(p)
    assert all(0 <= v <= 1 for v in q)
    assert [q[i] for i in np.argsort(p)] == sorted(q)
    assert q[0] <= p[0] * len(p)


def test_course_counts_chance_level():
    rng = np.random.default_rng(3)
    stats = {f"c{i}": {"fit": {"b": list(rng.normal(0, 1, 6)), "se": [1.0] * 6}} for i in range(80)}
    c = car.course_counts(stats, "fit", list(range(6)))
    assert c["n"] == 480 and abs(c["sd"] - 1) < 0.15 and c["hits"] < 25


def _zone_data(effect: float, seed: int = 0, n: int = 6000):
    from src.scripts.data import course_axis_rationale as host
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 6))
    surf = np.ones(n, dtype=bool)
    S = host.zone_scores(Z, {"芝": surf})
    cls = np.array(["1勝"] * n)
    course = np.arange(n) < n // 2
    p = np.where(course, 0.2 * np.exp(effect * Z[:, 0]), 0.2 * np.exp(0.5 * Z[:, 0]))
    good = (rng.random(n) < p / (1 + p)).astype(float)
    race = rng.integers(0, n // 8, n).astype(str)
    return host, S, good, surf, cls, course, race


def test_zone_scores_are_deciles_1_to_10():
    host, S, *_ = _zone_data(0.5)
    assert S.min() == 1 and S.max() == 10
    assert abs(np.bincount(S[:, 0])[1:].std()) < 5


def test_delta_minus_reference_isolates_the_course_effect():
    out = {}
    for eff in (0.5, 1.5):
        host, S, good, surf, cls, course, race = _zone_data(eff)
        per = {"all": np.ones(len(good), dtype=bool)}
        ref = {"g": host.delta_ref(S, good, cls, {"芝": surf}, per), "p": host.delta_ref(S, good * 3, cls, {"芝": surf}, per)}
        h = host.hist_counts(S, good, good * 3, course, race, per, cls, "芝", ref)
        k = host.KEYS[0]
        d = h[k]["dg_all"]
        out[eff] = (d["d"] - d["ref"]) / d["se"]
    assert abs(out[0.5]) < 2.5 and out[1.5] > 3
