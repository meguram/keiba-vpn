import numpy as np

from src.pipeline.models import ensemble as en


def _data(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    true_logit = rng.normal(-1.5, 1.0, n)
    y = (rng.uniform(size=n) < en.sigmoid(true_logit)).astype(float)
    good = en.sigmoid(true_logit + rng.normal(0, 0.2, n))
    bad = en.sigmoid(true_logit + rng.normal(0, 1.5, n))
    return np.stack([good, bad], axis=1), y


def test_weighted_logit_prefers_the_better_model():
    P, y = _data()
    w = en.fit_simplex_weights(en.logit(P), y)
    assert abs(w.sum() - 1) < 1e-9 and (w >= 0).all() and w[0] > w[1]


def test_greedy_selection_prefers_the_better_model_and_sums_to_one():
    P, y = _data()
    w = en.greedy_selection(P, y)
    assert abs(w.sum() - 1) < 1e-9 and w[0] > w[1]


def test_blends_beat_the_worse_single_model():
    P, y = _data()
    Z = en.logit(P)
    base_bad = en.ll(P[:, 1], y)
    for m in en.METHODS:
        st = en.fit_method(m, P, Z, y)
        assert en.ll(en.blend(m, P, Z, st), y) < base_bad, m


def test_loyo_eval_returns_normalized_probabilities():
    P, y = _data(3000)
    years = np.repeat([2022, 2023, 2024], 1000)
    race = np.repeat(np.arange(300), 10)
    ll_, out = en.loyo_eval("weighted_logit", P, years, race, y, 1.0)
    assert np.isfinite(ll_) and abs(out[race == 0].sum() - 1) < 1e-6


def test_apply_spec_matches_fit_and_blend():
    P, y = _data()
    Z = en.logit(P)
    for m in en.METHODS:
        st = en.fit_method(m, P, Z, y)
        entry = {"method": m}
        if m in ("weighted_logit", "greedy"):
            entry["weights"] = list(st)
        elif m == "stack_lr":
            entry["coef"], entry["intercept"] = list(st.coef_[0]), float(st.intercept_[0])
        assert np.allclose(en.apply_spec(entry, P), en.blend(m, P, Z, st)), m
