import numpy as np
import pandas as pd

from src.scripts.data import tp56_eval as ev


def _race(n=12, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(n))
    return pd.DataFrame({"p_win": p, "p_top3": np.clip(p * 3, 0, 0.95), "pw_int": rng.dirichlet(np.ones(n)), "p3_int": rng.random(n),
                         "fit_rank": rng.integers(1, 5, n).astype(float), "finish_position": rng.permutation(n) + 1.0})


def test_marks_are_distinct_and_seven():
    g = _race()
    for v in ("base", "int", "fit"):
        m = ev.make_marks(g, v)
        assert len(m) == 7 and len(set(m)) == 7


def test_tickets_sorted_and_counts():
    M = [3, 5, 7, 1, 2, 4, 6]
    odds = {h: 2.0 for h in M}
    assert ev.tickets_for("wide1", M, odds) == [("wide", (3, 5))]
    assert len(ev.tickets_for("tri4box", M, odds)) == 4
    assert len(ev.tickets_for("tri_axis6", M, odds)) == 6
    odds[1] = 30.0       # 高オッズ馬が印の中にいる
    t = ev.tickets_for("lsC_12_a", M, odds)
    assert any(c == (1, 3, 5) or 1 in c for _, c in t) and all(list(c) == sorted(c) for _, c in t)
    assert ev.tickets_for("lsC_12_a", M, {h: 2.0 for h in M}) == [("wide", (3, 5))]     # いなければ人気側の絞った買い


def test_settle_handles_broken_and_hits():
    hits = {"wide": {(3, 5): 250.0}, "sanrenpuku": {(1, 3, 5): 1200.0}, "_broken": set()}
    assert ev.settle(hits, [("wide", (3, 5)), ("wide", (3, 7))]) == 2.5
    assert ev.settle({**hits, "_broken": {"wide"}}, [("wide", (3, 5))]) is None


def test_clogit_recovers_signal():
    rng = np.random.default_rng(1)
    n_r = 400
    cnt = rng.integers(8, 14, n_r)
    ridx = np.repeat(np.arange(n_r), cnt)
    X = rng.normal(size=(len(ridx), 2))
    off = rng.normal(size=len(ridx))
    st = np.r_[0, np.cumsum(cnt)[:-1]]
    y = np.zeros(len(ridx))
    for a, c in zip(st, cnt):
        s = off[a:a + c] + X[a:a + c] @ np.array([0.8, 0.0])
        p = np.exp(s - s.max()); p /= p.sum()
        y[a + rng.choice(c, p=p)] = 1
    th = ev.fit_clogit(X, off, y, ridx, l2=0.1)
    assert abs(th[0] - 0.8) < 0.25 and abs(th[1]) < 0.25
