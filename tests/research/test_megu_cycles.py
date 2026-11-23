"""めぐサイクルの評価器（src/research/megu_cycles/mc_eval.py）の小さな確かめ。手元のデータを読まない。"""

import json

import numpy as np
import pandas as pd

from src.research.megu_cycles import mc_data as D
from src.research.megu_cycles import mc_eval as E


def _races(n_races: int = 40, n: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        q = rng.dirichlet(np.ones(n) * 2)
        odds = 0.8 / q
        order = rng.permutation(n)
        fin = np.empty(n, int)
        fin[order] = np.arange(1, n + 1)
        for i in range(n):
            rows.append({"race_id": f"2018{r:08d}", "date": f"2018-01-{1 + r % 28:02d}", "year": 2018, "hn": i + 1, "graded": False,
                         "odds_final": odds[i], "pop_final": float(np.argsort(np.argsort(odds))[i] + 1), "q_final": q[i],
                         "y1": float(fin[i] == 1), "y3": float(fin[i] <= 3), "ret_tan": odds[i] * (fin[i] == 1),
                         "ret_fuku": 1.5 * (fin[i] <= 3), **{f"odds_pre{k}": odds[i] for k in range(D.N_PRE)}})
    return pd.DataFrame(rows)


def test_harville_top3_sums_to_three():
    d = _races(5)
    p = E.harville_top3(d, d["q_final"].to_numpy())
    s = pd.Series(p).groupby(d["race_id"]).sum().to_numpy()
    assert np.allclose(s, 3.0, atol=1e-6)
    assert ((p >= 0) & (p <= 1)).all()


def test_evaluate_buy_all_roi_matches_returns():
    d = _races()
    r = E.evaluate(d, lambda e, k: np.ones(len(e)), "win", n_boot=100)
    assert abs(r["roi"] - d["ret_tan"].sum() / len(d)) < 1e-9
    assert r["bets"] == len(d)
    assert r["roi_ci"][0] <= r["roi"] <= r["roi_ci"][1]


def test_evaluate_uses_decision_odds_versions():
    d = _races()
    d["odds_pre1"] = 1e4                           # 版 1 だけ全部 omax を超える
    st = lambda e, k: (e[f"odds_pre{k}"] < 1e3).astype(float).to_numpy()
    r = E.evaluate(d, st, "win", n_boot=0)
    assert abs(r["bets"] - len(d) * (D.N_PRE - 1) / D.N_PRE) < 1e-9
    assert abs(r["roi"] - d["ret_tan"].sum() / len(d)) < 1e-9     # 賭け金で割るので、版を平均しても回収率は同じ


def test_goal_fixed_values():
    g = json.loads(D.GOAL.read_text(encoding="utf-8"))
    assert g["goal"]["roi_select_min"] == 1.0 and g["goal"]["roi_confirm_min"] == 1.0
    assert g["goal"]["confirm_over_select_min"] == 0.9 and g["goal"]["confirm_bets_per_year_min"] == 50
    assert g["confirm_budget"]["max_candidates"] == 10
    assert g["periods"]["confirm"] == ["2022-01-01", "2024-12-31"]


def test_goal_met_rules():
    ok, _ = E.goal_met({"roi": 1.10}, {"roi": 1.02, "bets_per_year": 60})
    assert ok
    ok, why = E.goal_met({"roi": 1.30}, {"roi": 1.05, "bets_per_year": 60})
    assert not ok and any("0.9" in w for w in why)
    ok, _ = E.goal_met({"roi": 1.10}, {"roi": 1.05, "bets_per_year": 40})
    assert not ok


def test_frame_reader_filters_2025():
    import inspect
    src = inspect.getsource(D.load_frame)
    assert '("year", "<=", YEAR_END)' in src and "assert" in src
    assert D.YEAR_END == 2024


def test_winplace_is_half_and_half():
    d = _races()
    d["fin"] = np.where(d["y1"] > 0, 1, np.where(d["y3"] > 0, 2, 9))
    d["field"] = 8
    r_w = E.evaluate(d, lambda e, k: np.ones(len(e)), "win", n_boot=0)
    r_p = E.evaluate(d, lambda e, k: np.ones(len(e)), "place", n_boot=0)
    r_wp = E.evaluate(d, lambda e, k: np.ones(len(e)), "winplace", n_boot=0)
    assert abs(r_wp["roi"] - (r_w["roi"] + r_p["roi"]) / 2) < 1e-9


def test_place_slots_small_field():
    d = _races(3, n=7)
    d["field"] = 7
    p = E.harville_top3(d, d["q_final"].to_numpy())
    assert np.allclose(pd.Series(p).groupby(d["race_id"]).sum().to_numpy(), 2.0, atol=1e-6)
    assert (E.place_slots(d) == 2).all()
