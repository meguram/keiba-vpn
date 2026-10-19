import numpy as np
import pandas as pd
import torch

from src.scripts.data import tp36_rl as rl


def _frame(years=(2022, 2023, 2024, 2025), races=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for y in years:
        for r in range(races):
            n = int(rng.integers(8, 15))
            q = rng.dirichlet(np.ones(n) * 2)
            p = rng.dirichlet(np.ones(n) * 2)
            w = int(rng.choice(n, p=q))
            odds = 0.8 / q
            for h in range(n):
                rows.append({"race_id": f"{y}{r:08d}", "horse_number": h + 1, "year": y, "date": f"{y}-01-{r % 28 + 1:02d}",
                             "p_win": p[h], "q_win": q[h], "p_top3": min(0.98, p[h] * 3), "win_odds": float(odds[h]),
                             "ret_win": float(odds[h] * 100) if h == w else 0.0})
    return rl.features(pd.DataFrame(rows).sort_values(["date", "race_id", "horse_number"]).reset_index(drop=True))


def test_split_years_has_no_leak():
    d = _frame()
    for y in (2024, 2025):
        tr, va, te = rl.split_years(d, y)
        assert tr["year"].max() <= y - 2
        assert set(va["year"]) == {y - 1}
        assert set(te["year"]) == {y}
        assert not (set(tr["race_id"]) & set(te["race_id"]))
        assert not (set(va["race_id"]) & set(te["race_id"]))


def test_standardization_uses_training_stats_only():
    d = _frame()
    tr, va, te = rl.split_years(d, 2025)
    ptr = rl.pack(tr)
    pte = rl.pack(te, ptr["mu"], ptr["sd"])
    assert np.allclose(pte["mu"], ptr["mu"]) and np.allclose(pte["sd"], ptr["sd"])


def test_allocation_within_budget_and_masked():
    d = _frame()
    tr, _, _ = rl.split_years(d, 2025)
    pk = rl.pack(tr)
    for forced in (0, 1):
        net = rl.Policy(len(rl.FEATS), 8, 0.0, 1.0, forced)
        f = rl.alloc(net, pk, 0.1)
        assert (f >= 0).all()
        assert (f.sum(1) <= 0.1 + 1e-6).all()
        assert (f[~pk["M"]] == 0).all()
        if forced:
            assert torch.allclose(f.sum(1), torch.full((f.shape[0],), 0.1), atol=1e-5)


def test_training_improves_growth_on_train():
    d = _frame(races=80)
    best = d.groupby("race_id")["f_ev_m"].transform("max") == d["f_ev_m"]
    d["ret_win"] = np.where(best, d["win_odds"] * 100.0, 0.0)      # 期待値が最大の馬が必ず勝つ（学習で成長が増えるはず）
    tr, _, _ = rl.split_years(d, 2025)
    pk = rl.pack(tr)
    hp = {"hidden": 8, "skip": 0.0, "forced": 0, "lr": 3e-3, "wd": 1e-5, "B": 0.1, "ent": 0.0, "risk": 0.0, "drop": 0.0}
    net0 = rl.Policy(len(rl.FEATS), 8, 0.0, 0.0, 0)
    g0 = float(rl.growth(rl.alloc(net0, pk, 0.1), pk["RET"], pk["M"]).mean())
    net = rl.train(pk, hp, 5, 0)
    g1 = float(rl.growth(rl.alloc(net, pk, 0.1), pk["RET"], pk["M"]).mean())
    assert g1 > g0


def test_evaluate_roi_consistent():
    d = _frame()
    _, _, te = rl.split_years(d, 2025)
    pk = rl.pack(te)
    f = torch.zeros(pk["X"].shape[:2])
    f[:, 0] = 0.01
    r = rl.evaluate(f, pk, n_boot=100)
    exp = (pk["RET"][:, 0].numpy()).mean()
    assert abs(r["roi"] - exp) < 1e-5
    assert r["lo"] <= r["roi"] <= r["hi"]
    assert r["lo_adj"] <= r["lo"] + 1e-9


def test_single_ticket_policy_buys_at_most_one_horse():
    d = _frame()
    tr, _, _ = rl.split_years(d, 2025)
    pk = rl.pack(tr)
    net = rl.Policy(len(rl.FEATS), 8, 0.0, 1.0, 0, 1)
    f = rl.alloc(net, pk, 0.1)
    assert ((f > 0).sum(1) <= 1).all()
    assert (f.sum(1) <= 0.1 + 1e-6).all()
    g = rl.restrict_single(torch.tensor([[0.0, 0.02, 0.05, 0.0], [0.0, 0.0, 0.0, 0.0]]))
    assert g[0].tolist() == [0.0, 0.0, 0.07, 0.0] or abs(g[0, 2].item() - 0.07) < 1e-7
    assert g[1].sum().item() == 0.0


def test_single_ticket_training_gradient_flows():
    d = _frame(races=60)
    best = d.groupby("race_id")["f_ev_m"].transform("max") == d["f_ev_m"]
    d["ret_win"] = np.where(best, d["win_odds"] * 100.0, 0.0)
    tr, _, _ = rl.split_years(d, 2025)
    pk = rl.pack(tr)
    hp = {"hidden": 8, "skip": 0.0, "forced": 0, "single": 1, "lr": 3e-3, "wd": 1e-5, "B": 0.1, "ent": 0.0, "risk": 0.0, "drop": 0.0}
    net0 = rl.Policy(len(rl.FEATS), 8, 0.0, 0.0, 0, 1)
    g0 = float(rl.growth(rl.alloc(net0, pk, 0.1), pk["RET"], pk["M"]).mean())
    net = rl.train(pk, hp, 8, 0)
    g1 = float(rl.growth(rl.alloc(net, pk, 0.1), pk["RET"], pk["M"]).mean())
    assert g1 > g0
