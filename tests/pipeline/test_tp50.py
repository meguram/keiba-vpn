import numpy as np
import pandas as pd

from src.scripts.data import tp50_run as run
from src.scripts.data.tp50_lib import FEATS, Stats, pad_races, race_starts, reality_check, roi_ci, standardize


def toy(n_races: int = 40, per: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        for k in range(per):
            odds = float(rng.uniform(2, 30))
            hit = float(rng.random() < 0.8 / odds)
            rows.append({"ridx": r, "race_id": f"R{r:03d}", "year": 2024, "bt": k % 2, "odds": odds, "ret": hit * odds, "ev_mix": 0.8 + 0.1 * k, "ev_model": 0.7 + 0.1 * k})
    d = pd.DataFrame(rows)
    for f in FEATS:
        if f not in d.columns:
            d[f] = rng.standard_normal(len(d)).astype(np.float32)
    return d.sort_values(["ridx", "ev_mix"], ascending=[True, False]).reset_index(drop=True)


def test_race_starts_and_stats():
    d = toy()
    starts, counts = race_starts(d["ridx"].to_numpy())
    assert len(starts) == 40 and (counts == 6).all()
    st = Stats(d)
    s = st.summarize(np.ones(len(d), np.float32))
    assert np.allclose(s["S"], 6)
    assert np.isclose(s["R"].sum(), d["ret"].sum())
    assert (s["nb"] == 6).all()


def test_roi_ci_and_reality_check():
    S = np.ones(200)
    R = np.where(np.arange(200) % 2 == 0, 2.0, 0.0)           # ROI = 1.0
    roi, lo, hi = roi_ci(S, R, 200)
    assert abs(roi - 1.0) < 1e-9 and lo < 1.0 < hi
    rng = np.random.default_rng(0)
    p = rng.standard_normal((300, 5))
    rc = reality_check(p, 200)
    assert rc["n_strategies"] == 5 and 0.0 <= rc["p_value"] <= 1.0


def test_pad_races_maps_rows_back():
    d = toy(10, 4)
    X = d[FEATS].to_numpy(np.float32)
    Xp, retp, maskp, rid, pos, keep = pad_races(d, X, K=3)          # 4 件のうち 3 件に切り詰め
    assert Xp.shape == (10, 3, len(FEATS)) and maskp.sum() == 30 and keep.sum() == 30
    w = np.zeros((10, 3), np.float32)
    w[:, 0] = 1.0
    stake = run.rows_from_padded(w, rid, pos, keep, len(d))
    assert stake.sum() == 10 and (stake[pos == 0] == 1.0).all()


def test_weights_to_policies_argmax_needs_more_than_null():
    d = toy(5, 4)
    X = d[FEATS].to_numpy(np.float32)
    _, _, _, rid, pos, keep = pad_races(d, X, K=4)
    W = np.zeros((5, 5), np.float32)
    W[:, 1] = 0.6
    W[:, 4] = 0.4          # null
    W[2, 1], W[2, 4] = 0.1, 0.9        # このレースは賭けない
    pols = run.policies_from_weights("t", W, rid, pos, keep, len(d))
    am = pols["t|argmax"]
    assert am.sum() == 4 and am[(d["ridx"] == 2).to_numpy()].sum() == 0
    assert np.isclose(pols["t|stake"].sum(), 0.6 * 4 + 0.1)


def test_kelly_stake_caps_race_total():
    d = toy(3, 6)
    starts, _ = race_starts(d["ridx"].to_numpy())
    p = np.full(len(d), 0.5)
    o = np.full(len(d), 5.0)
    st = run.kelly_stake(p, o, 0.25, d["ridx"].to_numpy(), starts, cap=0.2)
    tot = np.add.reduceat(st, starts)
    assert (tot <= 0.2 + 1e-6).all() and (st > 0).all()


def test_rules_policy_thresholds_and_standardize():
    tr, ev = toy(60, 6, 1), toy(30, 6, 2)
    pols = run.cls_rules(tr, ev, 2024)
    assert "rule|tansho|ev_mix>=1.0" in pols
    sel = pols["rule|fukusho|ev_mix>=1.0"]
    assert set(np.unique(sel)) <= {0.0, 1.0}
    a, b = standardize(tr[FEATS].to_numpy(np.float32), ev[FEATS].to_numpy(np.float32))
    assert abs(a.mean()) < 1e-4


def test_hit_scale_defaults_to_one():
    c = run.hit_scale(toy(50, 6))
    assert c.shape == (6,) and (c > 0).all()
