"""tp-42: めぐ指数から作るレースレベル特徴量の検査（先読みなし・手計算との一致）。"""

import numpy as np
import pandas as pd

from src.pipeline.features import tp42_racelevel as rl


def _synth(seed=0, n_h=36, n_r=70):
    rng = np.random.default_rng(seed)
    rows = []
    days = np.sort(rng.choice(np.arange(18000, 18600), size=n_r, replace=False))
    for r, dn in enumerate(days):
        k = int(rng.integers(7, 13))
        hs = rng.choice(n_h, size=k, replace=False)
        for h in hs:
            rows.append({"race_id": f"R{r:03d}", "horse_id": f"H{h:03d}", "dnum": float(dn),
                         "a": float(rng.normal(50, 10)) if rng.random() > 0.08 else np.nan,
                         "surf": float(rng.integers(0, 2)), "dist": float(rng.choice([1200, 1600, 2000, 2400]))})
    d = pd.DataFrame(rows).sort_values(["horse_id", "dnum", "race_id"], kind="stable").reset_index(drop=True)
    return d


def _all(d):
    parts = [rl.grp_A(d), rl.grp_A2(d), rl.grp_B(d), rl.grp_C(d)]
    return pd.concat(parts, axis=1)


def test_no_lookahead_features_ignore_results_on_or_after_the_race_day():
    d = _synth(1)
    base = _all(d)
    cut = float(np.sort(d["dnum"].unique())[len(d["dnum"].unique()) // 2])
    d2 = d.copy()
    rng = np.random.default_rng(5)
    late = d2["dnum"] >= cut
    d2.loc[late, "a"] = rng.normal(80, 30, late.sum())                       # 当日以降の結果を全部入れ替える
    alt = _all(d2)
    on_or_before = (d["dnum"] <= cut).to_numpy()
    a = base.loc[on_or_before].to_numpy(float)
    b = alt.loc[on_or_before].to_numpy(float)
    both_nan = np.isnan(a) & np.isnan(b)
    assert np.allclose(np.where(both_nan, 0, a), np.where(both_nan, 0, b), atol=1e-9, rtol=0), "当日以降の指数が今回の特徴量に入っている"


def test_opponent_level_matches_brute_force():
    d = _synth(2)
    x = rl.add_race_level(d)
    for rid in list(d["race_id"].unique())[:12]:
        g = d[d["race_id"] == rid]
        for i, row in g.iterrows():
            others = g.drop(index=i)["a"].dropna()
            want_m = others.mean() if len(others) >= rl.MIN_FIELD - 1 else np.nan
            want_t = np.sort(others.to_numpy())[::-1][:3].mean() if len(others) >= 3 else np.nan
            if np.isnan(row["a"]):
                others = g["a"].dropna()
                want_m = others.mean() if len(others) >= rl.MIN_FIELD - 1 else np.nan
                want_t = np.sort(others.to_numpy())[::-1][:3].mean() if len(others) >= 3 else np.nan
            assert np.isclose(x.loc[i, "opp_mean"], want_m, equal_nan=True)
            assert np.isclose(x.loc[i, "opp_top3"], want_t, equal_nan=True)


def test_hindsight_matches_brute_force_and_only_uses_runs_before_the_race_day():
    d = _synth(3)
    f = rl.grp_A2(d)
    d = d.reset_index(drop=True)
    by_h = {h: g.reset_index() for h, g in d.groupby("horse_id", sort=False)}
    nxt = {}
    for h, g in by_h.items():
        for k in range(len(g) - 1):
            nxt[g.loc[k, "index"]] = (g.loc[k + 1, "dnum"], g.loc[k + 1, "a"])
    checked = 0
    for i in range(0, len(d), 7):
        h = d.loc[i, "horse_id"]
        g = by_h[h]
        pos = int(g.index[g["index"] == i][0])
        if pos < 1:
            continue
        prev = int(g.loc[pos - 1, "index"])
        rid = d.loc[prev, "race_id"]
        vals = []
        for j in d.index[d["race_id"] == rid]:
            if d.loc[j, "horse_id"] == h or j not in nxt:
                continue
            nd, na = nxt[j]
            if nd < d.loc[i, "dnum"] and not np.isnan(na):
                vals.append(na)
        want = np.mean(vals) if len(vals) >= 2 else np.nan
        assert np.isclose(f.loc[i, "rlA2_next_last"], want, equal_nan=True), (i, vals)
        checked += 1
    assert checked > 20


def test_field_features_use_only_pre_race_expectations():
    d = _synth(4)
    b = rl.grp_B(d)
    c = rl.grp_C(d)
    E = c["rlC_E"].to_numpy(float)
    for rid in list(d["race_id"].unique())[:10]:
        m = (d["race_id"] == rid).to_numpy()
        e = E[m]
        if np.isfinite(e).sum() < rl.MIN_FIELD:
            continue
        assert np.isclose(b.loc[m, "rlB_max"].iloc[0], np.nanmax(e))
        assert np.isclose(b.loc[m, "rlB_mean"].iloc[0], np.nanmean(e))
        k = np.isfinite(e)
        i = int(np.flatnonzero(k)[0])
        assert b.loc[m, "rlB_n_stronger"].to_numpy()[i] == (e[k] > e[i]).sum()


def test_expectation_ewm_matches_manual():
    d = _synth(5)
    c = rl.grp_C(d)
    h = d["horse_id"].iloc[0]
    g = d[d["horse_id"] == h]
    for pos in range(2, len(g)):
        prior = g["a"].to_numpy()[:pos][::-1][: rl.K]
        w = rl.W ** np.arange(len(prior))
        m = np.isfinite(prior)
        want = (prior[m] * w[m]).sum() / w[m].sum() if m.any() else np.nan
        assert np.isclose(c.loc[g.index[pos], "rlC_E"], want, equal_nan=True)
