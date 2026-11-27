"""day-pedigree の更新の式の試験（合成データだけ。手元の parquet・台帳を読まない）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.day_pedigree import features as fe
from src.research.day_pedigree import posterior as po


def _synth(n_days=300, races=6, field=14, theta_sd=0.0, seed=0):
    """日ごとに効き θ_d ~ N(0, theta_sd²) を 1 特徴に仕込み、5 着内を条件付きで引く。"""
    rng = np.random.default_rng(seed)
    rows = []
    rid = 0
    t0 = pd.Timestamp("2018-01-06 10:00")
    for d in range(n_days):
        th = rng.normal(0, theta_sd)
        for k in range(races):
            x = rng.normal(size=field)
            base = rng.normal(size=field)
            # Plackett-Luce の上位 5 頭（強さ = base + θ x）
            g = base + th * x - np.log(-np.log(rng.random(field)))
            top = np.argsort(-g)[:5]
            y = np.zeros(field)
            y[top] = 1
            for j in range(field):
                rows.append({"rid": rid, "day": f"d{d}", "k": k + 1, "x": x[j], "y": y[j], "theta": th,
                             "t_start": t0 + pd.Timedelta(days=d, minutes=60 * k)})
            rid += 1
    R = pd.DataFrame(rows)
    R["pt"] = 5.0 / field
    Rc = R.drop_duplicates("rid")[["rid", "day", "k", "t_start", "theta"]].reset_index(drop=True)
    Rc["t_conf"] = Rc["t_start"] + pd.Timedelta(minutes=15)
    Rc["surf"], Rc["year"], Rc["part"] = "T", 2018, 0
    return R, Rc


def _pri(Fn=1, tau2=0.1):
    return {("T", 2018): {"mu": np.zeros(Fn), "tau2": np.full(Fn, tau2)}}


def test_calibrate_sums_to_m():
    rid = np.repeat(np.arange(50), 12)
    eta = np.random.default_rng(1).normal(size=len(rid))
    m = np.full(len(rid), 5.0)
    m[rid == 3] = 6.0
    pt = po.calibrate(eta, rid, m)
    s = np.bincount(rid, weights=pt)
    assert np.allclose(s[[i for i in range(50) if i != 3]], 5.0, atol=1e-6)
    assert abs(s[3] - 6.0) < 1e-6


def test_no_future_leak_and_precision_grows():
    R, Rc = _synth(n_days=20, theta_sd=0.5)
    U, I, _ = po.race_stats(R[["x"]].to_numpy(), R["y"].to_numpy(), R["pt"].to_numpy(), R["rid"].to_numpy())
    mean, var, nprev = po.day_posterior(Rc, U, I, 1.0, _pri(), lag_min=60)
    # 第 k レースの事後は、後のレースの結果を壊しても変わらない
    R2 = R.copy()
    later = R2["k"] >= 4
    R2.loc[later, "y"] = 1 - R2.loc[later, "y"]
    U2, I2, _ = po.race_stats(R2[["x"]].to_numpy(), R2["y"].to_numpy(), R2["pt"].to_numpy(), R2["rid"].to_numpy())
    mean2, _, _ = po.day_posterior(Rc, U2, I2, 1.0, _pri(), lag_min=60)
    early = (Rc["k"] <= 4).to_numpy()            # k = 4 は 60 分前の締め切りで k ≤ 2 だけを使う
    assert np.allclose(mean[early], mean2[early])
    # 1 時間おきの発走・確定 15 分後・T-60 では、第 k レースは第 1〜k−2 レースを使う
    assert (nprev == np.maximum(Rc["k"].to_numpy() - 2, 0)).all()
    v = var[:, 0].reshape(20, 6)
    assert (np.diff(v[:, 1:], axis=1) <= 1e-12).all()      # 前のレースが増えると区間が狭まる
    _, _, n0 = po.day_posterior(Rc, U, I, 1.0, _pri(), lag_min=0)
    assert (n0 == Rc["k"].to_numpy() - 1).all()             # 表示用は前のレース全部


def test_recovers_planted_day_effect_and_null():
    R, Rc = _synth(n_days=400, theta_sd=0.6, seed=3)
    U, I, Xc = po.race_stats(R[["x"]].to_numpy(), R["y"].to_numpy(), R["pt"].to_numpy(), R["rid"].to_numpy())
    mean, _, _ = po.day_posterior(Rc, U, I, 1.0, _pri(tau2=0.36), lag_min=0)
    r = np.corrcoef(mean[Rc["k"].to_numpy() == 6, 0], Rc.loc[Rc["k"] == 6, "theta"])[0, 1]
    assert r > 0.5
    s = mean[R["rid"].to_numpy(), 0] * Xc[:, 0]
    sel = (R["k"] >= 2).to_numpy()
    a = po.offset_alpha(R["y"].to_numpy()[sel], R["pt"].to_numpy()[sel], s[sel], R["day"].to_numpy()[sel])
    assert a["alpha"] - 2 * a["se"] > 0
    R0, Rc0 = _synth(n_days=400, theta_sd=0.0, seed=4)
    U0, I0, Xc0 = po.race_stats(R0[["x"]].to_numpy(), R0["y"].to_numpy(), R0["pt"].to_numpy(), R0["rid"].to_numpy())
    m0, _, _ = po.day_posterior(Rc0, U0, I0, 1.0, _pri(tau2=0.36), lag_min=0)
    s0 = m0[R0["rid"].to_numpy(), 0] * Xc0[:, 0]
    a0 = po.offset_alpha(R0["y"].to_numpy()[sel], R0["pt"].to_numpy()[sel], s0[sel], R0["day"].to_numpy()[sel])
    assert abs(a0["alpha"]) < 3 * a0["se"]


def test_band_does_not_read_outcomes():
    """帯の表は、π5H（オッズだけ）と札だけから決まり、着順の列を持たない入力で動く。"""
    T = pd.DataFrame({"surf": "T", "anc": [f"a{i}" for i in range(12)], "days_with": 10, "days": 10,
                      "I_med": np.linspace(0.2, 3.0, 12), "p_mean": np.linspace(0.02, 0.6, 12)})
    for dn, dv in fe.DELTAS.items():
        T[f"pow_{dn}"] = fe.norm.cdf(dv * np.sqrt(T["I_med"]) - fe.Z_A)
    c = fe.choose(T, "log2")
    assert 5 <= c["N"] <= len(c["band"])
    sp = fe.choose_special(T, 14.0)
    assert (sp["special"]["p_mean"] < 2 / 14).all() and (sp["common"]["p_mean"] >= 2 / 14).all()


def test_harville_top_sums_to_five():
    from src.research.day_pedigree import data as dd
    r = pd.DataFrame({"race_id": np.repeat(["a", "b"], [10, 14]), "q": np.r_[np.full(10, 0.1), np.linspace(1, 2, 14) / np.linspace(1, 2, 14).sum()]})
    p = dd.harville_top(r, n_sim=4000)
    assert abs(p[:10].sum() - 5) < 1e-9 and abs(p[10:].sum() - 5) < 1e-9
    assert np.allclose(p[:10], 0.5, atol=0.05)
    assert p[23] > p[10]


def test_lookup_takes_largest_n_not_above():
    from src.research.day_pedigree import check as ck
    path = [(1, np.array([1.0])), (3, np.array([3.0]))]
    assert ck.lookup(path, 0) is None
    assert ck.lookup(path, 2)[0] == 1.0          # n = 2 は無いので n = 1
    assert ck.lookup(path, 5)[0] == 3.0


def test_null_ii_add_back_keeps_static_mean():
    """札を入れ替えた入力に I·μ を足し戻すと、事後の平均は μ のまわりに残る（0 へ縮まない）。"""
    R, Rc = _synth(n_days=50, theta_sd=0.0, seed=5)
    rid = R["rid"].to_numpy()
    rng = np.random.default_rng(0)
    X = R[["x"]].to_numpy()
    Xp = X.copy()
    for ii in R.groupby("rid").indices.values():
        Xp[ii] = X[rng.permutation(ii)]
    Up, Ip, _ = po.race_stats(Xp, R["y"].to_numpy(), R["pt"].to_numpy(), rid)
    mu = np.array([0.5])
    pri = {("T", 2018): {"mu": mu, "tau2": np.array([0.01])}}
    m_raw, _, _ = po.day_posterior(Rc, Up, Ip, 1.0, pri, lag_min=0)
    m_fix, _, _ = po.day_posterior(Rc, Up + Ip @ mu, Ip, 1.0, pri, lag_min=0)
    k6 = (Rc["k"] == 6).to_numpy()
    assert abs(np.mean(m_fix[k6, 0]) - 0.5) < abs(np.mean(m_raw[k6, 0]) - 0.5)
