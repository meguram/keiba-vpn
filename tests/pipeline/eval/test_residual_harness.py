"""残差の評価の共通部品: 効く列は合格し、乱数の列は合格しないこと。"""

import numpy as np
import pandas as pd

from src.pipeline.eval import residual_harness as H


def _frame(seed=0, n_races=3000, k=8):
    rng = np.random.default_rng(seed)
    n = n_races * k
    d = pd.DataFrame({
        "race_id": np.repeat(np.arange(n_races), k).astype(str),
        "horse_id": np.tile(np.arange(k), n_races).astype(str),
        "year": np.repeat(rng.choice([2021, 2022, 2023, 2024], n_races), k),
        "venue_code": rng.choice(["05", "06"], n), "surface": rng.choice(["芝", "ダート", "障害"], n, p=[0.5, 0.4, 0.1]), "grade": "1", "track_condition": "良",
        "distance": 1600, "field_size": k, "a_style": rng.normal(size=n), "a_l3fz_ewm2": rng.normal(size=n), "gate": rng.random(n),
    })
    signal = rng.normal(size=n)
    d["signal"] = signal
    d["noise"] = rng.normal(size=n)
    # 基準の確率は signal を知らない。真の勝ちやすさは signal で上下する。
    base = np.full(n, 1 / k)
    lo = np.log(base / (1 - base)) + 0.6 * signal
    p_true = 1 / (1 + np.exp(-lo))
    win = np.zeros(n)
    for i in range(n_races):
        s = slice(i * k, (i + 1) * k)
        pr = p_true[s] / p_true[s].sum()
        win[s][rng.choice(k, p=pr)] = 1  # noqa
    d["finish_position"] = np.where(win == 1, 1, 2)
    d["p_win"] = base
    return H.prepare(d)


def test_score_test_passes_signal_and_rejects_noise():
    d = _frame()
    r = {x["name"]: x for x in H.score_test(d, [("sig", "signal", None), ("noise", "noise", None)], z_line=3.0)}
    assert r["sig"]["pass"] is True
    assert r["noise"]["pass"] is False


def test_boost_and_tier2_separate_signal_from_noise():
    # 小さな表では木の揺れが ±0.0004 ほど出るため、乱数の列の合否そのものではなく、効く列との差の大きさで見る。
    d = _frame(seed=1, n_races=6000)
    b = H.boost_test(d, {"SIG": ["signal"], "NOISE": ["noise"]}, ctx=["a_style", "gate", "venue_code", "surface"], years=(2022, 2023, 2024))
    assert b["SIG"]["pass"] is True
    assert b["SIG"]["mean_diff"] < -0.005
    assert b["NOISE"]["mean_diff"] > -0.001
    tr, te = d[d["year"] != 2024], d[d["year"] == 2024]
    t = H.tier2(tr, te, {"SIG": ["signal"], "NOISE": ["noise"]}, ctx=["a_style", "gate", "venue_code", "surface"], n_boot=50)
    assert t["SIG"]["adopt"] is True
    assert t["SIG"]["ci95"][1] < 0
    assert t["NOISE"]["diff_vs_control"] > -0.001


def test_multiplier_candidate_and_guards():
    d = _frame(seed=2)
    r = H.score_test(d, [("sig_x_s", "signal", "s")], z_line=1.0)   # 緩い線は 3.48 に引き上げられる
    assert r[0]["n"] == len(d) and "s" in d
    bad = d.assign(year=2026)
    try:
        H.prepare(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("2026 年を止めていない")
    try:
        H.boost_test(d, {"control": ["signal"]})
    except ValueError:
        pass
    else:
        raise AssertionError("control の名前を止めていない")
