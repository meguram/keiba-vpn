"""v6e_2026_gap の部品: Benter 型の混合の係数の推定と、レースごとの指標。"""

import numpy as np
import pandas as pd

from src.scripts.data import v6e_2026_gap as g


def _synthetic(a_true: float, b_true: float, n_races: int = 400, n: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        lq = rng.normal(0, 1, n)
        lp = 0.5 * lq + rng.normal(0, 0.7, n)                     # モデルは市場と相関しつつ独自の情報を持つ
        q, p = np.exp(lq) / np.exp(lq).sum(), np.exp(lp) / np.exp(lp).sum()
        z = a_true * np.log(p) + b_true * np.log(q)
        w = np.exp(z - z.max())
        win = rng.choice(n, p=w / w.sum())
        for i in range(n):
            rows.append({"race_id": f"r{r}", "horse_number": i + 1, "p": p[i], "q": q[i], "win": float(i == win), "odds": 0.8 / q[i]})
    return pd.DataFrame(rows)


def test_benter_a_recovers_model_weight():
    a, b = g.benter_a(_synthetic(0.6, 0.7))
    assert abs(a - 0.6) < 0.2 and abs(b - 0.7) < 0.2
    a0, _ = g.benter_a(_synthetic(0.0, 1.0, seed=1))                # モデルに市場以上の情報が無ければ a は 0 付近
    assert abs(a0) < 0.15


def test_summary_fields():
    s = g.summary(_synthetic(0.6, 0.7), n_boot=50)
    assert s["races"] == 400 and 0 <= s["top1_is_fav"] <= 1 and s["d_ll_lo"] <= s["d_ll"] <= s["d_ll_hi"]


def test_bands():
    assert g.new_band(0.05).endswith("10% 未満") and g.new_band(0.3).endswith("10〜49%") and g.new_band(0.5).endswith("50% 以上")
    assert g.attr_band(0.0).endswith("0%") and g.attr_band(0.6).endswith("50% 以上")


def test_race_frame_requires_all_runners_and_single_winner():
    res = pd.DataFrame({"race_id": ["r1"] * 3 + ["r2"] * 3 + ["r3"] * 3, "date": "2026-01-05", "horse_id": list("abcdefghi"),
                        "horse_number": [1, 2, 3] * 3, "finish_position": [1, 2, 3, 1, 1, 3, 1, 2, 3], "odds": [2.0, 4.0, 8.0] * 3, "popularity": [1, 2, 3] * 3})
    pred = pd.DataFrame({"race_id": ["r1"] * 3 + ["r2"] * 3 + ["r3"] * 2, "horse_id": list("abcdefgh"), "p_win": [0.5, 0.3, 0.4] * 2 + [0.6, 0.4]})
    d = g.race_frame(pred, res)
    assert set(d["race_id"]) == {"r1"}                      # r2 は同着、r3 は予測に無い馬がいる
    r1 = d.set_index("horse_id")
    assert np.isclose(r1["p"].sum(), 1.0) and np.isclose(r1["q"].sum(), 1.0) and np.isclose(r1.loc["a", "p"], 0.5 / 1.2)


def test_did_two_sample_and_paired():
    mk = lambda v: pd.DataFrame({"race_id": [f"r{i}" for i in range(len(v))], "ll_m": v, "ll_q": np.zeros(len(v))})   # noqa: E731
    r = g.did(mk(np.full(50, 0.1)), mk(np.full(50, 0.0)), mk(np.full(50, 0.4)), mk(np.full(50, 0.1)), n_boot=100)
    assert np.isclose(r["did"], 0.2) and r["lo"] - 1e-9 <= 0.2 <= r["hi"] + 1e-9
    assert np.isclose(g.two_sample(mk(np.full(30, 0.3)), mk(np.full(30, 0.1)), n_boot=100)["diff"], 0.2)
    dx = pd.DataFrame({"race_id": ["r1", "r1", "r2", "r2"], "p": [0.5, 0.5, 0.2, 0.8], "q": [0.5, 0.5, 0.5, 0.5], "win": [1.0, 0.0, 1.0, 0.0]})
    dy = dx.assign(p=[0.5, 0.5, 0.5, 0.5])
    pr = g.paired(dx, dy, n_boot=100)
    assert pr["races"] == 2 and np.isclose(pr["diff"], (-np.log(0.2) + np.log(0.5)) / 2)
