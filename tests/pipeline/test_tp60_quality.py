"""tp-60 レース質 × 血統グループ: 交互作用の効果表と、オラクル／事前の期待上乗せの部品を、小さな架空データで確かめる。"""

import numpy as np
import pandas as pd

from src.scripts.data import tp60_quality as Q


def _toy(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 2, n)
    q = rng.choice(["a", "b"], n)
    # g=1 は q=b のとき +0.05 の超過、それ以外は 0
    r = rng.normal(0, 0.3, n) + 0.05 * ((g == 1) & (q == "b"))
    return pd.DataFrame({"g": g.astype(float), "q": q, "r": r, "pa": 0.3, "pb": 0.7})


def test_effect_table_finds_the_interaction_and_shrinks():
    d = _toy()
    tab, gm = Q.effect_table(d, "g", "q", "r", k=0.0)
    row = tab[(tab["g"] == 1) & (tab["q"] == "b")].iloc[0]
    other = tab[(tab["g"] == 1) & (tab["q"] == "a")].iloc[0]
    assert row["raw"] > 0.015 and other["raw"] < -0.015            # g=1 の中で b が a より高い（交互作用は平均 g を基準にするので a は負）
    shr, _ = Q.effect_table(d, "g", "q", "r", k=300.0)
    assert abs(shr["eff"]).max() < abs(tab["raw"]).max()


def test_uplift_oracle_and_prior_weighted():
    d = _toy(1000)
    tab = pd.DataFrame({"g": [0.0, 0.0, 1.0, 1.0], "q": ["a", "b", "a", "b"], "eff": [0.0, 0.0, -0.02, 0.02]})
    orc, pred = Q.uplift(d.rename(columns={"pa": "p_x_a", "pb": "p_x_b"}), tab, "g", "q", ["a", "b"], "p_x_", K=2)
    g1 = d["g"].to_numpy() == 1
    assert np.allclose(orc[g1 & (d["q"].to_numpy() == "b")], 0.02) and np.allclose(orc[g1 & (d["q"].to_numpy() == "a")], -0.02)
    assert np.allclose(pred[g1], 0.3 * -0.02 + 0.7 * 0.02)         # 事前の確率で重み付けした期待値
    assert np.allclose(pred[~g1], 0.0)


def test_uplift_known_pre_equals_oracle():
    d = _toy(500)
    tab = pd.DataFrame({"g": [0.0, 1.0], "q": ["a", "a"], "eff": [0.01, -0.01]})
    orc, pred = Q.uplift(d, tab, "g", "q", ["a", "b"], None, K=2, known_pre=True)
    assert np.allclose(np.nan_to_num(orc), np.nan_to_num(pred))
