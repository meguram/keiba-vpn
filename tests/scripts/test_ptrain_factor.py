import numpy as np
import pandas as pd

from src.scripts.data import ptrain_factor as pf


def test_eb_shrink_moves_small_samples_to_line_mean():
    g = pd.DataFrame({"sire": ["a", "b", "c", "d"], "line": ["L", "L", "L", "L"], "n": [400, 30, 400, 30], "ybar": [1800.0, 1400.0, 2000.0, 2400.0], "var": [90000.0] * 4})
    d = pf.eb_shrink(g).set_index("sire")
    lm = d["theta_line"].iloc[0]
    assert abs(d.loc["b", "theta"] - lm) < abs(d.loc["b", "ybar"] - lm)     # 少数の種牡馬ほど系統の平均へ寄る
    assert abs(d.loc["a", "delta"]) < abs(1800.0 - lm) + 1e-9


def test_fisher_enrich_finds_shared_ancestor():
    P = {f"l{i}": {"X": 3} for i in range(8)} | {f"h{i}": {"Y": 3} for i in range(8)}
    e = pf.fisher_enrich(P, [f"l{i}" for i in range(8)], [f"h{i}" for i in range(8)], min_support=6).set_index("anc")
    assert e.loc["X", "p"] < 0.01 and e.loc["X", "in_low"] == 8
