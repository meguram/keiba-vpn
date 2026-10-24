import numpy as np
import pandas as pd

from src.scripts.data import tp74_pos as T


def test_band4_thresholds():
    s = T.band4(pd.Series([0.0, 0.15, 0.16, 0.35, 0.36, 0.65, 0.66, np.nan]))
    assert s.tolist()[:7] == [0, 0, 1, 1, 2, 2, 3] and np.isnan(s.iloc[7])


def test_bh_controls_false_discoveries():
    assert T.bh([0.001, 0.02, 0.5, 0.9]) == [True, True, False, False]
    assert T.bh([0.3, 0.6]) == [False, False]


def test_cell_stats_cluster_se_and_mean():
    d = pd.DataFrame({"k": [0, 0, 0, 0, 1, 1], "rid": [1, 1, 2, 2, 3, 3], "v": [1.0, 0.0, 1.0, 0.0, 1.0, 1.0]})
    t = T.cell_stats(d, ["k"], "v").set_index("k")
    assert abs(t.loc[0, "mean"] - 0.5) < 1e-9 and t.loc[0, "N"] == 4 and t.loc[0, "races"] == 2
    assert t.loc[1, "se"] == 0.0


def test_chain_shrinkage_pulls_small_cells_to_parent():
    d = pd.DataFrame({"bp": [0] * 300 + [0] * 3, "c": ["a"] * 300 + ["b"] * 3, "v": [0.0] * 300 + [1.0] * 3})
    kl, tb = T.fit_chain(d.assign(rid=range(len(d))), "bp", ("c",), "v", 0.0, k=200.0)
    off = T.apply_chain(pd.DataFrame({"bp": [0, 0], "c": ["a", "b"]}), kl, tb)
    assert 0.0 <= off[1] < 0.05          # 素の平均は 1.0 だが、3 出走の b は親（≈0.006）へ強く引き戻される
