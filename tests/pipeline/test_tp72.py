import numpy as np
import pandas as pd

from src.scripts.data.tp72_data import add_pincer, asof_merge
from src.scripts.data.tp72_eval import bh, cluster_z


def test_asof_merge_excludes_same_day_and_future():
    left = pd.DataFrame({"k": ["a", "a", "a", "b"], "date": pd.to_datetime(["2024-01-10", "2024-01-10", "2024-01-20", "2024-01-20"])})
    src = pd.DataFrame({"k": ["a", "a", "b"], "date": pd.to_datetime(["2024-01-05", "2024-01-10", "2024-01-25"]), "one": [1.0, 1.0, 1.0]})
    out = asof_merge(left.copy(), src, ["k"], ["one"], "h_")
    assert out["h_one"].tolist()[:2] == [1.0, 1.0]       # 1/10 の行は 1/5 までの 1 件だけ（同日の 1/10 は含めない）
    assert out["h_one"].iloc[2] == 2.0                    # 1/20 は 1/5 と 1/10 の 2 件
    assert np.isnan(out["h_one"].iloc[3])                 # b の出走は 1/25（未来）しか無い


def test_pincer_neighbors_use_runner_order_and_edges():
    d = pd.DataFrame({
        "race_id": ["r"] * 5, "horse_number": [1, 2, 4, 5, 6], "date": pd.to_datetime(["2024-01-01"] * 5),
        "prev_pos1": [9, 2, 8, 3, 7], "prev_posL": [9, 2, 8, 3, 7],
    })
    out = add_pincer(d).set_index("horse_number")
    assert out.loc[4, "d1_pin"] == 1 and out.loc[4, "d1_self_nl"] == 1      # 2 番と 5 番（取消の 3 番を飛ばした隣）が前走 3 番手以内
    assert out.loc[6, "d1_edge"] == 1 and out.loc[1, "d1_in1"] == 1         # 外端は内の隣、内端は外の隣だけを見る
    assert out.loc[2, "d1_self_nl"] == 0                                     # 自分が前走先行なら対象外


def test_cluster_z_and_bh():
    d = pd.DataFrame({"rc": [0, 0, 1, 1], "e": [0.1, 0.1, 0.1, 0.1]})
    z, tot, n = cluster_z(d, np.ones(4, bool), "e", 2)
    assert n == 4 and abs(tot - 0.4) < 1e-12 and abs(z - np.sqrt(2)) < 1e-12
    q = bh(np.array([0.001, 0.02, 0.5]))
    assert q[0] <= q[1] <= q[2] and q.max() <= 1.0
