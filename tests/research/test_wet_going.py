"""道悪適性（走法マップの道悪の地図の軸）の 1 走ごとの材料 wet_going と、地図の切り替え（mode_axes）。"""

import numpy as np
import pandas as pd

from src.research.race import running_map as rm
from src.research.race import wet_going as wg


def _runs():
    rows = []
    for rid, surf, cond, n in [("a", "芝", "稍重", 6), ("b", "ダート", "良", 6), ("c", "芝", "良", 4), ("d", "芝", "不明", 6)]:
        for i in range(n):
            rows.append({"race_id": rid, "horse_id": f"h{i}", "surface": surf, "track_condition": cond, "finish_position": i + 1})
    return pd.DataFrame(rows)


def test_compute_splits_by_surface_and_level_and_skips_small_fields_and_unknown_condition():
    o = wg.compute(_runs()).set_index(["race_id", "horse_id"])
    a = o.loc[("a", "h0")]
    assert a["wtT_p"] == 1.0 and a["wtT_w"] == 1 and np.isnan(a["wtD_p"])
    assert o.loc[("a", "h5")]["wtT_p"] == -1.0
    b = o.loc[("b", "h0")]
    assert b["wtD_w"] == 0 and np.isnan(b["wtT_p"])
    assert o.loc[("c", "h0")][wg.OUT_COLS].isna().all()      # 5 頭未満
    assert o.loc[("d", "h0")][wg.OUT_COLS].isna().all()      # 馬場状態が読めない


def test_mode_axes_swaps_basic_for_the_wet_axis_by_surface():
    gk, gi, gw, gs = rm.mode_axes(False)
    assert "basic" in gk and not {"wett", "wetd"} & set(gk) and len(gk) == 6
    for surf, wk in (("芝", "wett"), ("ダート", "wetd")):
        k, i, w, s = rm.mode_axes(True, surf)
        assert wk in k and "basic" not in k and len(k) == 6
        assert w[rm.AXIS_KEYS.index(wk)] == 1 and w[rm.AXIS_KEYS.index("basic")] == 0
        assert rm.AXIS_KEYS.index(wk) in s and rm.AXIS_KEYS.index("basic") in s   # 基礎スピードは馬の詳細に残す
    assert not {rm.AXIS_KEYS.index("wett"), rm.AXIS_KEYS.index("wetd")} & set(gs)
