"""めぐコラム『逃げ馬の数と前走の位置取りで、ペースはどう変わるか』（src/research/columns/pace_formation.py）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.columns import pace_formation as pf


def _runs() -> pd.DataFrame:
    # 馬 h1 の 5 走: 初角 1,1,5,1,3 番手（頭数 10）。今走を含めず直近 3 走だけから作ることを確かめる
    firsts = [1, 1, 5, 1, 3]
    n = len(firsts)
    r = pd.DataFrame({"horse_id": ["h1"] * n, "date": [f"2025-01-0{i + 1}" for i in range(n)], "race_id": [f"r{i}" for i in range(n)],
                      "first": [float(f) for f in firsts], "starters": [10] * n})
    r["q"] = (r["first"] - 1) / (r["starters"] - 1)
    return r


def test_history_uses_only_previous_runs():
    h = pf.add_history(_runs())
    assert h["lead3"].isna().tolist() == [True, True, True, False, False]            # 直近 3 走がそろうのは 4 走目から
    assert h.loc[3, "lead3"] == 2 and h.loc[4, "lead3"] == 2                         # 4 走目: 1,1,5 → 2 回。5 走目: 1,5,1 → 2 回（今走は含めない）
    assert bool(h.loc[3, "is_nige"]) and bool(h.loc[4, "is_nige"])
    assert h.loc[4, "prev_first"] == 1 and bool(h.loc[4, "prev_front3"])
    assert h.loc[3, "rear3"] == 1                                                    # 5 番手（q=0.44）だけが差し・追込
    assert not h.loc[3, "is_rear"] and pd.isna(h.loc[2, "is_rear"])


def test_rear_requires_all_three_runs_behind_boundary():
    r = _runs()
    r["first"] = [6.0, 7.0, 8.0, 2.0, 2.0]
    r["q"] = (r["first"] - 1) / (r["starters"] - 1)
    h = pf.add_history(r)
    assert bool(h.loc[3, "is_rear"]) and not bool(h.loc[4, "is_rear"])               # 6,7,8 番手は全部後ろ。次は 7,8,2 で後ろでない走がある


def test_bins_cover_all_counts():
    for edges, labels in (pf.NIGE_BINS, pf.PREV_BINS, pf.REAR_BINS):
        top = 18 if edges[-1] == 99 else edges[-1]
        cut = pd.cut(pd.Series(np.arange(0, top + 1)), edges, labels=labels)
        assert cut.notna().all()


def test_rate_series_blanks_small_cells():
    d = pd.DataFrame({"x": ["a"] * 40 + ["b"] * 5, "pace_cls": ["high"] * 20 + ["slow"] * 20 + ["high"] * 5})
    hi, lo = pf.rate_series(d, "x", ["a", "b"])
    assert hi["values"][0] == 0.5 and lo["values"][0] == 0.5
    assert hi["values"][1] is None and hi["ci"][1] is None                           # 5 レースの点は出さない
