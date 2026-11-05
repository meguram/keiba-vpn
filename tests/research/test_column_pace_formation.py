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


def test_parse_laps():
    assert pf.parse_laps("[12.3, 11.4, 0, -1, 12.0]") == [12.3, 11.4, 12.0]
    assert pf.parse_laps("12.3-11.4") == []
    assert pf.parse_laps(None) == []


def test_lap_series_blank_and_pad():
    dev = np.array([[0.1, -0.1, 0.0]] * 30) + np.arange(30)[:, None] * 0.001
    s = pf.lap_series("x", dev, 5)
    assert len(s["values"]) == 5 and s["values"][3:] == [None, None] and s["ci"][4] is None
    assert s["n"][:3] == [30, 30, 30]
    few = pf.lap_series("y", dev[:5], 5)
    assert few["values"] == [None] * 5


def test_lap_chart_pooled_and_variants():
    rng = np.random.default_rng(0)
    mats, rows = {}, []
    for key in ("東京芝1600", "中山芝1600", "東京芝2000"):
        n = 150
        idx = [f"{key}{i}" for i in range(n)]
        mats[key] = pd.DataFrame(12.0 + rng.normal(0, 0.2, (n, 8 if "1600" in key else 10)), index=idx)
        rows += [{"race_id": i, "nige_b": ["0頭", "1頭", "2頭以上"][j % 3], "prev_b": "2〜3頭", "rear_b": "0〜1頭"} for j, i in enumerate(idx)]
    d = pd.DataFrame(rows).set_index("race_id")
    ch = pf.lap_chart(d, mats)
    opts = ch["filters"][0]["options"]
    assert opts[0] == "芝1600（全場）" and "芝2000（全場）" not in opts            # 1 コースだけの距離は全場を作らない
    assert len(ch["xs"]) == 10
    v = ch["variants"][f"{opts[0]}|逃げ馬の頭数"]
    assert [s["name"] for s in v["series"]] == ["逃げ馬 0頭", "逃げ馬 1頭", "逃げ馬 2頭以上"]
    assert v["series"][0]["n"][0] == 100 and v["series"][0]["values"][8] is None
    assert len(ch["variants"]) == len(opts) * len(pf.LAP_AXES)
