"""めぐコラム『去勢明けは買い？』の N 戦目の数え方（src/research/columns/gelding_return.py）。"""

from __future__ import annotations

import pandas as pd

from src.research.columns import gelding_return as gr


def test_steps_count_from_first_gelded_run():
    sexes = ["牡", "牡", "牡", "牡", "セ", "セ", "セ", "セ", "セ", "セ", "セ"]
    d = pd.DataFrame({"horse_id": "h", "race_id": [f"r{i:02d}" for i in range(11)], "date": [f"2024-01-{i + 1:02d}" for i in range(11)],
                      "sex": sexes, "prev_sex": [None] + sexes[:-1], "age": 4, "top3": 0.0, "popularity": 1, "starters": 10})
    other = pd.DataFrame({"horse_id": "o", "race_id": ["x1", "x2"], "date": ["2024-01-01", "2024-01-02"], "sex": "セ", "prev_sex": [None, "セ"],
                          "age": 4, "top3": 1.0, "popularity": 1, "starters": 10})          # 最初から セ の馬は対象外
    f = gr.frame(pd.concat([d, other], ignore_index=True))
    assert set(f["horse_id"]) == {"h"}
    assert f.sort_values("date")["step"].tolist() == gr.STEPS[:8] + [gr.STEPS[8]] * 2      # 4 走前は去勢前 3 走前より古いので外れる
    assert int(f["g_age"].iloc[0]) == 4 and int(f["n_after_total"].iloc[0]) == 7
