"""めぐコラムの枠・脚質の列と比較グラフ（src/research/columns/race_context.py）。"""

from __future__ import annotations

import pandas as pd

from src.research.columns import race_context as rc


def _runs() -> pd.DataFrame:
    # 1 頭（h1）が 4 走。各レースは 5 頭立てで、h1 の初角は 1, 5, 3, 1 番手
    rows = []
    for k, first in enumerate([1, 5, 3, 1]):
        rid = f"r{k}"
        for j in range(5):
            hid = "h1" if j == 0 else f"o{k}{j}"
            pos = first if j == 0 else [p for p in range(1, 6) if p != first][j - 1]
            rows.append({"race_id": rid, "date": f"2024-01-0{k + 1}", "horse_id": hid, "passing_order": f"{pos}-{pos}"})
    return pd.DataFrame(rows)


def test_prior_style_uses_only_earlier_runs():
    r = rc.positions(_runs())
    h = r[r["horse_id"] == "h1"].sort_values("date")
    assert h["q_now"].tolist() == [0.0, 1.0, 0.5, 0.0]
    assert pd.isna(h["q_prev"].iloc[0]) and h["style_pre"].iloc[0] is None        # 初出走は前走がない
    assert h["q_prev"].tolist()[1:] == [0.0, 0.5, 0.5]                          # 直前まで最大 3 走の平均（今走を含めない）
    assert h["style_pre"].tolist()[1:] == ["前寄り", "中団", "中団"]
    assert h["style_now"].tolist() == ["逃げ", "追込", "差し", "逃げ"]


def test_pair_chart_variants_and_p():
    d = pd.DataFrame({"x": ["a"] * 200 + ["b"] * 200, "f": ["p", "q"] * 200,
                      "v": [1.0] * 60 + [0.0] * 40 + [1.0] * 20 + [0.0] * 80 + [0.0] * 200})
    cond = pd.Series([True] * 100 + [False] * 100 + [True] * 100 + [False] * 100)
    ch = rc.pair_chart(d, cond, "x", ["a", "b"], "v", "A", "B", title="t", x_label="x", y_label="y", caption="c",
                       filters={"F": {"全体": None, "p": lambda w: w["f"] == "p"}})
    a, b = ch["series"]
    assert a["values"] == [0.6, 0.0] and b["values"] == [0.2, 0.0] and a["n"] == [100, 100]
    assert ch["compare"]["p"][0] < 0.001
    assert ch["filters"] == [{"label": "F", "options": ["全体", "p"]}] and list(ch["variants"]) == ["p"]
    assert ch["variants"]["p"]["series"][0]["n"] == [50, 50]
