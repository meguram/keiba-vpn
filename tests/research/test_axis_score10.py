"""走法軸の 10 点換算（axis_score10.py）。合成データで、アンカー・基準点・向き・不明・丸めを確かめる。"""

import numpy as np
import pandas as pd

from src.research.race import axis_score10 as s


def _frame(seed=0):
    rng = np.random.default_rng(seed)
    # クラスごとの TS 平均（単調に上がる）。タイプ軸 gear/endure はクラスと無関係
    mu = {"未勝利": 52, "1勝": 57, "2勝": 60.5, "OP": 62, "G3": 63, "G2": 64, "G1": 66}
    rows = []
    for g, m in mu.items():
        n = 400
        rows.append(pd.DataFrame({"grade": g, "ax_ts": rng.normal(m, 4.5, n), "ax_gear": rng.normal(54, 7, n), "ax_endure": rng.normal(58, 4, n)}))
    d = pd.concat(rows, ignore_index=True)
    d["race_id"], d["horse_id"] = np.arange(len(d)).astype(str), "h"
    return d


def _ref(d=None):
    d = _frame() if d is None else d
    cols = {"ts": "ax_ts", "gear": "ax_gear", "endure": "ax_endure"}
    return s.build_score10(d, pd.Series(1, index=d.index), d["grade"], cols, {"ts": 1, "gear": 1, "endure": -1}), d


def test_ts_anchors_monotone_and_class_points():
    ref, d = _ref()
    knots = ref["ts"]["knots"]
    assert all(a[0] < b[0] and a[1] < b[1] for a, b in zip(knots, knots[1:]))   # TS も点も単調増加
    med = lambda g: float(d.loc[d.grade == g, "ax_ts"].median())
    assert abs(s.to_score10("ts", med("2勝"), ref) - 5.0) <= 0.5
    assert abs(s.to_score10("ts", med("1勝"), ref) - 4.0) <= 0.5
    assert abs(s.to_score10("ts", med("未勝利"), ref) - 3.0) <= 0.5


def test_ts_top_is_ten_and_clipped():
    ref, d = _ref()
    q95 = float(d.loc[d.grade == "G1", "ax_ts"].quantile(0.95))
    assert s.to_score10("ts", q95, ref) == 10.0
    assert s.to_score10("ts", 200, ref) == 10.0
    assert s.to_score10("ts", -50, ref) == 1.0


def test_type_axis_center_and_direction():
    ref, d = _ref()
    a = ref["axes"]["gear"]
    assert s.to_score10("gear", a["mu"], ref) == 5.0                      # 2勝の好走走の平均 = 5
    assert s.to_score10("gear", a["mu"] + a["sd"], ref) > 5.0
    e = ref["axes"]["endure"]
    assert s.to_score10("endure", e["mu"] + e["sd"], ref) < 5.0           # 向き -1 は値が大きいほど低い


def test_k_matches_ts_spacing():
    ref, d = _ref()
    # 3→5 の 2 点分が TS の約 (60.5-52)/4.5 ≒ 1.9 SD → k ≒ 1.06（範囲 1.0〜2.5 の中）
    assert 1.0 <= ref["k"] <= 1.3


def test_unknown_and_rounding():
    ref, d = _ref()
    assert s.to_score10("gear", None, ref) is None
    assert s.to_score10("gear", float("nan"), ref) is None
    assert s.to_score10("unknown_axis", 1.0, ref) is None
    assert s.to_score10("ts", 60, None) is None
    v = s.to_score10("gear", ref["axes"]["gear"]["mu"] + 0.3 * ref["axes"]["gear"]["sd"], ref)
    assert (v * 2) == round(v * 2)                                        # 0.5 点刻み


def test_deviation_axis_uses_same_width():
    ref, _ = _ref()
    assert s.to_score10_deviation(50, ref) == 5.0
    assert s.to_score10_deviation(60, ref) > 5.0 and s.to_score10_deviation(None, ref) is None


def test_no_grades_or_small_sample_gives_none():
    d = _frame().head(20)
    ref, _ = _ref(d)
    assert ref is None


def test_pav_flattens_inversion():
    out = s._pav(np.array([1.0, 3.0, 2.0, 4.0]), np.ones(4))
    assert list(out) == [1.0, 2.5, 2.5, 4.0]


def test_peak_ts_as_of_uses_only_past_and_top2():
    m = pd.DataFrame({"race_id": ["a", "b", "c", "d"], "horse_id": ["h"] * 4,
                      "date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"]), "ax_ts": [50.0, 70.0, 60.0, 99.0]})
    p = s.peak_ts_as_of(m, ["h", "none"], "2026-04-01")            # 4/1 の走は含めない（as_of より前だけ）
    assert p["h"] == 65.0                                          # 上位 2 走 (70, 60) の平均
    assert np.isnan(p["none"])                                     # 走が無い馬は不明


def test_peak_ts_runs_includes_the_run_itself():
    m = pd.DataFrame({"race_id": ["a", "b"], "horse_id": ["h"] * 2, "date": pd.to_datetime(["2026-01-01", "2026-02-01"]), "ax_ts": [60.0, 70.0]})
    r = s.peak_ts_runs(m)
    assert r["peak_ts"].tolist() == [55.0, 65.0]                   # 1 走目は (60+50)/2（足りない分は仮の値）、2 走目は (70+60)/2


def test_upper_classes_beyond_anchor_level_are_interpolated():
    d = _frame()
    ref, _ = _ref(d)
    # アンカーは 5.5 以下のクラスだけ。OP 以上の中央値は節点にしない
    assert max(y for _x, y in ref["ts"]["knots"]) == 10.0
    assert set(ref["ts"]["medians"]) <= {"新馬", "未勝利", "1勝", "2勝", "3勝"}
