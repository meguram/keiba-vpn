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


def test_score_series_matches_scalar():
    ref, d = _ref()
    for key, col in (("ts", "ax_ts"), ("gear", "ax_gear"), ("endure", "ax_endure")):
        vals = pd.Series(list(d[col].head(25)) + [float("nan"), 200.0, -50.0])
        vec = s.score_series(key, vals, ref)
        scalar = [s.to_score10(key, v, ref) for v in vals]
        assert [None if np.isnan(x) else x for x in vec] == scalar
    assert s.score_series("gear", pd.Series([1.0]), None).isna().all()
    assert s.score_series("unknown_axis", pd.Series([1.0]), ref).isna().all()


def _metrics(gear):
    from src.research.race import running_profile as rp

    n = len(gear)
    m = pd.DataFrame({"race_id": [f"r{i}" for i in range(n)], "horse_id": ["h"] * n, "date": pd.date_range("2026-01-01", periods=n, freq="30D"),
                      "venue": "東京", "surface": "芝", "distance": 1600, "finish_position": 1, "ax_ts": [60.0 + i for i in range(n)], "ax_gear": gear})
    for c in ("ax_sustain", "ax_debut", "ax_endure", "ax_basic", "ax_wett", "ax_wetd", "ax_front", "ax_cr", "ax_makuri", "ax_slope",
              "bs_p", "bs_s", "wtT_p", "wtT_w", "wtD_p", "wtD_w"):
        m[c] = np.nan
    return m, pd.DataFrame({"horse_id": ["h"], "sire_id": ["s"]}), [c for c in m.columns if c.startswith("ax_")], rp


def test_history_scores_use_only_earlier_runs():
    """測れない→測れた→測れた→測れない→測れた の馬。各行の点が、その走より前の走だけで作った値（profiles_as_of）と一致する。
    最初に測れた走の行は履歴が無いので不明（その走自身の結果を入れない）。最後に測れた走より後の行は最新の値。"""
    from src.research.race import build_axis_scores10 as b

    ref, _ = _ref()
    m, sires, cols, rp = _metrics([np.nan, 10.0, 20.0, np.nan, 30.0, np.nan])
    table = rp.profile_table(m, sires, cols=cols)
    prof, as_of = b.latest_profiles(m, sires)
    hist = b.build_history(m, table, prof, ref).sort_values("date").reset_index(drop=True)
    priors = rp.priors_as_of(m, as_of, sires, cols)                     # 表と同じ（全期間の）引き寄せ先
    for i, d in enumerate(m["date"]):
        pr = rp.profiles_as_of(m, ["h"], d, sires, cols=cols, priors=priors).set_index("horse_id").loc["h"]
        want = s.to_score10("gear", pr["ax_gear"], ref) if pr["n_ax_gear"] > 0 else None
        got = hist.loc[i, "gear"]
        assert (want is None and np.isnan(got)) or got == want, (i, want, got)
    assert np.isnan(hist.loc[0, "gear"]) and np.isnan(hist.loc[1, "gear"])      # 履歴なし（2 行目は測れた最初の走）
    assert hist["n_prior_runs"].tolist() == [0, 1, 2, 3, 4, 5]                    # TS は全走で測れる


def test_build_latest_reflects_all_history_and_marks_unknown():
    from src.research.race import build_axis_scores10 as b

    ref, _ = _ref()
    m, sires, cols, rp = _metrics([np.nan, np.nan, np.nan])      # TS だけ値あり。ギアなど他の軸は自分の走が無い
    prof, as_of = b.latest_profiles(m, sires)
    out = b.build_latest(m, prof, as_of, ref)
    r = out.set_index("horse_id").loc["h"]
    assert out["as_of"].iloc[0] == m["date"].max() + pd.Timedelta(days=1)   # 最終日の翌日
    assert r["n_prior_runs"] == 3 and r["last_run_date"] == m["date"].max()
    assert r["ts"] == s.to_score10("ts", 61.5, ref)                       # 最新の走（62）まで入った上位 2 走（62・61）の平均
    assert np.isnan(r["gear"]) and not r["complete"]                      # 自分の走が無い軸は不明
