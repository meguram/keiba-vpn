import numpy as np
import pandas as pd

from src.scripts.data import final_week_course_eval as C


def test_course_band_edges():
    s = pd.Series(["芝", "芝", "芝", "芝", "ダート", "ダート", "ダート", "ダート"])
    d = pd.Series([1200, 1400, 2400, 3000, 1300, 1700, 1800, 2100])
    assert list(C.course_band(s, d)) == ["芝~1200", "芝1400-1600", "芝2200-2400", "芝2500~", "ダ1300-1600", "ダ1700-1800", "ダ1700-1800", "ダ1900~"]


def test_course_band_missing_and_boundaries():
    s = pd.Series(["芝", "芝", "芝", "ダート", "ダート", "ダート", "障害", "芝"])
    d = pd.Series([1000, 1201, 2300, 1150, 2000, np.nan, 3000, 0])
    out = C.course_band(s, d)
    assert list(out.iloc[:5]) == ["芝~1200", "芝1400-1600", "芝2200-2400", "ダ~1200", "ダ1900~"]
    assert out.iloc[5:].isna().all()


def test_wcorr_weighted_and_nonfinite():
    x, y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan, 9.0]), np.array([2.0, 4.0, 6.0, 8.0, 10.0, 1.0, 0.0])
    w = np.array([1.0, 1, 1, 1, 1, 1, 0.0])
    assert abs(C.wcorr(x, y, w) - 1) < 1e-9


def test_wcorr_perfect_and_degenerate():
    x = np.arange(10.0)
    assert abs(C.wcorr(x, 2 * x + 1, np.ones(10)) - 1) < 1e-9
    assert np.isnan(C.wcorr(x[:3], x[:3], np.ones(3)))


def _frame():
    rows = []
    for yr in (2020, 2021, 2023):
        for r in range(30):
            for h in range(8):
                rows.append({"race_id": f"{yr}-{r}", "year": yr, "surface": "芝", "distance": 1600 if r % 2 else 2000, "trainer_id": f"t{h % 2}", "trainer_name": "n",
                             "finish_position": h + 1, "pat": "栗東坂路|A2" if h < 4 else "美浦坂路|X"})
    d = pd.DataFrame(rows)
    d["win"], d["t3"], d["t2"] = (d.finish_position == 1).astype(int), (d.finish_position <= 3).astype(int), (d.finish_position <= 2).astype(int)
    d["fs"] = 8
    for c in ("win_ret", "place_ret", "win_odds", "p_win"):
        d[c] = np.nan
    return d


def test_with_course_prefixes_pattern_and_keeps_only_pattern():
    c = C.with_course(_frame())
    assert set(c["course"]) == {"芝1400-1600", "芝1800-2000"}
    assert (c["pat"].str.split("/", n=1).str[0] == c["course"]).all()
    assert (c["pat_only"] == c["pat"].str.split("/", n=1).str[1]).all()


def test_cell_table_lifts_and_reproduce_run():
    c = C.with_course(_frame())
    T = C.cell_table(c, 2020, 2021)
    assert set(T.index) == {f"{b}/{p}" for b in ("芝1400-1600", "芝1800-2000") for p in ("栗東坂路|A2", "美浦坂路|X")}
    assert (T.loc["芝1400-1600/栗東坂路|A2", "lift_top3"] > 1) and (T.loc["芝1400-1600/美浦坂路|X", "lift_top3"] < 1)
    q = C.predictive(c, (2020, 2021), (2023, 2023), 5, boot=10)
    assert len(q) > 0


def test_direction_b_by_course_counts():
    c = C.with_course(_frame())
    b = C.direction_b_by_course(c, 2020, 2025, nmin_t3=1)
    x = b[(b["course"] == "芝1400-1600") & (b["pat"] == "栗東坂路|A2")].iloc[0]
    assert x["3着以内の頭数"] == 45 * 3 and abs(x["全出走の構成比"] - 0.5) < 1e-9


def test_residual_corr_runs_and_bounded():
    c = C.with_course(_frame())
    r = C.residual_corr(c, (2020, 2021), (2023, 2023), 5)
    assert set(r["指標"]) == {"複勝率", "勝率"}


def _signal_frame(course_effect: bool):
    """4 パターン × 3 コース。パターンごとの強さは全コース共通で、course_effect=True のときだけ 芝1400-1600 でその効果が 2.5 倍になる。"""
    rows = []
    rng = np.random.default_rng(1)
    for yr in (2020, 2021, 2023):
        for r in range(600):
            dist = (1200, 1600, 2000)[r % 3]
            mult = 2.5 if (course_effect and dist == 1600) else 1.0
            strength = np.array([0.0, 0.3, 0.6, 0.9])[np.arange(8) % 4] * mult
            score = rng.normal(size=8) + strength
            fp = np.empty(8, int)
            fp[np.argsort(-score)] = np.arange(1, 9)
            for h in range(8):
                rows.append({"race_id": f"{yr}-{r}", "year": yr, "surface": "芝", "distance": dist, "trainer_id": f"t{h % 4}", "trainer_name": "n",
                             "finish_position": int(fp[h]), "pat": f"栗東坂路|P{h % 4}"})
    d = pd.DataFrame(rows)
    d["win"], d["t3"], d["t2"] = (d.finish_position == 1).astype(int), (d.finish_position <= 3).astype(int), (d.finish_position <= 2).astype(int)
    d["fs"] = 8
    for c in ("win_ret", "place_ret", "win_odds", "p_win"):
        d[c] = np.nan
    return d


def test_residual_detects_course_specific_effect_but_not_when_absent():
    on = C.residual_corr(C.with_course(_signal_frame(True)), (2020, 2021), (2023, 2023), 20).set_index("指標").loc["複勝率"]
    off = C.residual_corr(C.with_course(_signal_frame(False)), (2020, 2021), (2023, 2023), 20).set_index("指標").loc["複勝率"]
    assert on["共分散"] > 0 and on["相関"] > 0.5
    assert off["共分散"] < on["共分散"] / 3


def test_predictive_and_reproduce_have_values():
    c = C.with_course(_signal_frame(True))
    q = C.predictive(c, (2020, 2021), (2023, 2023), 20, boot=20)
    assert q["相関"].notna().all() and q["MSE"].notna().all() and len(q) == 2 * 5
    r = C.reproduce(c, (2020, 2021), (2023, 2023), 20)
    assert set(r["選び方"]) == {"セルの z", "パターン全体の z"} and set(r["前の期間での位置"]) == {"上位20%", "下位20%"}
    top, bot = r[(r["指標"] == "複勝率") & (r["選び方"] == "セルの z")].set_index("前の期間での位置").loc[["上位20%", "下位20%"], "後の期間のlift"]
    assert top > bot
