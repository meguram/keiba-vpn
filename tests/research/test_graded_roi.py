"""重賞のルールベース予想家: 払戻の解き方・回収率の件数・回収率で選ぶ版の採否（2 回目の評価）。"""

import json

import numpy as np
import pandas as pd

from src.research.graded import payout as PO
from src.research.graded import roi as R
from src.research.graded import tally as TL


def test_parse_win_place_and_dead_heat():
    js = {"単勝": {"numbers": "3", "payout": "1,450"}, "複勝": [{"numbers": "3", "payout": "160"}, {"numbers": "5", "payout": "200"}]}
    tan, fuku = PO.parse_win_place(json.dumps(js, ensure_ascii=False))
    assert tan == {3: 14.5} and fuku == {3: 1.6, 5: 2.0}
    # 同着で 2 頭とも別の欄（list）なら両方に付く
    js["単勝"] = [{"numbers": "3", "payout": "700"}, {"numbers": "5", "payout": "900"}]
    tan, _ = PO.parse_win_place(json.dumps(js, ensure_ascii=False))
    assert tan == {3: 7.0, 5: 9.0}
    # 2 頭がつながった表記（馬番 1315・払戻 1,480150）は分けられないので、そのレースは解けない扱い
    js["単勝"] = {"numbers": "1315", "payout": "1,480150"}
    assert PO.parse_win_place(json.dumps(js, ensure_ascii=False)) is None
    assert PO.parse_win_place("not json") is None
    # つながった表記が 1〜18 に収まる場合（1 番と 5 番の同着が「15」）も、単勝の馬が複勝に無いので解けない扱い
    js["単勝"] = {"numbers": "15", "payout": "200410"}
    js["複勝"] = [{"numbers": "1", "payout": "110"}, {"numbers": "5", "payout": "150"}, {"numbers": "7", "payout": "300"}]
    assert PO.parse_win_place(json.dumps(js, ensure_ascii=False)) is None


def test_attach_returns_caps_and_unparsed_race():
    d = pd.DataFrame({"race_id": ["A", "A", "A", "B"], "hn": [1, 2, 3, 1]})
    hits = pd.DataFrame({"race_id": ["A", "A"], "hn": [1, 2], "tan": [80.0, 0.0], "fuku": [12.0, 1.5]})
    out = PO.attach_returns(d, hits, pd.Index(["A"]))
    assert out["ret_tan"].tolist() == [80.0, 0.0, 0.0, 0.0]
    assert out["ret_tan_c"].tolist()[0] == PO.CAP_TAN and out["ret_fuku_c"].tolist()[0] == PO.CAP_FUKU
    assert out["ret_fuku"].tolist()[1] == 1.5
    assert out["pay_ok"].tolist() == [True, True, True, False]


def _frame():
    # 同じコース・同じレース名の重賞を 3 日、各 4 頭。水準は 1 頭目だけ「a」
    rows = []
    for k, dt in enumerate(["2020-01-01", "2021-01-01", "2022-01-01"]):
        for h in range(4):
            rows.append({"race_id": f"R{k}", "dt": pd.Timestamp(dt), "rkey": "X", "course": "C", "grade": "G2", "cls": 6, "fin": h + 1,
                         "field": 4, "ret_tan": 5.0 if h == 0 else 0.0, "ret_fuku": 2.0 if h < 3 else 0.0, "ret_tan_c": 5.0 if h == 0 else 0.0,
                         "ret_fuku_c": 2.0 if h < 3 else 0.0, "pay_ok": k != 1})
    d = pd.DataFrame(rows)
    lv = pd.Series(np.where(d.index % 4 == 0, "a", "b"), index=d.index)
    return d, lv


def test_tally_returns_are_asof_and_skip_unparsed_races():
    d, lv = _frame()
    b = TL.base_frame(d)
    t = TL.tally_type(b, lv, np.ones(len(b), dtype=bool))
    last = t.loc[8]                        # 3 日目の 1 頭目（水準 a）。前の 2 日のうち払戻が解けたのは 1 日目だけ
    assert last["main_n"] == 2 and last["main_np"] == 1 and last["main_rt"] == 5.0 and last["main_rf"] == 2.0
    first = t.loc[0]
    assert first["main_n"] == 0 and first["main_rt"] == 0


def test_tally_without_return_columns_is_zero():
    d, lv = _frame()
    d = d.drop(columns=["ret_tan", "ret_fuku", "ret_tan_c", "ret_fuku_c", "pay_ok"])
    t = TL.tally_type(TL.base_frame(d), lv, np.ones(len(d), dtype=bool))
    assert (t["main_rt"] == 0).all() and (t["main_np"] == 0).all()


def _unit(dir_, n=100, y3=40, e3=30, roi_f=1.0, q_lo=0.9, q_hi=1.1, roi_fc=1.0, roi_t=1.0, roi_tc=1.0):
    u = {"dir": dir_, "n": n, "y3": y3, "e3": e3, "roi_ret_fuku": roi_f, "roi_ret_fuku_c": roi_fc, "roi_ret_tan": roi_t, "roi_ret_tan_c": roi_tc}
    for c in ("ret_fuku", "ret_tan"):
        for q in (25, 50, 100):
            u[f"q{q:03d}_{c}"] = q_lo
        for q in (900, 950, 975):
            u[f"q{q:03d}_{c}"] = q_hi
    return pd.Series(u)


def test_passes_needs_both_periods_roi_and_performance():
    rho = {"ret_fuku": 0.8, "ret_fuku_c": 0.78, "ret_tan": 0.75, "ret_tan_c": 0.7}
    v = dict(R.PRIMARY)
    good = _unit("好", q_lo=0.85)
    assert R.passes(good, good, v, rho, rho)
    assert not R.passes(good, _unit("好", y3=30, q_lo=0.85), v, rho, rho)   # 期待ちょうどは通さない
    assert not R.passes(good, _unit("好", q_lo=0.7), v, rho, rho)          # 確認で下限が全体を下回る
    assert not R.passes(good, None, v, rho, rho)                            # 確認に当てはまる馬がいない
    assert not R.passes(good, _unit("好", n=10, q_lo=0.85), v, rho, rho)    # 件数の下限
    assert not R.passes(good, _unit("好", y3=20, q_lo=0.85), v, rho, rho)   # 回収率は良いが成績が期待を下回る
    assert not R.passes(good, _unit("好", q_lo=0.85, roi_fc=0.7), v, rho, rho)          # 頭打ちにすると向きが変わる
    assert R.passes(good, _unit("好", q_lo=0.85, roi_fc=0.7), v | {"cap": "頭打ちを見ない"}, rho, rho)
    bad = _unit("凡", y3=20, q_hi=0.7, roi_fc=0.5)
    assert R.passes(bad, bad, v, rho, rho)
    assert not R.passes(bad, _unit("凡", y3=20, q_hi=0.9, roi_fc=0.5), v, rho, rho)
    # 点＋0.10: 好は 0.8 + 0.10 以上
    vp = v | {"crit": "点＋0.10"}
    assert R.passes(_unit("好", roi_f=0.95), _unit("好", roi_f=0.91), vp, rho, rho)
    assert not R.passes(_unit("好", roi_f=0.95), _unit("好", roi_f=0.85), vp, rho, rho)
    # 縮めた点: 5 走で 1.2 は m=30 だと (6 + 24) / 35 = 0.857 < 0.9 で落ち、m=5 だと (6 + 4) / 10 = 1.0 で通る
    small = _unit("好", n=5, y3=3, e3=1, roi_f=1.2)
    assert not R.passes(small, small, v | {"crit": "縮めた点＋0.10（m=30）", "nmin": 3}, rho, rho)
    assert R.passes(small, small, v | {"crit": "縮めた点＋0.10（m=5）", "nmin": 3}, rho, rho)
    # 両方: 単勝も満たす必要がある
    assert not R.passes(good, good, v | {"bet": "両方"}, rho, {**rho, "ret_tan": 0.95})


def test_filtered_keeps_rows_and_zeroes_other_patterns():
    S = pd.DataFrame({"row": [0, 1, 2, 3], "type": ["A", "A", "A", "B"], "level": ["a", "a", "b", "a"], "x": [0.3, -0.2, 0.5, 0.4],
                      "layer": ["主", "主", "副", "主"]})
    f = R.filtered(S, [["A", "a", "好"], ["A", "b", "好"]], ("主",))
    assert len(f) == 4
    assert f["x"].tolist() == [0.3, 0.0, 0.0, 0.0]    # 向きが違う・層が版に入らない・別の型は 0
    f = R.filtered(S, [["A", "a", "好"], ["A", "b", "好"]], ("主", "副"))
    assert f["x"].tolist() == [0.3, 0.0, 0.5, 0.0]


def test_unit_stats_and_overall():
    inst = pd.DataFrame({"row": range(6), "type": "A", "level": "a", "dir": "好", "race_id": ["r1", "r1", "r2", "r2", "r3", "r3"],
                         "y3": [1, 0, 1, 0, 0, 0], "e3": 0.5, "ret_tan": [4.0, 0, 0, 0, 0, 0], "ret_fuku": [2.0, 0, 1.5, 0, 0, 0],
                         "ret_tan_c": [4.0, 0, 0, 0, 0, 0], "ret_fuku_c": [2.0, 0, 1.5, 0, 0, 0]})
    ridx, W = R.boot_weights(inst["race_id"], n_boot=500)
    u = R.unit_stats(inst, ridx, W).iloc[0]
    assert u["n"] == 6 and u["races"] == 3 and np.isclose(u["roi_ret_fuku"], 3.5 / 6)
    assert u["q025_ret_fuku"] <= u["roi_ret_fuku"] <= u["q975_ret_fuku"]
    rows = inst.assign(pay_ok=[True] * 4 + [False] * 2)
    o = R.overall_roi(rows)
    assert np.isclose(o["ret_fuku"], 3.5 / 4) and o["runners"] == 4


def test_versions_grid_and_primary_is_in_grid():
    vs = R.all_versions()
    assert len(vs) == np.prod([len(v) for v in R.GRID.values()]) + np.prod([len(v) for v in R.SMALL_GRID.values()])
    assert sum(v == R.PRIMARY | {"family": R.FAMILY_BIG} for v in vs) == 1
    assert len({R.version_name(v) for v in vs}) == len(vs)


def test_marks_for_empty_version_is_none_and_ties():
    assert R.marks_for(pd.DataFrame(), {"used": [], "patterns": [], "params": R.PRIMARY}) is None
    S = pd.DataFrame({"row": [0, 1], "race_id": "r", "type": "A", "level": "a", "x": [0.3, -0.1], "layer": "主", "main_edhit": 0,
                      "hn": [1, 2], "y3": [1, 0], "y1": [0, 0], "year": 2025})
    rec = {"used": ["A"], "patterns": [["A", "b", "好"]], "params": R.PRIMARY}
    assert R.marks_for(S, rec) is None                      # 残ったパターンが当たらない版
    rec["patterns"] = [["A", "a", "好"]]
    assert R.marks_for(S, rec, pd.Series({0: 0.0, 1: 0.0})).sort_values("rank")["hn"].tolist() == [1, 2]
    # 凡だけが残る版: 同点（点 0）の馬は成績版の点で並ぶ。「馬番」の版は馬番の順
    S3 = pd.DataFrame({"row": [0, 1, 2], "race_id": "r", "type": "A", "level": ["b", "b", "c"], "x": [0.2, 0.1, -0.3], "layer": "主",
                       "main_edhit": 0, "hn": [1, 2, 3], "y3": 0, "y1": 0, "year": 2025})
    rec = {"used": ["A"], "patterns": [["A", "c", "凡"]], "params": R.PRIMARY}
    assert R.marks_for(S3, rec, pd.Series({0: 0.1, 1: 0.9, 2: 2.0})).sort_values("rank")["hn"].tolist() == [2, 1, 3]
    rec["params"] = R.PRIMARY | {"tie": "馬番"}
    assert R.marks_for(S3, rec).sort_values("rank")["hn"].tolist() == [1, 2, 3]
    mk = pd.DataFrame({"race_id": ["a", "a", "b", "b"], "rank": [1, 2, 1, 2], "score": [0.0, 0.0, 0.5, 0.1]})
    t = R.tie_share(mk)
    assert t["tie_top"] == 0.5 and t["zero_top"] == 0.5
