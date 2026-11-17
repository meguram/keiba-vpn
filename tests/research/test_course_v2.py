"""全レース版の第 2 版（時間で分けて採る版）。"""

import inspect

import numpy as np
import pandas as pd
import pytest

from src.research.course import run as CR
from src.research.course import v2 as V
from tests.research.test_course_run import _toy

U = "東京芝1600|1勝・2勝"


def _cells():
    return {U: {"buy": [{"types": ["A01"], "levels": ["1着"], "n": 10, "k": 10, "rate": 1.0},
                        {"types": ["C01"], "levels": ["同距離"], "n": 10, "k": 3, "rate": 0.3}],
                "kill": [{"types": ["A01"], "levels": ["2-3着"], "n": 30}]}}


def test_confirm_keeps_only_cells_that_hold_in_confirm():
    d, L = _toy(n_races=12)
    unit = np.array([U] * len(d))
    kept, log = V.confirm_cells(_cells(), d, L, unit, V.P2(c_min=5, k_min=10, kill_prec=0.7))
    assert [c["types"] for c in kept[U]["buy"]] == [["A01"]]                    # 1 着の馬（確認でも 100% 3 着内・単回収 500%）だけ残る
    assert kept[U]["buy"][0]["n_c"] == 12 and kept[U]["buy"][0]["rate"] == 1.0
    assert log["buy_found"] == 2 and log["buy_kept"] == 1
    assert kept[U]["kill"] and kept[U]["kill"][0]["prec_c"] == pytest.approx(1 - 24 / 108)
    kept2, log2 = V.confirm_cells(_cells(), d, L, unit, V.P2(c_min=50))
    assert kept2[U]["buy"] == [] and log2["buy_small"] == 1                       # 1 着の条件（12 走）は確認の走数が下限に満たない
    kept3, _ = V.confirm_cells(_cells(), d, L, unit, V.P2(kill_prec=0.95))
    assert kept3[U]["kill"] == []                                                 # 消しは 4 着以下の割合が基準に届かない


def test_score_reproduction_and_filters():
    d, L = _toy(n_races=12)
    d["odds_final"] = np.where(d["hn"] == 1, 1.5, 20.0)
    d["pop_final"] = np.where(d["hn"] == 1, 1, 5)
    unit = np.array([U] * len(d))
    kept, _ = V.confirm_cells(_cells(), d, L, unit, V.P2())
    shift = {"lr": [[0.0], [0.0], [0.0], [0.0]]}                                  # ずれ 0 → 再現は楽観と同じ
    s = V.score(d, L, unit, kept, V.P2(odds_min=2.0), shift)
    assert s["no_filter"]["bet_races"] == 12 and s["optimistic"]["bet_races"] == 0 and s["reproduction"]["bet_races"] == 0
    shift2 = {"lr": [[-0.7], [0.0], [0.0], [0.0]]}                                # 1 番人気は確定で下がる → 発走前は 1.5 × e^0.7 ≈ 3.0 倍
    s2 = V.score(d, L, unit, kept, V.P2(odds_min=2.0), shift2)
    assert s2["optimistic"]["bet_races"] == 0 and s2["reproduction"]["bet_races"] == 12
    s3 = V.score(d, L, unit, kept, V.P2(), shift)
    assert s3["optimistic"] == s3["reproduction"] == s3["no_filter"]             # 絞り込みの無い版は 3 つが同じ
    real = {(r, 1): (3.0, 1.0) for r in d["race_id"].unique()[:4]}
    s4 = V.score(d, L, unit, kept, V.P2(odds_min=2.0), shift, real_pre=real)
    assert s4["real_t60"]["bet_races"] == 4 and s4["real_t60"]["races"] == 4 and s4["real_t60_optimistic"]["bet_races"] == 0


def test_simulate_pre_rank_and_missing():
    d = pd.DataFrame({"race_id": ["a"] * 3, "odds_final": [2.0, 5.0, np.nan], "pop_final": [1, 2, np.nan]})
    pre, pop = V.simulate_pre(d, {"lr": [[0.0], [0.0], [0.0], [0.0]]}, 0)
    assert list(pre[:2]) == [2.0, 5.0] and np.isnan(pre[2]) and list(pop[:2]) == [1, 2]


def test_ev_filter_uses_confirm_win_rate():
    keep = V.odds_filter(V.P2(ev_min=1.0), np.array([3.0, 3.0, np.nan]), np.array([1, 2, 3]), np.array([0.4, 0.3, 0.9]))
    assert list(keep) == [True, False, False]


def test_periods_and_guards(tmp_path, monkeypatch):
    src = inspect.getsource(V.load_split)
    assert "yr <= DISC_END" in src and "CR.YEAR_END" in src and V.DISC_END == 2021 and V.CONF == (2022, 2024)
    assert "frame_and_levels" in inspect.getsource(V._eval_inner)
    monkeypatch.setattr(V, "PREREG2", tmp_path / "p.json")
    monkeypatch.setattr(V, "EVAL_MARK2", tmp_path / "e.json")
    V.PREREG2.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        V.cmd_cycles()                                                            # 凍結の後は流し直さない
    monkeypatch.delenv("COURSE_EVAL_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        V._eval_inner()                                                           # トークンが無ければ流さない


def test_disc_params_scale_and_cycles():
    p = V.P2().disc_params(0.66)
    assert (p.min_races, p.kill_n, p.n_min) == (7, 13, 2)
    assert len(V.CYCLES) == 20 and all(set(c["delta"]) <= set(V.P2.__dataclass_fields__) for c in V.CYCLES)
    assert CR.YEAR_END == 2024


def test_real_t60_denominator_and_no_filter():
    d, L = _toy(n_races=12)
    d["odds_final"] = np.where(d["hn"] == 1, 1.5, 20.0)
    d["pop_final"] = np.where(d["hn"] == 1, 1, 5)
    unit = np.array([U] * len(d))
    kept, _ = V.confirm_cells(_cells(), d, L, unit, V.P2())
    real = {(r, 1): (3.0, 1.0) for r in d["race_id"].unique()[:4]} | {("他のレース", 1): (3.0, 1.0)}
    s = V.score(d, L, unit, kept, V.P2(), {"lr": [[0.0]] * 4}, real_pre=real)
    assert s["real_t60"]["races"] == 4 and s["real_t60"]["bet_races"] == 4           # 表に無いレースは分母に入れない・絞り込み無しでも出す


def test_neighbors_and_adopt_key():
    assert [v.odds_min for v in V.neighbors(V.P2(odds_min=2.0))] == [1.6, 2.4]
    assert V.neighbors(V.P2()) == []
    small = {"conf": {"reproduction": {"bet_races": 10, "roi_t": 3.0}}}
    big = {"conf": {"reproduction": {"bet_races": 600, "roi_t": 1.01}}}
    assert V.adopt_key(big) > V.adopt_key(small) and not V.qualified(small)


def test_shift_cells_fallback():
    shift = {"lr": [[0.0], [0.0], [0.0], [0.0]], "all": [0.0], "min_pairs": 2,
             "lr_cell": [[] for _ in range(16)]}
    shift["lr_cell"][int(V.cell_of(np.array([1]), np.array([1.5]))[0])] = [-0.7, -0.7]
    d = pd.DataFrame({"race_id": ["a", "a"], "odds_final": [1.5, 20.0], "pop_final": [1, 5]})
    pre, _ = V.simulate_pre(d, shift, 0)
    assert pre[0] == pytest.approx(3.0, abs=0.05) and pre[1] == 20.0               # 組の足りない帯は人気帯の分布（ずれ 0）


def test_confirm_sub_layer_and_unmatched():
    d, L = _toy(n_races=12)
    d2, L2 = _toy(n_races=12, course="東京芝1600")
    d2["race_id"] = d2["race_id"] + "b"
    d2["group"] = "3勝・OP"
    d, L = pd.concat([d, d2], ignore_index=True), pd.concat([L, L2], ignore_index=True)
    unit = (d["course"] + "|" + d["group"]).to_numpy()
    cells = {"東京芝1600|3勝・OP": {"layer": "副", "buy": [{"types": ["A01"], "levels": ["1着"]}, {"types": ["A01"], "levels": ["無い"]}], "kill": []}}
    kept, log = V.confirm_cells(cells, d, L, unit, V.P2(c_min=20))
    assert kept["東京芝1600|3勝・OP"]["buy"][0]["n_c"] == 24 and log["sub_units"] == 1  # 副は同じコースの全群で数える
    assert log["buy_unmatched"] == 1 and log["buy_small"] == 0


def test_freeze_guards(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(V, "OUT2", tmp_path)
    monkeypatch.setattr(V, "PREREG2", tmp_path / "p.json")
    monkeypatch.setattr(V, "EVAL_MARK2", tmp_path / "e.json")
    (tmp_path / "shift.json").write_text("{}", encoding="utf-8")
    log = {"finished_at": "x", "best_cycle": 1, "shift_sha": CR._sha_file(tmp_path / "shift.json"),
           "cycles": [{"cycle": 1, "params": {"a": 1}, "qualified": False}]}
    (tmp_path / "course_v2_cycles.json").write_text(json.dumps(log), encoding="utf-8")
    (tmp_path / "course_v2_best.json").write_text(json.dumps({"params": {"a": 1}, "kept": {}}), encoding="utf-8")
    with pytest.raises(SystemExit):
        V.cmd_freeze()                                                                # 下限に届かない版は凍結しない
    log["cycles"][0]["qualified"] = True
    log["shift_sha"] = "other"
    (tmp_path / "course_v2_cycles.json").write_text(json.dumps(log), encoding="utf-8")
    with pytest.raises(SystemExit):
        V.cmd_freeze()                                                                # ずれの分布がサイクルのときと違う
    with pytest.raises(SystemExit):
        V.cmd_shift()                                                                 # サイクルの後に作り直さない
