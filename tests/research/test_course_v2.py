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


def test_kill_rank_and_adopt_tie_break():
    a = {"kill_precision": 0.96, "kill_coverage": 0.01}
    b = {"kill_precision": 0.99, "kill_coverage": 0.005}
    c = {"kill_precision": 0.90, "kill_coverage": 0.5}
    a, b = a | {"killed": 20000}, b | {"killed": 20000}
    assert V.kill_rank(a) > V.kill_rank(b) > V.kill_rank(c)                       # 下限 95% 以上の中で広い方。届かない版は後ろ
    ra = {"conf": {"reproduction": {"bet_races": 600, "roi_t": 1.5} | a}}
    rb = {"conf": {"reproduction": {"bet_races": 600, "roi_t": 1.5} | b}}
    assert V.adopt_key(ra) == V.adopt_key(rb)                                     # 採否は ◎ だけ（消しは cmd_kill_pick で選ぶ）
    assert V.kill_rank({"kill_precision": None, "kill_coverage": None}) == (False, 0.0, 0.0)


def test_v2_html(tmp_path, monkeypatch):
    import json

    from src.scripts.docs import build_course_v2_html as H
    m = {"bet_races": 600, "coverage": 0.5, "win": 0.4, "rentai": 0.6, "place": 0.8, "roi_t": 1.5, "roi_f": 1.2, "kill_precision": 0.97, "kill_coverage": 0.3}
    cyc = {"cycle": 1, "hyp": "<基準>", "adopted": True, "qualified": True, "params": {"name": "x"},
           "conf": {"reproduction": m, "optimistic": m | {"roi_t": 1.7}}, "disc": {"reproduction": m}}
    (tmp_path / "v2").mkdir()
    (tmp_path / "v2/course_v2_cycles.json").write_text(json.dumps({"cycles": [cyc], "best_cycle": 1, "note_conf": "高めに出る"}), encoding="utf-8")
    monkeypatch.setattr(H, "BASE", tmp_path)
    h = H.build(tmp_path / "o.html").read_text(encoding="utf-8")
    assert "&lt;基準&gt;" in h and "◎ 連対率" in h and "60.0%" in h and "楽観" in h and "高めに出る" in h


def test_metrics_rentai():
    from src.research.course import search as SE
    d, _ = _toy(n_races=4)
    Y = SE.y_matrix(d)
    race = d["race_id"].astype(str).to_numpy()
    m = SE.metrics(Y, race, np.array([0, 1]), np.array([], int), 4)     # 1 着の馬と、2〜10 着のどれか
    assert m["rentai"] == pytest.approx((1 + (d["fin"].iloc[1] <= 2)) / 2) and m["win"] == 0.5


def _log_files(tmp_path, monkeypatch, r_conf, r_disc):
    import json
    monkeypatch.setattr(V, "OUT2", tmp_path)
    monkeypatch.setattr(V, "PREREG2", tmp_path / "p.json")
    monkeypatch.setattr(V, "EVAL_MARK2", tmp_path / "e.json")
    (tmp_path / "shift.json").write_text("{}", encoding="utf-8")
    params = {k: v for k, v in V.P2().__dict__.items()} | {"name": V.P2().name()}
    cyc = {"cycle": 1, "params": params, "qualified": True, "adopted": True, "conf": r_conf, "disc": r_disc}
    log = {"finished_at": "x", "best_cycle": 1, "shift_sha": CR._sha_file(tmp_path / "shift.json"), "cycles": [cyc]}
    (tmp_path / "course_v2_cycles.json").write_text(json.dumps(log), encoding="utf-8")
    (tmp_path / "course_v2_best.json").write_text(json.dumps({"params": params, "cycle": 1, "kept": {}}), encoding="utf-8")
    monkeypatch.setattr(V, "load_split", lambda: (None, None, None, None, 1.0))
    monkeypatch.setattr(CR, "wait_resources", lambda: None)


def test_recount_and_kill_pick_flow(tmp_path, monkeypatch):
    import json
    m = {"bet_races": 600, "win": 0.4, "roi_t": 1.5, "kill_precision": 0.99, "kill_coverage": 0.01, "killed": 50}
    rec = {"no_filter": m, "optimistic": m, "reproduction": m}
    _log_files(tmp_path, monkeypatch, rec, rec)
    new = {"no_filter": m | {"rentai": 0.6}, "optimistic": m | {"rentai": 0.6}, "reproduction": m | {"rentai": 0.6}}

    def fake_run(p, *a):
        k = m | ({"kill_precision": 0.97, "kill_coverage": 0.8, "killed": 5000} if p.kill_depth == 2 else {})
        r = {"no_filter": k | {"rentai": 0.6}, "optimistic": k | {"rentai": 0.6}, "reproduction": k | {"rentai": 0.6}}
        return {"params": p.__dict__ | {"name": p.name()}, "conf": r, "disc": new, "kept": {}, "confirm_log": {}, "seconds": 0, "mem": {}}
    monkeypatch.setattr(V, "run_one", fake_run)
    V.cmd_recount()
    log = json.loads((tmp_path / "course_v2_cycles.json").read_text(encoding="utf-8"))
    assert log["cycles"][0]["conf"]["reproduction"]["rentai"] == 0.6 and log.get("recounted_at")
    log["cycles"].append(log["cycles"][0] | {"cycle": 2, "adopted": False, "params": log["cycles"][0]["params"] | {"kill_depth": 2}})
    (tmp_path / "course_v2_cycles.json").write_text(json.dumps(log), encoding="utf-8")
    V.cmd_kill_pick()
    log = json.loads((tmp_path / "course_v2_cycles.json").read_text(encoding="utf-8"))
    assert log["best_cycle"] == 3 and log["cycles"][-1]["kill_pick"]["selected"]["kill_depth"] == 2
    with pytest.raises(SystemExit):
        V.cmd_kill_pick()                                                             # 2 回目は止まる
    with pytest.raises(SystemExit):
        V.cmd_recount()                                                               # 選び直しの後は数え直さない
    V.cmd_freeze()                                                                    # 凍結の照らし（版・qualified・shift）がつながる
    assert json.loads(V.PREREG2.read_text(encoding="utf-8"))["best_cycle"] == 3


def test_recount_stops_on_mismatch(tmp_path, monkeypatch):
    m = {"bet_races": 600, "win": 0.4, "roi_t": 1.5, "kill_precision": 0.99, "kill_coverage": 0.01}
    rec = {"no_filter": m, "optimistic": m, "reproduction": m}
    _log_files(tmp_path, monkeypatch, rec, rec)
    bad = {"no_filter": m, "optimistic": m, "reproduction": m | {"kill_coverage": 0.02}}
    monkeypatch.setattr(V, "run_one", lambda p, *a: {"conf": bad, "disc": rec})
    with pytest.raises(SystemExit):
        V.cmd_recount()


def test_kill_pick_stops_if_honmei_changes(tmp_path, monkeypatch):
    m = {"bet_races": 600, "win": 0.4, "roi_t": 1.5, "kill_precision": 0.99, "kill_coverage": 0.01, "killed": 50}
    rec = {"no_filter": m, "optimistic": m, "reproduction": m}
    _log_files(tmp_path, monkeypatch, rec, rec)
    monkeypatch.setattr(V, "run_one", lambda p, *a: {"params": {}, "conf": {"reproduction": m | {"win": 0.5}}, "disc": rec})
    with pytest.raises(SystemExit):
        V.cmd_kill_pick()
