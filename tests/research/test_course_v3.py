"""全レース版の第 3 版（2025 年を見ながら試行錯誤・2026 年は封印・根拠の検査）。"""

import numpy as np
import pandas as pd
import pytest

from src.research.course import v3 as V


def test_year_guards(tmp_path, monkeypatch):
    e = V.env3()
    assert e["FOLKLORE_YEAR_END"] == e["GRADED_YEAR_END"] == "2025" and V.YEAR_END == 2025 and V.EVAL_YEAR == 2025 and V.TRAIN_END == 2024
    monkeypatch.setattr(V, "OUT3", tmp_path)
    pd.DataFrame({"year": [2025, 2026], "race_id": ["a", "b"], "horse_id": ["x", "y"]}).to_parquet(tmp_path / "frame.parquet")
    pd.DataFrame({"__race_id": ["a", "b"], "__horse_id": ["x", "y"]}).to_parquet(tmp_path / "levels.parquet")
    with pytest.raises(ValueError):
        V.load3()                                                                      # 2026 年の行があれば止める
    import inspect
    assert "range(2016, YEAR_END + 1)" in inspect.getsource(V.load3)                   # 確定オッズの表も 2025 年まで


def test_hyp_types():
    assert V.hyp_ok(["A01"]) == (True, "能力")
    assert V.hyp_ok(["H09"])[0] is False and V.hyp_ok(["K06"])[0] is False and V.hyp_ok(["I02"])[0] is False
    assert V.hyp_ok(["A01", "D01"]) == (True, "能力×状態")
    assert V.hyp_ok(["E01", "D01"])[0] is False                                     # 展開 × 状態は 1 文で書けない組
    assert V.hyp_ok(["A01", "K02"])[0] is False                                     # 型なしを含む組


def test_direction():
    order = ["1着", "2-3着", "4-5着", "6-9着"]
    lv = pd.Series(["1着"] * 10 + ["2-3着"] * 10 + ["4-5着"] * 10 + ["6-9着"] * 10)
    y = np.array([1] * 8 + [0] * 2 + [1] * 6 + [0] * 4 + [1] * 4 + [0] * 6 + [1] * 2 + [0] * 8, float)
    assert V.direction(lv, y, order, "1着") == "ok"
    assert V.direction(lv, y, order, "2-3着") == "ng"                                # 間の水準
    y2 = y.copy()
    y2[10:20] = 1.0                                                                  # 2-3 着が 1 着より上（向きが揃わない）
    assert V.direction(lv, y2, order, "1着") == "ng"


def test_ev_pass_and_binom():
    c = V.C3()
    good = {"ev": {"hyp_ok": True, "dir": "ok", "stab": "ok", "breadth": "ok", "p_null": 0.01}}
    assert V.ev_pass(good, c) is None
    assert V.ev_pass(good | {"ev": good["ev"] | {"stab": "ng"}}, c) == "年ごとの安定性"
    assert V.ev_pass(good | {"ev": good["ev"] | {"p_null": 0.2}}, c) == "偶然の見込み"
    assert V.ev_pass(good | {"ev": good["ev"] | {"breadth": "不足"}}, V.C3(req_breadth=False)) is None
    assert V.binom_p(8, 10, 0.3) < 0.01 < V.binom_p(4, 10, 0.3)


def test_trimmed_roi():
    Y = np.zeros((6, 10))
    Y[:, 5] = 1
    Y[:, 6] = [10, 0, 0, 0, 0, 2]
    assert V.trimmed(Y, np.arange(6), 6, k=1) == pytest.approx(2 / 6)


def test_v3_html(tmp_path, monkeypatch):
    import json

    from src.scripts.docs import build_course_v3_html as H
    m = {"bet_races": 400, "coverage": 0.3, "win": 0.3, "rentai": 0.5, "place": 0.7, "roi_t": 1.25, "roi_f": 1.1, "roi_t_trim5": 1.05,
         "kill_precision": 0.9, "kill_coverage": 0.5}
    cyc = {"cycle": 1, "hyp": "<仮説>", "params": {"name": "基準"}, "review": "効いた", "decision": "採用", "goal_met": True, "n_buy": 10,
           "eval2025": {"reproduction": m, "optimistic": m | {"roi_t": 1.3}}, "train": {"reproduction": m}, "ev_summary": {"units": 3, "units_with_any": 2}}
    (tmp_path / "v3").mkdir()
    (tmp_path / "v3/course_v3_cycles.json").write_text(json.dumps({"cycles": [cyc], "adopted_cycle": 1, "note_looks": "2 回見ている"}), encoding="utf-8")
    kept = {"東京芝1600|1勝・2勝": {"buy": [{"types": ["A01"], "levels": ["1着"], "n_t": 20, "k_t": 15, "rt_t": 1.4,
                                           "ev": {"hyp": "能力", "dir": "ok", "stab": "ok", "breadth": "ok", "p_null": 0.001}}]}}
    (tmp_path / "v3/kept_1.json").write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(H, "BASE", tmp_path)
    h = H.build(tmp_path / "o.html").read_text(encoding="utf-8")
    assert "&lt;仮説&gt;" in h and "125.0%" in h and "上位 5 件を除く" in h and "A01=1着" in h and "能力" in h and "2 回見ている" in h


def test_hyp_ex():
    b = {"types": ["A01"], "levels": ["1着"], "n_t": 10, "k_t": 8, "w_t": 4, "rt_t": 1.2,
         "ev": {"hyp": "能力×人", "hyp_ok": True, "dir": "ok", "stab": "ok", "breadth": "ok", "p_null": 0.01}}
    cells = {"u": {"buy": [b], "kill": []}}
    assert len(V.select_cells(cells, V.C3())["u"]["buy"]) == 1
    assert V.select_cells(cells, V.C3(hyp_ex="能力"))["u"]["buy"] == []                 # 能力の入った組は外す


def test_race_side_filters():
    from tests.research.test_course_run import _toy
    d, L = _toy(n_races=6)
    d["field_decl"] = [16 if r.endswith(("0", "1", "2")) else 10 for r in d["race_id"]]
    kept = {"東京芝1600|1勝・2勝": {"buy": [{"types": ["A01"], "levels": ["1着"], "rate": 1.0, "win_c": 1.0}], "kill": []}}
    H, _, _ = V.pick(d, L, kept, V.C3(groups=3))
    assert len(H) == 6
    H2, _, _ = V.pick(d, L, kept, V.C3(groups=3, field_min=15))
    assert len(H2) == 3                                                               # 15 頭以上のレースだけ
    H3, _, _ = V.pick(d, L, kept, V.C3(groups=3, classes="新馬"))
    assert len(H3) == 0                                                               # 群が違う


def test_market_q_and_edge():
    d = pd.DataFrame({"race_id": ["a", "a", "b", "b"], "odds_final": [2.0, 2.0, 1.5, np.nan]})
    q = V.market_q(d)
    assert q[0] == pytest.approx(0.5) and q[2] == pytest.approx(1.0) and q[3] == 0.0
    b = {"types": ["A01"], "levels": ["1着"], "n_t": 10, "k_t": 8, "w_t": 4, "rt_t": 1.2, "q_t": 2.0,
         "ev": {"hyp": "能力", "hyp_ok": True, "dir": "ok", "stab": "ok", "breadth": "ok", "p_null": 0.01}}
    cells = {"u": {"buy": [b], "kill": []}}
    assert V.select_cells(cells, V.C3(edge_min=1.5, edge_pmax=0.2))["u"]["buy"][0]["rate"] == pytest.approx(0.8)
    assert V.select_cells(cells, V.C3(edge_min=1.5))["u"]["buy"] == []                 # 勝ち 4・見込み 2 は偶然の見込み 14% で 5% を超える
    assert V.edge_p(b) == pytest.approx(0.1429, abs=1e-3)
    assert V.select_cells(cells, V.C3(edge_min=2.5))["u"]["buy"] == []                 # 勝ち 4 ÷ 市場 2 = 2.0
    assert V.select_cells(cells, V.C3(weight="edge"))["u"]["buy"][0]["rate"] == pytest.approx(2.0)
