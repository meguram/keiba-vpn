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
