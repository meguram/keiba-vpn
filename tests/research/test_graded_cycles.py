"""重賞のルールベース予想家の分析サイクル（src/research/graded/cycles.py）と、その HTML（build_graded_cycles_html）のテスト。"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from src.research.graded import cycles as CY
from src.scripts.docs import build_graded_cycles_html as B


def _table():
    # 2 レース（2024 年 = 開発、2025 年 = 評価）× 7 頭。特徴: 前日版の点と型 1 つ
    rows = []
    for rid, yr, rk in (("r24", 2024, "R"), ("r25", 2025, "R")):
        for h in range(7):
            rows.append({"race_id": rid, "rkey": rk, "grade": "G3", "year": yr, "hn": h + 1, "fin": h + 1, "y1": float(h == 0), "y3": float(h < 3),
                         "pay_ok": True, "ret_tan": 5.0 if h == 0 else 0.0, "ret_fuku": 2.0 if h < 3 else 0.0, "score_pre": float(7 - h)})
    r = pd.DataFrame(rows)
    F = np.column_stack([r["score_pre"], np.zeros(len(r)), np.zeros(len(r)), np.zeros(len(r)), -r["score_pre"]])
    return CY.Table(r, F, ["score_pre", "score_day", "top3_pre", "top5_pre", "x_A01"])


def test_honmei_picks_max_and_breaks_ties_by_preday_then_number():
    t = _table()
    W = np.zeros((5, 3), np.float32)
    W[0, 0] = 1                      # 前日版の点 → 1 番
    W[4, 1] = 1                      # 逆向き → 7 番
    H = CY.honmei(t, W)              # 3 列目は全馬 0 の同点 → 前日版の点の高い 1 番
    assert t.r["hn"].to_numpy()[H[:, 0]].tolist() == [1, 1]
    assert t.r["hn"].to_numpy()[H[:, 1]].tolist() == [7, 7]
    assert t.r["hn"].to_numpy()[H[:, 2]].tolist() == [1, 1]


def test_killed_takes_lowest_and_keeps_three():
    t = _table()
    W = np.zeros((5, 1), np.float32)
    W[0, 0] = 1
    nk, nb = CY.killed(t, W, 3)
    assert nk[:, 0].tolist() == [3, 3] and nb[:, 0].tolist() == [3, 3]      # 下位 3 頭（5〜7 着）はすべて 4 着以下
    nk6, nb6 = CY.killed(t, W, 6)
    assert nk6[:, 0].tolist() == [4, 4] and nb6[:, 0].tolist() == [4, 4]   # 7 頭なので 3 頭は残す


def test_best_selects_on_eval_for_A_and_dev_for_B():
    t = _table()
    bst = CY.Best(t)
    W = np.zeros((5, 2), np.float32)
    W[0, 0] = 1
    W[4, 1] = 1
    bst.update(W, ["前日版", "逆向き"])
    rep = bst.report()
    assert rep["n_rkeys"] == 1 and rep["tried"] == 2
    a, b = rep["A"]["honmei_by_rkey"], rep["B"]["honmei_by_rkey"]
    assert a["all"]["win"] == 1.0 and a["all"]["roi_t"] == 5.0 and a["achieved"] == 1 and a["per_rkey"][0]["rule"] == "前日版"
    assert b["all"]["win"] == 1.0                                    # 開発でも前日版が勝つので同じ
    assert rep["A"]["kill_by_rkey"]["all"]["precision"] == 1.0
    assert CY.goal_met(rep)["goal_rkey"] and CY.goal_met(rep)["sub_rkey"]


def test_objective():
    assert CY.objective(np.array([0.3]), np.array([1.2]))[0] == 1.0
    assert CY.objective(np.array([0.15]), np.array([2.4]))[0] == 0.5


def test_cycles_html(tmp_path):
    t = _table()
    bst = CY.Best(t)
    W = np.zeros((5, 1), np.float32)
    W[0, 0] = 1
    bst.update(W, ["前日版"])
    rep = bst.report()
    L = {"note": "テスト", "goal": {"win": 0.3, "roi_t": 1.2}, "sub_goal": {"precision": 0.95, "killed_per_race": 3}, "races": 2, "rows": 14,
         "n_features": 5, "cycles": [{"cycle": 1, "at": "2026-11-15T16:00:00+09:00", "hypothesis": "基準 <x>", "n_new": 1, "met": CY.goal_met(rep), "report": rep}],
         "stopped": {"cycle": 1, "reason": "テスト"}}
    src = tmp_path / "cycles.json"
    src.write_text(json.dumps(L, ensure_ascii=False), encoding="utf-8")
    h = B.build(src, tmp_path / "c.html").read_text(encoding="utf-8")
    assert "<svg" in h and not re.search(r"<script[^>]+src=", h) and "prefers-color-scheme:dark" in h
    assert "基準 &lt;x&gt;" in h and "A と B の差" in h and 'href="graded_handicapper.html"' in h


def test_B_never_uses_eval_to_break_ties():
    """開発の値が同じ 2 つの決め方で、評価だけ違うとき、B は先に見つけた方を残す（評価の漏れが無い）。"""
    t = _table()
    ev = (t.r["race_id"] == "r25").to_numpy()
    extra = np.where(ev, -t.r["score_pre"].to_numpy(), t.r["score_pre"].to_numpy())[:, None]
    t2 = CY.Table(t.r, np.column_stack([t.F, extra]), t.names + ["x_B01"])
    bst = CY.Best(t2)
    W = np.zeros((6, 2), np.float32)
    W[5, 0] = 1      # 開発では 1 番（勝ち）、評価では 7 番（負け）
    W[0, 1] = 1      # 開発でも評価でも 1 番（勝ち）
    bst.update(W, ["先", "後"])
    rep = bst.report()
    assert rep["B"]["honmei_by_rkey"]["per_rkey"][0]["rule"] == "先" and rep["B"]["honmei_by_rkey"]["all"]["win"] == 0.0
    assert rep["A"]["honmei_by_rkey"]["per_rkey"][0]["rule"] == "後"


def test_handicapper_links_to_cycles(tmp_path):
    from src.scripts.docs import build_graded_handicapper_html as OLD
    src = tmp_path / "r.json"
    src.write_text(json.dumps({"prereg": {"frozen_at": "x"}, "races": [], "eval_races": []}), encoding="utf-8")
    assert 'href="graded_cycles.html"' in OLD.build(src, tmp_path / "h.html").read_text(encoding="utf-8")


def test_apply_rule_matches_update_and_kills_weaker_on_ties():
    t = _table()
    terms = {"前日版": {"score_pre": 1.0}, "同点": {"score_day": 1.0}}
    W = np.stack([CY._vec(t, terms["前日版"]), CY._vec(t, terms["同点"])], axis=1)
    H = CY.honmei(t, W)
    for j, lab in enumerate(["前日版", "同点"]):
        r = CY.apply_rule(t, CY.rule_of(lab, terms), np.ones(len(t.starts), bool))["races"]
        assert [x["hn"] for x in r] == t.r["hn"].to_numpy()[H[:, j]].tolist()
    # 全馬が同点の決め方で消すと、並びの後ろ（前日版の点が低い 7・6・5 番）から消す
    r = CY.apply_rule(t, CY.rule_of("同点・下位 3 頭", terms), np.ones(len(t.starts), bool))["races"]
    assert sorted(r[0]["killed"]) == [5, 6, 7] and r[0]["bad"] == 3
    nk, nb = CY.killed(t, W[:, [1]], 3)
    assert nb[:, 0].tolist() == [3, 3]


def test_cycles_hash_covers_track_from_and_features():
    from src.research.graded.run import _cycles_hash
    body = {"rules": {"A": 1}, "track_from": "2026-11-16", "features": ["a"]}
    h = _cycles_hash(body)
    assert _cycles_hash(body | {"track_from": "2026-01-01"}) != h and _cycles_hash(body | {"features": ["b"]}) != h
