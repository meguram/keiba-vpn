"""重賞と通常レースのルールの一覧の HTML（src/scripts/docs/build_rules_overview_html.py）。"""

import json
import re

from src.scripts.docs import build_rules_overview_html as B


def _h(fin, tan=0, fuku=0):
    return {"hn": 1, "name": "x", "odds": 3.0, "pop": 1, "fin": fin, "tan": tan, "fuku": fuku}


def _race(date, fin, tan=0, kill_fins=(5, 6)):
    return {"race_id": date.replace("-", ""), "date": date, "pay_ok": True, "base": _h(2),
            "B": {"honmei": _h(fin, tan, 150 if fin <= 3 else 0), "kill": [_h(f) for f in kill_fins]}}


def _history():
    rule = {"rule": {"filter": None, "terms": [{"text": "「前走の着順」の点（A01）が高いほど上", "w": 1.0, "eff": 0.6, "key": "x_A01"}], "tie": "同点は馬番"},
            "label": "1×A01", "k": 2, "default": False, "dayof": False}
    races = [_race("2018-05-01", 1, 500), _race("2019-05-01", 1, 400), _race("2025-05-01", 1, 300), _race("2026-05-01", 4, 0, (2, 7))]
    return {"periods": {"dev": [2018, 2024], "eval": [2025, 2026]}, "notes": [], "data_until": "2026-11-08", "frozen_at": "2026-11-15",
            "rkeys": {"テスト記念": {"rkey": "テスト記念", "grade": "G3", "venue": "東京", "course": "東京芝1600", "honmei": rule, "kill": rule, "races": races}}}


def test_wilson_and_flags():
    lo, hi = B.wilson(5, 10)
    assert 0.2 < lo < 0.5 < hi < 0.8
    assert B.wilson(0, 0) is None
    tr = {"bet": 100, "win": 0.6, "roi_t": 2.0}
    f = B.flags(tr, {"bet": 20, "win": 0.1, "roi_t": 1.2}, {"bet": 20, "win": 0.1, "roi_t": 1.1})
    assert f["over"] and f["gap"] and not f["few"]                                     # 区間が重ならない・両年とも 100% 超
    f2 = B.flags(tr, {"bet": 2, "win": 0.5, "roi_t": 1.5}, {"bet": 0, "win": None, "roi_t": None})
    assert not f2["over"] and f2["few"]                                                # 2026 年に買っていなければ「テストでも」にしない


def test_graded_rows_periods():
    H = _history()
    rows = B.graded_rows(H)
    r = rows[0]
    assert r["p"]["train"]["bet"] == 2 and r["p"]["train"]["win"] == 1.0 and abs(r["p"]["train"]["roi_t"] - 4.5) < 1e-9
    assert r["p"]["2025"]["bet"] == 1 and abs(r["p"]["2025"]["roi_t"] - 3.0) < 1e-9
    assert r["p"]["2026"]["win"] == 0 and r["p"]["2026"]["kill_precision"] == 0.5        # 消した 2 頭のうち 1 頭が 2 着
    assert r["venue"] == "東京" and r["surf"] == "芝" and r["dist"] == "1600"
    assert r["flags"]["test"]["bet"] == 2 and abs(r["flags"]["test"]["roi_t"] - 1.5) < 1e-9   # 2 年を合わせた値は回の記録から直接
    r2 = B.graded_rows(H, {"rkeys": {"テスト記念": [{"id": "v1"}]}})[0]
    assert "新しい版が 1 個" in r2["version"] and "最初の版" in r2["version"]            # 成績は最初の版のまま。版の名前を取り違えない
    t = B.graded_total(H)
    assert t["train"]["bet"] == 2 and t["2026"]["bet"] == 1


def test_build(tmp_path, monkeypatch):
    out3 = tmp_path / "course" / "v3"
    out3.mkdir(parents=True)
    unit = "東京芝1600|1勝・2勝"
    m = {"races": 10, "bet_races": 4, "coverage": 0.4, "win": 0.5, "rentai": 0.5, "place": 0.75, "roi_t": 1.2, "roi_f": 1.0, "roi_t_trim5": 0.0,
         "kill_precision": 0.9, "kill_coverage": 0.5, "killed": 10}
    v = {"no_filter": m, "optimistic": m, "reproduction": m}
    U = {"cycle": 17, "sha256_cells": "abc", "made_at": "x", "type_names": {"A01": "前走の着順"},
         "values": {k: {"units": {unit: v}, "total": v} for k in ("train", "2025", "2026")}}
    (out3 / "per_unit_v3.json").write_text(json.dumps(U), encoding="utf-8")
    kept = {unit: {"layer": "主", "buy": [{"types": ["A01"], "levels": ["1着"], "rate": 1.5, "n_t": 5, "k_t": 4, "w_t": 3, "rt_t": 2.0,
                                          "ev": {"hyp": "能力", "dir": "ok", "stab": "ok", "breadth": "ok", "p_null": 0.01}}],
                   "kill": [{"types": ["A01"], "levels": ["10着以下"], "n": 30, "ev": {"hyp": "能力"}}]}}
    (out3 / "kept_17.json").write_text(json.dumps(kept), encoding="utf-8")
    (out3 / "course_v3_cycles.json").write_text(json.dumps({"cycles": [{"cycle": 17, "event": "事象の文", "basis": "根拠の文"}]}), encoding="utf-8")
    prereg = tmp_path / "p.json"
    prereg.write_text(json.dumps({"cycle": 17, "sha256_cells": "abc", "cells_file": "kept_17.json", "params": {"name": "n"}, "frozen_at": "f"}), encoding="utf-8")
    hist = tmp_path / "history.json"
    hist.write_text(json.dumps(_history()), encoding="utf-8")
    monkeypatch.setattr(B, "COURSE_OUT", tmp_path / "course")
    monkeypatch.setattr(B, "PREREG3", prereg)
    monkeypatch.setattr(B, "LEDGER", tmp_path / "none.json")
    out = B.build(tmp_path / "o.html", history=hist)
    s = out.read_text(encoding="utf-8")
    D = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', s, re.S).group(1).replace("<\\/", "</"))
    assert D["graded"]["rows"][0]["unit"] == "テスト記念" and D["course"]["rows"][0]["unit"] == unit
    assert "「前走の着順」が 1着" in D["course"]["rows"][0]["buy"][0]["text"]
    assert D["course"]["event"] == "事象の文" and "prefers-color-scheme" in s and "<script src" not in s
    U["sha256_cells"] = "zzz"                                                           # 凍結と違う数え直しは使わない
    (out3 / "per_unit_v3.json").write_text(json.dumps(U), encoding="utf-8")
    try:
        B.build(tmp_path / "o2.html", history=hist)
        raise AssertionError("止まるはず")
    except SystemExit:
        pass
