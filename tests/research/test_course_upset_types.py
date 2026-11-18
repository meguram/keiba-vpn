"""穴馬が勝つレースの型（src/research/course/upset_types.py）。合成の表だけで確かめる（実データの結果は見ない）。"""

import json

import numpy as np
import pandas as pd
import pytest

from src.research.course import upset_types as UT


def _synth(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for y in range(2018, 2025):
        for r in range(40):
            n = 16 if r % 3 == 0 else 10
            cls = [0, 2, 5, 6][r % 4]
            odds = np.sort(rng.uniform(1.5, 80, n))
            fin = rng.permutation(np.arange(1, n + 1))
            for h in range(n):
                rows.append({"race_id": f"{y}{r:03d}", "horse_id": f"H{y}{r}{h}", "year": y, "yr": y, "clsn": float(cls), "grade": "G3" if cls == 6 else "1勝",
                             "odds_final": odds[h], "pop_final": h + 1, "fin": fin[h], "date": f"{y}-06-{1 + r % 20:02d}", "venue": "東京",
                             "field": n, "field_decl": n, "weight_rule": "定量", "hn": h + 1, "p_race_id": None, "jockey_id": "J", "wt": 470,
                             "jw": 55, "p_jw": 55, "p_grade": "G3", "p_fin": 3, "p_margin": 0.3, "p_l3rank": 5, "p_pos1": 5, "p_pos4": 5,
                             "age": 4, "interval": 40, "dist": 1600, "p_dist": 1600, "bracket_number": 1, "prev_region": None,
                             "race_class": "サラ系２歳 未勝利" if r % 5 == 0 else "サラ系３歳以上 １勝クラス"})
    return pd.DataFrame(rows)


def test_counts_and_evaluate_smoke():
    C = json.loads(UT.CONFIG.read_text(encoding="utf-8")) | {"bootstrap": {"reps": 10, "seed": 1}}
    d = _synth()
    c = UT.counts(d, C)
    assert {"R2/course", "R2/graded", "R3/course", "R4/course"} <= set(c) and c["R2/course"]["採否"]["type_races"] > 0
    assert sum(1 for k in c if k.startswith("段2")) == 11
    res = UT.evaluate(d, C)
    assert res["stage1"]["n_tests"] == 2 and res["stage2"]["n_tests"] == 10
    assert all("checks" in r for r in res["stage1"]["rows"] if not r["control"])
    low = [r for r in res["stage1"]["rows"] if not r["control"] and not r["in_bh"]]
    assert low and all(not r["candidate"] and r["q"] is None for r in low)                 # 検出力の足りない組は BH の外


def test_expected_sums_to_one_and_longshot():
    C = json.loads(UT.CONFIG.read_text(encoding="utf-8"))
    d = UT.prepare(_synth(), C)
    s = d.groupby("race_id")["p"].sum()
    assert np.allclose(s, 1.0)                                                             # 見込みはレースの中で和 1
    r = UT.race_table(d, C)
    assert ((r["e"] > 0) & (r["e"] < 1)).all() and set(r["y"].unique()) <= {0.0, 1.0}


def test_run_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(UT, "MARK", tmp_path / "m.json")
    monkeypatch.setattr(UT, "OUTU", tmp_path)
    UT.MARK.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="1 回だけ"):
        UT.cmd_run()


def test_two_year_old_fullwidth_and_stage2_follows_stage1_controls():
    C = json.loads(UT.CONFIG.read_text(encoding="utf-8"))
    d = UT.prepare(_synth(), C)
    assert d["two"].any() and not d["two"].all()                                          # 全角の「２歳」を拾う
    rows = [{"id": "U01", "face": "graded", "control": False, "in_bh": True,
             "res": {"採否": {"mean": -0.1, "se": 0.01, "n": 500}, "確かめ": {"mean": -0.08, "n": 300}, "2024": {"ci": (-0.2, -0.01)}}}]
    ok = UT.judge([dict(r) for r in rows], C, "2")
    assert ok["candidates"] == ["U01/graded"]
    bad = UT.judge([dict(r) for r in rows], C, "2", broken_from=True)
    assert bad["candidates"] == [] and bad["null_broken"]                                 # 段 1 の負の対照が通れば、段 2 も使わない


def test_upset_run_refuses_after_prereg_changed(tmp_path, monkeypatch):
    monkeypatch.setattr(UT, "MARK", tmp_path / "u.json")
    monkeypatch.setattr(UT, "OUTU", tmp_path)
    monkeypatch.setattr(UT.PR, "_git_clean", lambda paths: True)
    monkeypatch.setattr(UT.PR, "MARK", tmp_path / "p.json")
    UT.PR.MARK.write_text(json.dumps({"sha256": {UT.CONFIG.name: "old"}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="§13 を流した後"):
        UT.cmd_run()
