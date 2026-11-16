"""全レース版（src/research/course/）のテスト。2025・2026 年の表を読まない作り・クラスの群・条件の探索と当てはめ。"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from src.research.course import run as CR
from src.research.course import search as SE


@pytest.fixture()
def year_end_2024(monkeypatch):
    """GRADED_YEAR_END・FOLKLORE_YEAR_END=2024 で features・payout・folklore を読み直し、終わったら（失敗しても）元に戻す。"""
    import src.research.graded.features as F
    import src.research.graded.payout as PO
    import src.scripts.data.folklore_build as FB
    monkeypatch.setenv("GRADED_YEAR_END", "2024")
    monkeypatch.setenv("FOLKLORE_YEAR_END", "2024")
    mods = [importlib.reload(F), importlib.reload(PO), importlib.reload(FB)]
    try:
        yield mods
    finally:
        monkeypatch.delenv("GRADED_YEAR_END", raising=False)
        monkeypatch.delenv("FOLKLORE_YEAR_END", raising=False)
        for m in (F, PO, FB):
            importlib.reload(m)


def test_env_limits_years_and_drops_entry_rows(monkeypatch):
    monkeypatch.setenv("FOLKLORE_ENTRIES", "/x.parquet")
    monkeypatch.setenv("FOLKLORE_CUTOFF", "2026-11-22")
    e = CR.env()
    assert e["FOLKLORE_YEAR_END"] == "2024" and e["GRADED_YEAR_END"] == "2024"
    assert "FOLKLORE_ENTRIES" not in e and "FOLKLORE_CUTOFF" not in e   # 出馬表の段階の行（2026 年）を足さない


def test_features_and_payout_read_only_up_to_year_end(year_end_2024, monkeypatch, tmp_path):
    """GRADED_YEAR_END=2024 のとき、frame・払戻が読む表の年は 2024 まで（2025・2026 年のファイルを開かない）。"""
    F, PO, FB = year_end_2024
    assert max(F.YEARS) == 2024 and max(PO.YEARS) == 2024 and max(FB.YEARS) == 2024
    tb = tmp_path / "tables"
    for y in range(2016, 2027):   # どの年にも表がある状態で、どれを開くかを見る
        (tb / str(y)).mkdir(parents=True)
        for n in ("race_result_flat.parquet", "race_result_race.parquet", "race_shutuba_past_flat.parquet"):
            (tb / str(y) / n).write_bytes(b"")
    monkeypatch.setattr(F, "TABLES", tb)
    opened = []

    def spy(path, columns=None, **k):   # 全部の年を開かせて、開いた年を数える（中身は空）
        opened.append(str(path))
        return pd.DataFrame({c: pd.Series(dtype=object) for c in (columns or ["race_id"])})
    monkeypatch.setattr(pd, "read_parquet", spy)
    for fn in (F.entry_weights, lambda: PO.load_win_place(tables=tb)):
        try:
            fn()
        except Exception:  # noqa: BLE001  空の表で後段が落ちるのは構わない（開いた年だけを見る）
            pass
    assert {p.split("/")[-2] for p in opened} == {str(y) for y in range(2016, 2025)}, opened   # 2016〜2024 年を開き、2025・2026 年は開かない


def test_groups():
    assert CR.group_of(0) == "新馬・未勝利" and CR.group_of(1) == "新馬・未勝利"
    assert CR.group_of(2) == "1勝・2勝" and CR.group_of(3) == "1勝・2勝"
    assert CR.group_of(4) == "3勝・OP" and CR.group_of(5) == "3勝・OP"   # L（リステッド）は OP と同じ 5
    assert CR.group_of(6) is None and CR.group_of(None) is None           # 重賞は単位から外す


def test_load_units_rejects_rows_after_2024_and_misaligned_levels(tmp_path, monkeypatch):
    monkeypatch.setattr(CR, "OUT", tmp_path)
    d = pd.DataFrame({"year": [2024, 2025], "cls": [2, 2], "track": ["良", "良"], "course": ["東京芝1600"] * 2, "race_id": ["a", "b"], "horse_id": ["x", "y"]})
    d.to_parquet(tmp_path / "frame.parquet")
    pd.DataFrame({"A01": ["1着", "1着"], "__race_id": ["a", "b"], "__horse_id": ["x", "y"]}).to_parquet(tmp_path / "levels.parquet")
    with pytest.raises(ValueError):
        CR.load_units()                                   # 2025 年の行がある
    d.assign(year=[2024, 2024]).to_parquet(tmp_path / "frame.parquet")
    pd.DataFrame({"A01": ["1着", "1着"], "__race_id": ["b", "a"], "__horse_id": ["y", "x"]}).to_parquet(tmp_path / "levels.parquet")
    with pytest.raises(ValueError):
        CR.load_units()                                   # frame と levels の行がずれている


def _toy(n_races=30, course="東京芝1600"):
    rng = np.random.default_rng(0)
    rows, levels = [], []
    for r in range(n_races):
        field = 10
        fin = list(rng.permutation(range(2, field + 1)))   # 馬番 1 がいつも 1 着、ほかは 2〜10 着（勝ち馬は 1 頭）
        for h in range(field):
            f = 1 if h == 0 else int(fin[h - 1])
            rows.append({"race_id": f"{course}{r}", "horse_id": f"h{r}_{h}", "year": 2018 + r % 7, "cls": 2, "track": "良", "course": course,
                         "group": "1勝・2勝", "race_class": "1勝クラス", "hn": h + 1, "fin": float(f), "field": field, "pay_ok": True,
                         "ret_tan": 5.0 if f == 1 else 0.0, "ret_fuku": 2.0 if f <= 3 else 0.0, "ret_tan_c": 5.0 if f == 1 else 0.0})
            levels.append({"A01": "1着" if h == 0 else "2-3着", "C01": "同距離"})
    return pd.DataFrame(rows), pd.DataFrame(levels)


def test_run_config_finds_buy_cell_and_scores_in_sample():
    d, L = _toy()
    r = CR.run_config(d, L, SE.Params(depth=1, n_min=2, min_races=10, kill_n=10), keep_cells=True)
    assert r["all"]["win"] == 1.0 and r["achieved_units"] == 1                          # 当てはめなので勝つ（過学習してよい）
    cells = r["cells"]["東京芝1600|1勝・2勝"]
    assert ["A01"] in [c["types"] for c in cells["buy"]] and all("n" in c and "roi_t" in c for c in cells["buy"])
    assert r["conf_2022_2024"]["races"] < r["all"]["races"]                             # 参考の期間は 2022〜2024 年だけ


def test_small_units_use_course_population_and_dedupe():
    d, L = _toy(n_races=8)
    r = CR.run_config(d, L, SE.Params(depth=2, n_min=2, min_races=20), keep_cells=True)
    assert r["per_unit"][0]["layer"] == "副"                                          # 8 回しかない単位は同じコースの全群で探す
    keys = [tuple(c["types"]) for c in r["cells"]["東京芝1600|1勝・2勝"]["buy"]]
    assert ("A01",) in keys and ("A01", "C01") not in keys                             # A01=1着 と A01=1着 ∧ C01=同距離 は同じ走の集まり → 1 つ


def test_achieved_needs_coverage():
    d, L = _toy(n_races=30)
    lv = L.assign(A01=np.where((d["hn"] == 1) & (d["race_id"] == "東京芝16000"), "1着", "2-3着"))   # 1 回だけ当たる条件
    r = CR.run_config(d, lv, SE.Params(n_min=1, min_races=10))
    assert r["per_unit"][0]["bet_races"] <= 1 and not r["per_unit"][0]["achieved"]   # 1 回だけ買って勝っても達成に数えない


def test_kill_needs_three_horses_and_skips_honmei():
    X = np.array([[0], [1], [1], [1], [1]])
    cells = {"buy": [{"ts": (0,), "cs": (0,), "rate": 1.0}], "kill": [{"ts": (0,), "cs": (1,)}]}
    race = np.array(["a"] * 5)
    hm, kl = SE.apply_cells(X, race, np.arange(1, 6), cells, "count")
    assert hm == {"a": 0} and kl.tolist() == [False, True, True, True, True]
    hm, kl = SE.apply_cells(X[:3], race[:3], np.arange(1, 4), cells, "count")
    assert not kl.any()                                                                # 2 頭だけなら消さない


def test_honmei_tie_goes_to_smaller_number():
    X = np.array([[0], [0], [1]])
    hm, _ = SE.apply_cells(X, np.array(["a"] * 3), np.array([5, 2, 1]), {"buy": [{"ts": (0,), "cs": (0,), "rate": 1.0}], "kill": []}, "count")
    assert hm == {"a": 1}                                                              # 同点は馬番の小さい方（2 番）


def test_cycles_html_even_midway(tmp_path):
    import json

    from src.scripts.docs import build_course_cycles_html as H
    L = {"note": "全レース版 <x>", "goal": CR.GOAL, "races": 10, "rows": 100,
         "cycles": [{"cycle": 1, "hyp": "基準 <b>", "params": {"name": "前日版"}, "adopted": True, "achieved_units": 2, "units": 5,
                     "all": {"win": 0.4, "roi_t": 1.5, "roi_t_capped": 1.2, "coverage": 0.5, "kill_coverage": 0.3},
                     "conf_2022_2024": {"win": 0.35, "roi_t": 1.1}, "closeness": 1.2, "seconds": 3}]}
    src = tmp_path / "c.json"
    src.write_text(json.dumps(L, ensure_ascii=False), encoding="utf-8")
    h = H.build(src, tmp_path / "c.html").read_text(encoding="utf-8")
    assert "途中" in h and f"1 / {len(CR.CYCLES)}" in h and "基準 &lt;b&gt;" in h and "全レース版 &lt;x&gt;" in h and "2025・2026 年は" in h


def test_levels_work_with_nullable_numeric_columns():
    """欠けありの数値型（馬体重が Int64 で欠けあり）でも、float に直してから水準を作れる。"""
    from src.research.graded import catalog as CAT
    d = pd.DataFrame({"wt": pd.array([480, None, 500], dtype="Int64"), "wtc": pd.array([2.0, None, -4.0], dtype="Float64")})
    t = CAT.BY_ID["G06"]
    with pytest.raises(ValueError):
        t.fn(d)                                                       # そのままだと落ちる（元の不具合）
    out = t.fn(CR._plain_numeric(d))
    assert out.iloc[0] == "480-519kg" and out.iloc[2] == "480-519kg"


def test_master_reuse_only_when_2024_and_same_settings(tmp_path):
    p = tmp_path / "master.parquet"
    assert not CR.master_ok(p)                                        # 無い
    pd.DataFrame({"year": [2016, 2024]}).to_parquet(p)
    assert not CR.master_ok(p)                                        # 作ったときの設定の記録が無い
    (tmp_path / "course_master.json").write_text('{"year_end": 2024, "keep_dnf": true}', encoding="utf-8")
    assert CR.master_ok(p)
    pd.DataFrame({"year": [2016, 2025]}).to_parquet(p)
    assert not CR.master_ok(p)                                        # 2025 年を含む（統計だけで判定し、行は読まない）


def test_metrics_counts_kill_races():
    Y = SE.y_matrix(pd.DataFrame({"fin": [1, 4, 5, 6, 2, 7], "field": [6] * 6, "pay_ok": [True] * 6, "ret_tan": [3.0, 0, 0, 0, 0, 0],
                                  "ret_fuku": [1.5, 0, 0, 0, 1.2, 0], "ret_tan_c": [3.0, 0, 0, 0, 0, 0]}))
    race = np.array(["a", "a", "a", "a", "b", "b"])
    m = SE.metrics(Y, race, np.array([0]), np.array([1, 2, 3]), 2)
    assert m["win"] == 1.0 and m["roi_t"] == 3.0 and m["kill_races"] == 1 and m["kill_per_race"] == 3.0 and m["coverage"] == 0.5


def test_evaluate_frozen_matches_names_not_codes():
    """凍結した条件は型・水準の名前で照らす（評価の表で水準の番号の振り方が変わっても同じ馬を選ぶ）。"""
    d, L = _toy(n_races=6)
    d["year"] = 2025
    P = {"params": SE.params_dict(SE.Params(min_races=1))}
    cells = {"東京芝1600|1勝・2勝": {"buy": [{"types": ["A01"], "levels": ["1着"], "rate": 1.0}], "kill": [{"types": ["A01"], "levels": ["2-3着"]}]}}
    r = CR.evaluate_frozen(d, L, P, cells)
    assert r["all"]["win"] == 1.0 and r["all"]["bet_races"] == 6 and r["all"]["kill_races"] == 6
    r2 = CR.evaluate_frozen(d.assign(course="京都芝1600"), L, P, cells)
    assert r2["all"]["bet_races"] == 0 and r2["per_unit"][0]["no_rule"]               # 2016〜2024 年に無い単位は買わない


def test_playbook_html_from_best(tmp_path):
    import json

    from src.scripts.docs import build_course_playbook_html as H
    d, L = _toy()
    best = CR.run_config(d, L, SE.Params(depth=2, min_races=10), keep_cells=True)
    src = tmp_path / "b.json"
    src.write_text(json.dumps(best, ensure_ascii=False, default=float), encoding="utf-8")
    h = H.build(src, tmp_path / "p.html", None).read_text(encoding="utf-8")
    assert "前走で 1着だった馬" in h and "評価（2025・2026 年）はまだ" in h and "function setupCourse" in h and 'data-course="東京芝1600"' in h


def test_evaluate_frozen_weight_rate_groups4_and_missing_type():
    d, L = _toy(n_races=6)
    d["year"] = 2025
    d.loc[d["race_id"] == "東京芝16000", "race_class"] = "2歳新馬"
    d["group"] = "新馬・未勝利"
    P = {"params": SE.params_dict(SE.Params(min_races=1, groups=4, weight="rate"))}
    cells = {"東京芝1600|新馬・未勝利": {"buy": [{"types": ["A01"], "levels": ["1着"], "rate": 0.9}], "kill": []},
             "東京芝1600|新馬": {"buy": [{"types": ["A01"], "levels": ["無い水準"], "rate": 0.9}], "kill": []}}
    r = CR.evaluate_frozen(d, L, P, cells)
    assert r["all"]["bet_races"] == 5 and r["unmatched_levels"] == 1                 # 新馬の 1 回は別の単位・その単位の水準は評価の表に無い
    with pytest.raises(ValueError):
        CR.evaluate_frozen(d, L, P, {"東京芝1600|新馬・未勝利": {"buy": [{"types": ["Z99"], "levels": ["x"]}], "kill": []}})


def test_eval_and_freeze_guards(tmp_path, monkeypatch):
    monkeypatch.setattr(CR, "OUT", tmp_path)
    monkeypatch.setattr(CR, "EVAL_MARK", tmp_path / "EVALUATED_COURSE.json")
    monkeypatch.setattr(CR, "PREREG", tmp_path / "prereg_course.json")
    monkeypatch.delenv("COURSE_EVAL_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        CR._eval_inner()                                                              # トークンが無ければ流さない
    (tmp_path / "course_best.json").write_text('{"params": {"a": 1}, "cells": {}}', encoding="utf-8")
    (tmp_path / "course_cycles.json").write_text('{"cycles": [{"cycle": 1, "params": {"a": 1}}], "best_cycle": 1}', encoding="utf-8")
    with pytest.raises(SystemExit):
        CR.cmd_freeze()                                                               # サイクルが最後まで回っていない
    CR.EVAL_MARK.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        CR.cmd_freeze()                                                               # 評価済みの印があれば凍結し直さない


def test_frame_and_levels_before_filtering_keeps_rate_cutoffs():
    """水準は単位に絞る前の表で作る: 絞った後に作ると、境目の基準（2017 年の重賞の行）が無くなる。"""
    from src.research.graded import catalog as CAT
    assert "2017" in __import__("inspect").getsource(CAT._ref) and "G1" in __import__("inspect").getsource(CAT._ref)
    src = __import__("inspect").getsource(CR._eval_inner)
    assert src.index("frame_and_levels") < src.index("keep =")                       # 前処理と水準 → その後で 2025 年以降・良馬場に絞る
