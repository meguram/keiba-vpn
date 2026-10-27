"""想定めぐ指数の HEM 推論: 学習時の計算と一致すること・先読みが無いこと・旧版へ戻せること。"""

import numpy as np
import pandas as pd
import pytest

from src.pipeline.inference import expected_megu as em
from src.pipeline.inference import hem_megu_infer as hi
from src.scripts.data import hem_megu as hm

CELLS = {"glob": 40.0, "fine": {203 * 1: 41.0}, "mid": {21: 42.0}}


def _hem(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    coef = np.concatenate([[3.0], rng.normal(0.3, 0.2, len(hm.STACK_COLS))])
    return {"version": "t", "params": dict(hm.DEFAULT) | {"N": 6}, "stack_cols": list(hm.STACK_COLS), "coef": coef.tolist(), "coef_arr": coef,
            "cells": CELLS, "resid_sd": {"n1": 14.0, "n2": 13.0, "n3_4": 12.0, "n5p": 11.0}}


def _history(seed: int = 1, n_horse: int = 6, runs: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for hid in range(n_horse):
        ability = rng.normal(0, 8)
        for k in range(runs):
            rows.append({"race_id": f"R{hid}_{k}", "horse_id": f"H{hid}", "jockey_id": f"J{hid % 3}", "jw": 54.0 + (k % 3), "dnum": 20240101 + k * 45 + hid,
                         "surf": int(k % 2), "dist": 1200.0 + 200 * (k % 4), "lvl": 2 + (k % 2), "age": 3, "y": 40 + ability + rng.normal(0, 10), "low": bool(k == 2)})
    h = pd.DataFrame(rows)
    h["year"] = h["dnum"] // 10000
    return h


def _runs_of(h: pd.DataFrame, hid: str) -> list:
    g = h[h["horse_id"] == hid]
    return [(str(int(r.dnum)), float(r.y), bool(r.low), "芝" if r.surf == 1 else "ダ", float(r.dist), "東京", int(r.lvl), float(r.jw), int(r.age), r.race_id)
            for r in g.itertuples()]


TODAY = {"surface": "ダ", "distance": 1400.0, "venue": "東京", "level": 3, "weight": 55.0, "age": 3}


def test_inference_matches_training_computation_exactly():
    h, hem = _history(), _hem()
    hist = hm.Hist(h, CELLS)
    jt = hm.jockey_table(hist, hem["params"])
    race_dn = 20250601
    T = pd.DataFrame({"race_id": [f"X{i}" for i in range(6)], "horse_id": [f"H{i}" for i in range(6)], "jockey_id": [f"J{i % 3}" for i in range(6)],
                      "dnum": race_dn, "surf": 0, "dist": 1400.0, "lvl": 3, "age": 3, "jw": 55.0})
    F = hm.features(hist, T, hem["params"], jt)
    want = hm.predict_stack(F, hem["coef_arr"], hem["stack_cols"])
    table = {str(j): (g["dnum"].to_numpy(), g["cse"].to_numpy(), g["ccnt"].to_numpy()) for j, g in jt.groupby("jockey_id")}
    runs = {i: _runs_of(h, f"H{i}") for i in range(6)}
    got = hi.predict_horses(runs, str(race_dn), {i: TODAY for i in range(6)}, {i: f"J{i % 3}" for i in range(6)}, hem, table)
    assert all(got[i] is not None for i in range(6))
    assert np.allclose([got[i]["pred"] for i in range(6)], want, atol=1e-9)


def test_no_lookahead_runs_on_or_after_the_race_day_are_ignored():
    h, hem = _history(), _hem()
    runs = _runs_of(h, "H0")
    race_date = runs[4][0]                                   # 5 走目の当日
    base = hi.predict_horses({0: runs}, race_date, {0: TODAY}, {0: None}, hem, {})[0]
    shifted = [(r[0], r[1] + (500.0 if r[0] >= race_date else 0.0)) + tuple(r[2:]) for r in runs]
    after = hi.predict_horses({0: shifted}, race_date, {0: TODAY}, {0: None}, hem, {})[0]
    assert base["pred"] == pytest.approx(after["pred"]) and base["n"] == 4


def test_horse_without_usable_runs_or_without_conditions_is_none():
    hem = _hem()
    assert hi.predict_horses({0: []}, "20250601", {0: TODAY}, {0: None}, hem, {})[0] is None
    assert hi.predict_horses({0: _runs_of(_history(), "H0")}, "20250601", {0: None}, {0: None}, hem, {})[0] is None
    assert hi.predict_horses({0: _runs_of(_history(), "H0")}, "20230101", {0: TODAY}, {0: None}, hem, {})[0] is None      # 全部が未来の走


def test_fewer_runs_shrink_towards_the_cell_mean():
    hem = _hem()
    strong = [("20250101", 90.0, False, "ダ", 1400.0, "東京", 3, 55.0, 3, "a")]
    many = [(f"2025{m:02d}01", 90.0, False, "ダ", 1400.0, "東京", 3, 55.0, 3, f"b{m}") for m in range(1, 10)]
    one = hi.predict_horses({0: strong}, "20251001", {0: TODAY}, {0: None}, hem, {})[0]
    nine = hi.predict_horses({0: many}, "20251001", {0: TODAY}, {0: None}, hem, {})[0]
    assert one["contrib"]["dhat"] < nine["contrib"]["dhat"]


def test_adjust_items_add_up_to_value_minus_base_and_range_follows_run_count():
    hem = _hem()
    out = hi.predict_horses({0: _runs_of(_history(), "H0")}, "20250601", {0: TODAY}, {0: "J0"}, hem, {})[0]
    base = 41.0
    items = hi.adjust_items(out, base, out["pred"], min_abs=0.0)
    assert sum(i["delta"] for i in items) == pytest.approx(out["pred"] - base, abs=0.06)
    assert hi.range_for(hem, 1) == 14.0 and hi.range_for(hem, 3) == 12.0 and hi.range_for(hem, 12) == 11.0


def test_expected_from_runs_uses_hem_value_range_and_keeps_response_keys():
    hem = _hem()
    runs = _runs_of(_history(), "H1")
    out = hi.predict_horses({0: runs}, "20250601", {0: TODAY}, {0: None}, hem, {})[0]
    legacy = em.expected_from_runs(runs, "20250601", TODAY, None)
    new = em.expected_from_runs(runs, "20250601", TODAY, None, hem, out)
    assert set(new) == set(legacy)                                      # 画面が読むキーは変わらない
    assert new["value"] == round(out["pred"], 1) and new["base_value"] == legacy["base_value"]
    assert new["range"] == max(em.MIN_RANGE, hi.range_for(hem, out["n"]))


def test_legacy_switch_and_missing_file_fall_back(monkeypatch, tmp_path):
    monkeypatch.setenv("KEIBA_EXPECTED_MEGU", "legacy")
    assert hi.load_hem() is None and hi.enabled() is False
    monkeypatch.delenv("KEIBA_EXPECTED_MEGU")
    assert hi.load_hem(tmp_path / "none.json") is None


def test_attach_replaces_value_with_hem_and_legacy_env_restores_it(tmp_path, monkeypatch):
    em._cache.clear()
    base = tmp_path / "tables"
    for y, rows, res in ((2025, [("202501010101", "A", 55.0, "2025-01-10"), ("202502010101", "A", 61.0, "2025-02-10")], None),):
        (base / str(y)).mkdir(parents=True)
        pd.DataFrame([{"race_id": r, "horse_id": h, "megu_index_v3a_abs_robust": v, "megu_v3a_status": "ok", "megu_v3a_low_confidence": False} for r, h, v, d in rows]
                     ).to_parquet(base / str(y) / "megu_index_flat.parquet")
        pd.DataFrame([{"race_id": r, "horse_id": h, "date": d, "surface": "ダ1400", "distance": 1400, "grade": "1勝", "venue": "東京", "jockey_weight": 55.0,
                       "sex_age": "牡3"} for r, h, v, d in rows]).to_parquet(base / str(y) / "race_result_flat.parquet")
    import json

    hem = _hem()
    p = tmp_path / "hem.json"
    doc = {k: v for k, v in hem.items() if k != "coef_arr"}
    doc["cells"] = {"glob": 40.0, "fine": {"203": 41.0}, "mid": {"21": 42.0}}
    p.write_text(json.dumps(doc), encoding="utf-8")
    monkeypatch.setattr(hi, "HEM_PATH", p)
    monkeypatch.setattr(hi, "STATE_PATH", tmp_path / "no_state.parquet")
    card = {"surface": "ダ", "distance": 1400, "venue": "東京", "grade": "2勝", "entries": [{"horse_id": "A", "jockey_id": "J1", "sex_age": "牡3", "jockey_weight": 55.0}]}
    resp = {"predictions": [{"horse_number": 1, "horse_id": "A"}]}
    new = em.attach_expected_megu(resp, "2025-05-01", base=base, card=card)
    ex = new["predictions"][0]["expected_megu"]
    assert new["expected_megu_meta"]["model"]["version"] == "t" and ex["range"] == 13.0       # 2 走 → n2 の区分
    assert ex["adjust"] is not None and "階層" in new["expected_megu_meta"]["method"]
    monkeypatch.setenv("KEIBA_EXPECTED_MEGU", "legacy")
    old = em.attach_expected_megu({"predictions": [{"horse_number": 1, "horse_id": "A"}]}, "2025-05-01", base=base, card=card)
    assert "階層" not in old["expected_megu_meta"]["method"]
